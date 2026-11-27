# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The multi-rank coordinator without MPI or devices: the owner election over the ranks' parts, the merge of the ranks'
results into one index, the phase protocol over an in-process transport (a failing rank stops every rank with its
number; a failing collective is fatal and ends the exchanges), and one load over four in-process ranks, its device
faked."""

import dataclasses
import gc
import json
import os
import re
import threading
from types import SimpleNamespace

import pytest
import ttnn
from dcp_oracle import FakeCheckpoint, Loader, arange, assert_shards_equal, expected_shards, info, tp_split_of, world

import ttml.distributed_checkpointing as dcp
from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing import _coordinator, _load, _save, _state_dict
from ttml.distributed_checkpointing._coordinator import (
    Election,
    Group,
    LoopbackTransport,
    PlanPart,
    RankRecord,
    ResultPart,
    checked_payload,
    merge_plan_parts,
    merge_results,
)
from ttml.distributed_checkpointing._errors import (
    CheckpointError,
    CheckpointFatalError,
    IncompleteCheckpointError,
    MissingKeyError,
)
from ttml.distributed_checkpointing._index import ChunkRecord, Entry, FileRecord, data_file_name
from ttml.distributed_checkpointing._layouts import Identity, Pad
from ttml.distributed_checkpointing._planner import TensorInfo, create_save_plan
from ttml.distributed_checkpointing._state_dict import (
    LiveTensor,
    check_param_groups,
    is_structural_scalar,
    restore_scalars,
)
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords
from ttml.distributed_checkpointing._train_state import this_world_rank


def _label(placements, shape):
    return Label(tuple(placements), tuple(shape), tuple(dist_coords(shape)))


def _info(key, placements, mesh_shape, local_shape, role="param"):
    label = _label(placements, mesh_shape)
    return TensorInfo(
        key, role, "BF16", "TILE", label, tuple(local_shape), Identity(), tp_split=tp_split_of(label, len(local_shape))
    )


def _record(rank, local_coords, mesh_id=0, shape=(1, 4)):
    return RankRecord(rank, rank, mesh_id, list(shape), ["dp", "tp"], [tuple(c) for c in local_coords])


R, S = Placement.replicate(), Placement.shard


def _big_mesh_parts():
    """Two ranks co-owning a [1, 4] mesh: a column-sharded weight, a replicated bias, both on every rank."""
    infos = [_info("model/w", [R, S(3)], (1, 4), (1, 1, 32, 16)), _info("model/b", [R, R], (1, 4), (1, 1, 1, 64))]
    return [
        PlanPart(_record(0, [(0, 0), (0, 1)]), infos),
        PlanPart(_record(1, [(0, 2), (0, 3)]), infos),
    ]


def test_local_coords_of_an_l_shaped_host(monkeypatch):
    """get_local_mesh_coord_range is the bounding box of this host's devices. An L-shaped set of them spans another
    host's device too, which a save would otherwise be elected to write and a load would hand to
    from_local_host_shards."""
    mine = {(0, 0), (0, 1), (1, 0)}

    class View:
        def get_local_mesh_coord_range(self):
            return [(r, c) for r in range(2) for c in range(2)]  # the box, with (1, 1) on another host

        def is_local(self, coord):
            return tuple(coord) in mine

    class Device:
        def get_view(self):
            return View()

    monkeypatch.setattr(_state_dict, "mesh_device", lambda: Device())
    assert _state_dict.local_coords() == [(0, 0), (0, 1), (1, 0)]


def test_the_election_gives_every_box_one_owner_that_holds_it():
    election = merge_plan_parts(_big_mesh_parts())
    assert set(election.owners) == {"model/w", "model/b"} and election.keys == 2
    w = election.owners["model/w"]
    assert len(w) == 4, "a column-sharded weight has four unique boxes"
    for group, (rank, coord) in w.items():
        assert coord == (0, group[0]) and rank == (0 if coord[1] < 2 else 1), "each box is written where it lives"
    (bias_owner,) = election.owners["model/b"].values()
    assert bias_owner in {(0, (0, 0)), (0, (0, 1)), (1, (0, 2)), (1, (0, 3))}
    # The election every rank decodes from rank 0's broadcast is the one rank 0 made.
    assert Election.from_bytes(election.to_bytes()) == election


def test_the_election_evens_out_the_bytes_each_rank_and_device_writes():
    """A few large replicated keys decide where most bytes land. The boxes one holder has are charged first, then the
    others, largest first, go to the holder on the least-loaded rank, its device's bytes breaking a tie, so two ranks
    end up writing equal shares and their devices similar ones."""
    coords = {0: [(0, 0), (0, 1)], 1: [(0, 2), (0, 3)]}
    sharded = _info("model/w", [R, S(3)], (1, 4), (1, 1, 32, 16))  # one box per device: forced
    rows = [512, 384, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16]  # replicated keys of very different sizes
    replicated = [_info(f"model/r{i}", [R, R], (1, 4), (1, 1, n, 64)) for i, n in enumerate(rows)]
    parts = [PlanPart(_record(r, coords[r]), [sharded, *replicated]) for r in (0, 1)]
    election = merge_plan_parts(parts)
    assert election == merge_plan_parts(list(reversed(parts))), "the same parts always elect the same owners"
    rank_bytes, device_bytes = {0: 0, 1: 0}, {}
    for info in [sharded, *replicated]:
        for rank, coord in election.owners[info.key].values():
            nbytes = 2 * info.local_shape[2] * info.local_shape[3]
            rank_bytes[rank] += nbytes
            device_bytes[coord] = device_bytes.get(coord, 0) + nbytes
    assert rank_bytes[0] == rank_bytes[1], rank_bytes
    assert len(device_bytes) == 4 and max(device_bytes.values()) <= 1.25 * min(device_bytes.values()), device_bytes
    # A key one rank holds alone is that rank's, whatever it already writes.
    alone = _info("model/alone", [R, R], (1, 4), (1, 1, 512, 64))
    election = merge_plan_parts([PlanPart(_record(0, coords[0]), [alone]), PlanPart(_record(1, coords[1]), [])])
    assert {r for r, _ in election.owners["model/alone"].values()} == {0}


def test_equal_boxes_rotate_over_the_devices_of_tied_ranks():
    """Equal boxes, a TP key's columns, leave the ranks tied after every round. Were the lower rank always to take
    the tie, each rank would write the same columns of every key, all on the few devices holding them, and most of
    its writer threads would idle: the devices' bytes break the tie instead, so every device writes one box."""
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    keys = [_info(f"model/w{i}", [R, S(3)], (2, 2), (1, 1, 32, 64)) for i in range(4)]  # columns on the second axis
    parts = [PlanPart(_record(r, coords, mesh_id=r, shape=(2, 2)), keys) for r in (0, 1)]
    election = merge_plan_parts(parts)
    written = {}
    for key in keys:
        for owner in election.owners[key.key].values():
            written[owner] = written.get(owner, 0) + 1
    assert sorted(written) == [(r, c) for r in (0, 1) for c in coords], written
    assert set(written.values()) == {1}, written


def test_each_rank_plans_only_its_owned_chunks_and_files():
    parts = _big_mesh_parts()
    election = merge_plan_parts(parts)
    plans = {}
    for part in parts:
        rank = part.record.rank
        plans[rank] = create_save_plan(
            part.infos, file_for_coord=lambda c, r=rank: data_file_name(r, 0, c), owner_of=election.owner_of(rank)
        )
    for rank, plan in plans.items():
        assert all(
            it.chunk.mesh_coord in set(parts[rank].record.local_coords) for its in plan.items.values() for it in its
        )
        assert [f.coord for f in plan.files] == sorted(
            {it.chunk.mesh_coord for its in plan.items.values() for it in its}
        )
        assert all(f.path.startswith(f"data/w{rank:03d}_") for f in plan.files)
    assert len(plans[0].items["model/w"]) == 2 and len(plans[1].items["model/w"]) == 2
    assert len(plans[0].items["model/b"]) + len(plans[1].items["model/b"]) == 1
    # A key another rank writes entirely has no entry in this rank's part.
    bias_writer = election.owners["model/b"][()][0]
    other = 1 - bias_writer

    def entries(plan):
        return plan.entries({(it.key, it.file): 0 for its in plan.items.values() for it in its})

    assert "model/b" not in entries(plans[other]) and "model/b" in entries(plans[bias_writer])


def test_ranks_must_cover_the_label(expect_error):
    """The ranks must hold every coordinate of a label between them."""
    infos = [_info("model/w", [R, S(3)], (1, 4), (1, 1, 32, 16))]
    with expect_error(CheckpointError, "no rank holds"):
        merge_plan_parts([PlanPart(_record(0, [(0, 0), (0, 1)]), infos), PlanPart(_record(1, [(0, 2)]), infos)])


# One change per field _same_description compares; each is made alone, the other fields left as rank 0 has them.
DESCRIPTION_CHANGES = {
    "role": {"role": "buffer"},
    "dtype": {"dtype": "F32"},
    "label": {"label": _label([R, S(2)], (1, 4))},
    "local_shape": {"local_shape": (1, 1, 32, 32)},
    "transform": {"transform": Pad(2, logical=30, padded=32)},
    "param_ref": {"param_ref": "model/v"},
    "state_path": {"state_path": ("exp_avg",)},
    "tp_split": {"tp_split": (1, 1, 1, 1)},
}


@pytest.mark.parametrize("field", sorted(DESCRIPTION_CHANGES))
def test_ranks_must_describe_a_shared_key_alike(field, expect_error):
    """Ranks holding one key must describe it alike: a difference in any one field of the description is refused."""
    mine = _info("model/w", [R, S(3)], (1, 4), (1, 1, 32, 16))
    theirs = dataclasses.replace(mine, **DESCRIPTION_CHANGES[field])
    with expect_error(CheckpointError, r"^model/w: rank 1 describes it differently from rank 0 "):
        merge_plan_parts(
            [PlanPart(_record(0, [(0, 0), (0, 1)]), [mine]), PlanPart(_record(1, [(0, 2), (0, 3)]), [theirs])]
        )


def test_many_replicated_keys_spread_over_every_device_and_are_written_where_elected():
    """Many equal replicated keys: every (rank, device) pair writes the same number of them, and each rank's plan
    writes exactly the boxes it was elected for, each from the elected device's chunk."""
    coords = {0: [(0, 0), (0, 1)], 1: [(0, 2), (0, 3)]}
    infos = [_info(f"model/r{i:02d}", [R, R], (1, 4), (1, 1, 32, 64)) for i in range(64)]
    parts = [PlanPart(_record(r, coords[r]), infos) for r in (0, 1)]
    election = merge_plan_parts(parts)
    writers = {}
    for groups in election.owners.values():
        for owner in groups.values():
            writers[owner] = writers.get(owner, 0) + 1
    assert writers == {(r, c): 16 for r in coords for c in coords[r]}, writers
    for part in parts:
        rank = part.record.rank
        plan = create_save_plan(
            part.infos, file_for_coord=lambda c, r=rank: data_file_name(r, 0, c), owner_of=election.owner_of(rank)
        )
        planned = {(key, it.chunk.mesh_coord) for key, its in plan.items.items() for it in its}
        elected = {(key, c) for key, groups in election.owners.items() for r, c in groups.values() if r == rank}
        assert planned == elected, rank


def test_a_plan_part_stores_each_label_once():
    """Keys of one layout share a label: a plan part stores each label once, in a table its tensors point into, and
    decodes back to the part it was made of: its rank record (the coordinates the rank holds among them), its
    hashes and every field of its tensors as sent (a TP split, which the index records and load compares, among
    them), with the cyclic garbage collector back on afterwards."""
    infos = [_info(f"model/w{i}", [R, S(3)], (1, 4), (1, 1, 32, 16)) for i in range(50)]
    infos.append(_info("model/b", [R, R], (1, 4), (1, 1, 1, 64)))
    split = _info("model/qkv_linear/w", [R, S(2)], (1, 4), (1, 1, 32, 64))
    infos.append(dataclasses.replace(split, tp_split=(1, 1, 4, 1)))
    part = PlanPart(_record(0, [(0, 0), (0, 1)]), infos, scheduler_hash=11, dataloader_hash=13)
    data = part.to_bytes()
    assert len(json.loads(data)["labels"]) == 3
    assert PlanPart.from_bytes(data) == part and gc.isenabled()


def test_multi_mesh_ranks_with_their_own_keys_and_one_shared_key():
    """Two ranks, one [1, 2] mesh each (the same coordinates): disjoint keys, plus a replicated key both hold."""
    stage0 = _info("model/stage0/w", [R, S(3)], (1, 2), (1, 1, 32, 32))
    stage1 = _info("model/stage1/w", [R, S(3)], (1, 2), (1, 1, 32, 32))
    shared = _info("model/norm", [R, R], (1, 2), (1, 1, 1, 64))
    parts = [
        PlanPart(_record(0, [(0, 0), (0, 1)], mesh_id=0, shape=(1, 2)), [stage0, shared]),
        PlanPart(_record(1, [(0, 0), (0, 1)], mesh_id=1, shape=(1, 2)), [stage1, shared]),
    ]
    election = merge_plan_parts(parts)
    assert {r for r, _ in election.owners["model/stage0/w"].values()} == {0}
    assert {r for r, _ in election.owners["model/stage1/w"].values()} == {1}
    assert len(election.owners["model/norm"]) == 1, "one box, held by four (rank, coordinate) pairs"


def test_every_ranks_live_names_reach_the_index(expect_error):
    """A layer both ranks hold, tied to another name on one rank only: the index maps that name too, though the other
    rank writes the layer, so the rank holding it under the extra name loads it through that name."""
    stage0 = _info("model/stage0/w", [R, S(3)], (1, 2), (1, 1, 32, 32))
    stage1 = _info("model/stage1/w", [R, S(3)], (1, 2), (1, 1, 32, 32))
    norm = _info("model/norm", [R, R], (1, 2), (1, 1, 1, 64))

    def parts(norm_of):
        return [
            PlanPart(_record(r, [(0, 0), (0, 1)], mesh_id=r, shape=(1, 2)), [stage, norm_of(r)])
            for r, stage in ((0, stage0), (1, stage1))
        ]

    writer = merge_plan_parts(parts(lambda r: norm)).owners["model/norm"][()][0]  # names do not move the owner
    tied = dataclasses.replace(norm, live_keys=("model/head/norm", "model/norm"))
    election = merge_plan_parts(parts(lambda r: tied if r != writer else norm))
    assert election.owners["model/norm"][()][0] == writer
    assert election.live_keys["model/norm"] == ("model/head/norm", "model/norm")
    assert Election.from_bytes(election.to_bytes()) == election

    def result(rank, own_key):
        files = [FileRecord(f"data/w{rank:03d}_m{rank}_d0-{c}.safetensors", 5120, rank, rank, (0, c)) for c in (0, 1)]
        entries = {own_key: _entry(own_key, [_chunk(0, 0), _chunk(1, 1)])}
        if rank == writer:
            bias = ChunkRecord(Box((0, 0, 0, 0), (1, 1, 1, 64)), 0, 8192, 128, 9)
            entries["model/norm"] = _entry("model/norm", [bias], (1, 1, 1, 64))
        return _result(rank, files, entries, f"gen{rank}")

    index, _ = merge_results([result(0, "model/stage0/w"), result(1, "model/stage1/w")], election, "step_0000001")
    assert index.entries["model/norm"].live_keys == ("model/head/norm", "model/norm")
    assert index.live_key_map()["model/head/norm"] == "model/norm"

    # A name one rank uses for a key another rank holds under that very name cannot map to both.
    clash = dataclasses.replace(norm, live_keys=("model/norm", "model/stage0/w"))
    with expect_error(CheckpointError, r"model/stage0/w names model/norm on rank 1 but model/stage0/w on rank 0"):
        merge_plan_parts(parts(lambda r: clash if r == 1 else norm))


def _result(rank, files, entries, rng_text):
    return ResultPart(
        _record(rank, [(0, 2 * rank), (0, 2 * rank + 1)]),
        files,
        entries,
        {"format": 1, "ranks": {str(rank): {"ttml": rng_text}}},
        scalars={"optimizer/steps": 3} if rank == 0 else {},
        header={"step": 3} if rank == 0 else {},
        train_state={"scheduler": {"m_last_step": 3}} if rank == 0 else {},
    )


def _entry(key, chunks, global_shape=(1, 1, 32, 64)):
    label = _label([R, S(3)], (1, 4))
    return Entry(
        key=key,
        role="param",
        dtype="BF16",
        layout="TILE",
        global_shape=global_shape,
        canonical_shape=global_shape,
        transform=Identity(),
        label=label,
        live_keys=(key,),
        chunks=chunks,
        tp_split=tp_split_of(label, len(global_shape)),
    )


def _chunk(col, file):
    """Column ``col`` of ``model/w`` (a [1, 1, 32, 16] box) as written by device (0, ``col``) into its rank's file
    ``file``."""
    return ChunkRecord(Box((0, 0, 0, 16 * col), (1, 1, 32, 16)), file, 4096, 1024, 7)


def _big_mesh_save():
    """The election over ``_big_mesh_parts`` and the two result parts a save on it reports: each rank wrote the two
    columns of ``model/w`` its devices hold, one file per device, and rank 1 also wrote the replicated bias, so the
    parts hold exactly one chunk per elected box."""
    election = merge_plan_parts(_big_mesh_parts())
    r0 = _result(
        0,
        [
            FileRecord("data/w000_m0_d0-0.safetensors", 5120, 0, 0, (0, 0)),
            FileRecord("data/w000_m0_d0-1.safetensors", 5120, 0, 0, (0, 1)),
        ],
        {"model/w": _entry("model/w", [_chunk(0, 0), _chunk(1, 1)])},
        "gen0",
    )
    bias = ChunkRecord(Box((0, 0, 0, 0), (1, 1, 1, 64)), 1, 8192, 128, 9)
    r1 = _result(
        1,
        [
            FileRecord("data/w001_m0_d0-2.safetensors", 5120, 1, 0, (0, 2)),
            FileRecord("data/w001_m0_d0-3.safetensors", 5248, 1, 0, (0, 3)),
        ],
        {
            "model/w": _entry("model/w", [_chunk(2, 0), _chunk(3, 1)]),
            "model/b": _entry("model/b", [bias], (1, 1, 1, 64)),
        },
        "gen1",
    )
    return election, r0, r1


def test_merge_results_builds_one_index(expect_error):
    """merge_results builds one index, with each rank's file indexes shifted, and keeps every rank's generators."""
    election, r0, r1 = _big_mesh_save()
    index, header = merge_results([r1, r0], election, "step_0000003")
    assert [f.path for f in index.files] == [
        "data/w000_m0_d0-0.safetensors",
        "data/w000_m0_d0-1.safetensors",
        "data/w001_m0_d0-2.safetensors",
        "data/w001_m0_d0-3.safetensors",
    ]
    assert [c.file for c in index.entries["model/w"].chunks] == [0, 1, 2, 3], "rank 1's file indexes are shifted"
    assert index.entries["model/b"].chunks[0].file == 3
    assert index.world["group_size"] == 2 and [r["rank"] for r in index.world["ranks"]] == [0, 1]
    assert index.scalars == {"optimizer/steps": 3}
    assert header["user"] == {"step": 3} and header["train_state"]["scheduler"] == {"m_last_step": 3}
    assert header["train_state"]["rng"]["ranks"] == {"0": {"ttml": "gen0"}, "1": {"ttml": "gen1"}}
    assert [len(p.entries["model/w"].chunks) for p in (r0, r1)] == [2, 2], "the parts are left as they were"
    assert [c.file for c in r1.entries["model/w"].chunks] == [0, 1], "rank 1's records are renumbered as copies"
    assert gc.isenabled(), "the cyclic GC, paused for the merge, is back on"
    # Parts travel as bytes without loss.
    assert ResultPart.from_bytes(r1.to_bytes()).to_bytes() == r1.to_bytes()


def _differently(entry, **changes):
    return dataclasses.replace(entry, chunks=list(entry.chunks), **changes)


MERGE_REFUSALS = {
    # rank 1 reports one of its two columns of model/w
    "too_few_chunks": (
        lambda r1: {"model/w": _entry("model/w", [_chunk(2, 0)]), "model/b": r1.entries["model/b"]},
        r"model/w: 3 chunk\(s\) reported by the ranks, 4 elected",
    ),
    # rank 1 reports a column twice
    "too_many_chunks": (
        lambda r1: {
            "model/w": _entry("model/w", [_chunk(2, 0), _chunk(3, 1), _chunk(3, 1)]),
            "model/b": r1.entries["model/b"],
        },
        r"model/w: 5 chunk\(s\) reported by the ranks, 4 elected",
    ),
    # rank 1 was elected for the bias but reports no entry for it
    "unreported_key": (
        lambda r1: {"model/w": r1.entries["model/w"]},
        r"model/b: 0 chunk\(s\) reported by the ranks, 1 elected",
    ),
    **{
        f"different_{name}": (
            lambda r1, change=change: {
                "model/w": _differently(r1.entries["model/w"], **change),
                "model/b": r1.entries["model/b"],
            },
            r"model/w: rank 1 reports a different entry than earlier ranks",
        )
        for name, change in (
            ("role", {"role": "buffer"}),
            ("dtype", {"dtype": "F32"}),
            ("global_shape", {"global_shape": (1, 1, 32, 128)}),
            ("label", {"label": _label([R, S(2)], (1, 4))}),
            ("tp_split", {"tp_split": (1, 1, 1, 1)}),
        )
    },
}


@pytest.mark.parametrize("case", sorted(MERGE_REFUSALS))
def test_merge_results_refuses_parts_that_do_not_match_the_election(case, expect_error):
    """The index must hold exactly one chunk per elected box, and the ranks must report a key alike. Each case starts
    from a fresh save, so no case sees entries another merge extended."""
    entries_of, message = MERGE_REFUSALS[case]
    election, r0, r1 = _big_mesh_save()
    with expect_error(CheckpointError, f"^{message}$"):
        merge_results([r0, dataclasses.replace(r1, entries=entries_of(r1))], election, "x")


class _Fake:
    """An in-process transport: ``size`` threads share a barrier for allgathers and queues for point-to-point."""

    def __init__(self, size):
        self.size = size
        self.values = [None] * size
        self.barrier = threading.Barrier(size)
        self.boxes = {(s, d): [] for s in range(size) for d in range(size)}
        self.cv = threading.Condition()
        self.die = set()

    def transport(self, rank):
        fake = self

        class T:
            def __init__(self):
                self.broken = None
                self.rank = rank
                self.size = fake.size
                self.world_rank = rank

            def allgather_int(self, value):
                if rank in fake.die:
                    raise RuntimeError("MPI_ERR_PROC_FAILED")
                fake.values[rank] = int(value)
                fake.barrier.wait()
                out = list(fake.values)
                fake.barrier.wait()
                return out

            def send_bytes(self, data, dest):
                with fake.cv:
                    fake.boxes[(rank, dest)].append(bytes(data))
                    fake.cv.notify_all()

            def recv_bytes(self, size, source):
                with fake.cv:
                    fake.cv.wait_for(lambda: fake.boxes[(source, rank)] or source in fake.die)
                    if not fake.boxes[(source, rank)]:  # the sender died: what ULFM reports to the receiver
                        raise RuntimeError("MPI_ERR_PROC_FAILED")
                    data = fake.boxes[(source, rank)].pop(0)
                assert len(data) == size
                return data

        return T()


def _run_ranks(size, body):
    fake = _Fake(size)
    results, errors = {}, {}

    def run(rank):
        try:
            results[rank] = body(Group(fake.transport(rank)), fake)
        except BaseException as err:  # noqa: BLE001
            errors[rank] = err

    # Daemon threads: a rank that hangs fails its test without keeping the process from exiting.
    threads = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a rank hung"
    return results, errors


def test_phases_gather_and_broadcast_over_three_ranks():
    def body(group, fake):
        assert group.agree("x", 7) == [7, 7, 7]
        parts = group.gather(json.dumps({"rank": group.rank}).encode())
        merged = group.phase(
            "merge",
            lambda: json.dumps([json.loads(p)["rank"] for p in parts]).encode() if group.is_coordinator else None,
        )
        return json.loads(group.broadcast(merged))

    results, errors = _run_ranks(3, body)
    assert errors == {} and results == {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}


@pytest.mark.parametrize(
    "failures",
    [
        {0: ValueError("disk full")},
        {1: IncompleteCheckpointError("no index")},
        {2: ValueError("disk full")},
        {0: MissingKeyError("no entry"), 2: IncompleteCheckpointError("no index")},
        {1: IncompleteCheckpointError("no index"), 2: MissingKeyError("no entry")},
    ],
    ids=["rank_0", "rank_1", "rank_2", "two_classes_from_rank_0", "two_classes_from_rank_1"],
)
def test_a_failing_rank_stops_every_rank_and_names_itself(failures):
    """Whichever ranks fail, the coordinator among them, every rank raises the class the first failing rank reported
    (its own CheckpointError subclass, or CheckpointError for an error from outside the package) and names every
    failing rank with its message."""

    def body(group, fake):
        def work():
            if group.rank in failures:
                raise failures[group.rank]

        group.phase("writing", work)

    results, errors = _run_ranks(3, body)
    first = failures[min(failures)]
    expected = type(first) if isinstance(first, CheckpointError) else CheckpointError
    assert set(errors) == {0, 1, 2}
    for rank, err in errors.items():
        assert type(err) is expected, (rank, repr(err))
        assert f"writing failed on rank(s) {sorted(failures)} of 3" in str(err), (rank, str(err))
        for failing, cause in failures.items():
            assert f"rank {failing}: {type(cause).__name__}: {cause}" in str(err), "every rank sees every cause"


def test_the_ranks_that_succeeded_add_their_notes_to_another_ranks_failure():
    """A rank whose step succeeded can say how far it got (a load's placed tensors, so a failing rank of a shared mesh
    learns that the job must restart): every rank's error carries the failing ranks' messages, then the others' notes.
    A note is built only on that path, and one that raises is replaced by its error."""
    built = []

    def note_of(rank):
        def note():
            built.append(rank)
            if rank == 2:
                raise ValueError("no count")
            return f"rank {rank} had placed 3 of 3 tensors"

        return note

    def body(group, fake):
        def work():
            if group.rank == 1:
                raise ValueError("disk full")

        group.phase("placing", lambda: None, note=note_of(group.rank))  # no rank fails: no note is built
        group.phase("placing", work, note=note_of(group.rank))

    _, errors = _run_ranks(3, body)
    assert set(errors) == {0, 1, 2} and sorted(built) == [0, 2]
    for rank, err in errors.items():
        assert str(err) == (
            "placing failed on rank(s) [1] of 3: rank 1: ValueError: disk full\n"
            "rank 0 had placed 3 of 3 tensors\n"
            "(a note that could not be built: ValueError: no count)"
        ), (rank, str(err))


def test_a_message_that_cannot_be_encoded_or_rendered_still_reaches_every_rank():
    """A failing rank's message is built after the status exchange, while the others wait in the gather that follows,
    so building it must not raise: a lone surrogate (a non-UTF-8 file name, as os.fsdecode returns it) is escaped, and
    an exception whose message cannot be rendered is named by its class."""

    class Unprintable(Exception):
        def __str__(self):
            raise ValueError("no message")

    def body(group, fake):
        def work():
            if group.rank == 1:
                raise OSError("cannot open " + os.fsdecode(b"ckpt_\xff"))  # the name in its own text, not a repr
            if group.rank == 2:
                raise Unprintable()

        group.phase("reading", work)

    results, errors = _run_ranks(3, body)
    assert set(errors) == {0, 1, 2}, "every rank raises; none waits in the gather"
    for err in errors.values():
        assert type(err) is CheckpointError and "rank(s) [1, 2] of 3" in str(err), err
        assert "rank 1: OSError: cannot open ckpt_\\udcff" in str(err), err
        assert "rank 2: Unprintable (its message could not be rendered)" in str(err), err


def _dies_sending(group, fake):
    """Make this rank's next send fail as a dying process's would, and its peers' receives with it."""

    def broken(data, dest):
        with fake.cv:
            fake.die.add(group.rank)
            fake.cv.notify_all()
        raise RuntimeError("MPI_ERR_PROC_FAILED")

    group.transport.send_bytes = broken


def test_a_failing_point_to_point_transfer_is_fatal_on_both_ends():
    def gather_body(group, fake):
        if group.rank == 1:
            _dies_sending(group, fake)
        group.gather(b"this rank's part")

    def broadcast_body(group, fake):
        if group.rank == 0:
            _dies_sending(group, fake)
        group.broadcast(b"the election" if group.is_coordinator else None)

    for body in (gather_body, broadcast_body):
        transports = {}

        def recording(group, fake, body=body):
            transports[group.rank] = group.transport
            body(group, fake)

        _, errors = _run_ranks(2, recording)
        assert set(errors) == {0, 1}, body.__name__
        assert all(type(e) is CheckpointFatalError for e in errors.values()), (body.__name__, errors)
        assert all("point-to-point transfer failed" in str(e) for e in errors.values()), (body.__name__, errors)
        # Both ends record the failure, so no later save or load in either process attempts an exchange.
        assert all("point-to-point transfer failed" in (t.broken or "") for t in transports.values()), body.__name__


def test_no_exchange_after_a_fatal_failure(expect_error):
    """A rank whose collective fails (a peer died, under ULFM) raises CheckpointFatalError, and a retry must not reach
    the other ranks: MPI pairs collectives by order, so an exchange on a communicator that failed could hang, or be
    paired with an unrelated one on a rank that moved on. The transport outlives the group of one save or load and
    records the failure, so a group built later on it raises before exchanging anything."""
    fake = _Fake(2)
    fake.die.add(1)
    transport = fake.transport(1)
    with expect_error(CheckpointFatalError, "collective failed .*; a peer may have died"):
        Group(transport).allgather(1)
    calls = []
    transport.allgather_int = lambda value: calls.append(value) or [value, value]  # would succeed if reached
    with expect_error(CheckpointFatalError, "already failed in this process .*; no further exchange is attempted"):
        Group(transport).allgather(2)
    with expect_error(CheckpointFatalError, "already failed in this process"):
        Group(transport).phase("a later save", lambda: None)
    assert calls == [], "the retry reached the transport"


def test_the_mpi_transport_is_one_per_process(monkeypatch):
    monkeypatch.setattr(_coordinator, "_TRANSPORTS", {})
    for name, value in (
        ("distributed_context_is_initialized", True),
        ("distributed_context_get_size", 2),
        ("distributed_context_get_rank", 1),
        ("distributed_context_world_rank", 1),
    ):
        monkeypatch.setattr(ttnn, name, lambda value=value: value, raising=False)
    first = _coordinator.current_transport()
    assert first is _coordinator.current_transport(), "every save and load sees the same transport"
    first.broken = "rank 1: collective failed"
    assert _coordinator.current_transport().broken == "rank 1: collective failed"


def test_scalar_restore_keeps_own_group_members():
    """On a multi-mesh world the index carries rank 0's optimizer scalars; another rank's optimizer holds other
    parameters, and the membership list must stay its own while steps and learning rates are restored."""

    class _Opt:
        def __init__(self):
            self.state = {
                "steps": 0,
                "num_param_groups": 1,
                "param_groups.0.params": "shared/w\nstage1/w",
                "param_groups.0.lr": 1e-3,
            }

        def get_state_dict(self):
            return dict(self.state)

        def set_state_dict(self, state):
            assert state["param_groups.0.params"] == "shared/w\nstage1/w", "membership must be this optimizer's own"
            self.state = dict(state)

    opt = _Opt()
    saved = {
        "optimizer/steps": 7,
        "optimizer/num_param_groups": 1,
        "optimizer/param_groups.0.params": "shared/w\nstage0/w",
        "optimizer/param_groups.0.lr": 5e-4,
    }
    assert is_structural_scalar("optimizer/param_groups.0.params") and not is_structural_scalar("optimizer/steps")
    applied = restore_scalars(opt, saved)
    assert set(applied) == {"optimizer/steps", "optimizer/num_param_groups", "optimizer/param_groups.0.lr"}
    assert opt.state["steps"] == 7 and opt.state["param_groups.0.lr"] == 5e-4
    assert opt.state["param_groups.0.params"] == "shared/w\nstage1/w"


def test_disagreeing_ranks_are_refused():
    def body(group, fake):
        group.agree("the checkpoint directory", group.rank)

    _, errors = _run_ranks(2, body)
    assert set(errors) == {0, 1}
    assert all(
        f"disagree on the checkpoint directory: rank {r} has {r} (all ranks: [0, 1])" in str(e)
        for r, e in errors.items()
    )


def test_interrupt_is_exchanged_before_it_propagates():
    """An interrupt on one rank is exchanged with the others before it propagates."""

    def body(group, fake):
        def work():
            if group.rank == 1:
                raise KeyboardInterrupt

        group.phase("writing the shards", work)  # rank 0 would wait here for good if rank 1 never reported

    _, errors = _run_ranks(2, body)
    assert type(errors[1]) is KeyboardInterrupt, "re-raised as it was on the interrupted rank"
    assert type(errors[0]) is CheckpointError and "writing the shards failed on rank(s) [1] of 2" in str(errors[0])


def test_every_exported_error_class_crosses_the_ranks():
    exported = {name for name in dcp.__all__ if name.endswith("Error")} - {"CheckpointFatalError"}
    assert {cls.__name__ for cls in _coordinator.ERROR_CLASSES} == exported
    assert _coordinator.ERROR_CLASSES[0] is CheckpointError, "status 1, the class of anything else"


def test_an_oversized_payload_fails_every_rank_inside_its_phase(monkeypatch):
    monkeypatch.setattr(_coordinator, "MAX_PAYLOAD", 64)

    def body(group, fake):
        payload = b"x" * (100 if group.rank == 1 else 10)
        payload = group.phase("serializing", lambda: checked_payload(payload, "this rank's part"))
        return group.gather(payload)

    _, errors = _run_ranks(2, body)
    assert set(errors) == {0, 1} and all("serializing failed on rank(s) [1]" in str(e) for e in errors.values())
    assert all("more than a transfer can carry" in str(e) for e in errors.values()), "every rank names the cause"


def test_ranks_must_agree_on_the_scheduler_and_differing_loaders_are_stored_per_rank(expect_error):
    """The checkpoint stores one scheduler state, which every rank restores, so the ranks' must agree. Their dataloader
    states are stored once when they agree, and per rank when they differ."""
    parts = _big_mesh_parts()
    parts[0].scheduler_hash, parts[1].scheduler_hash = 11, 12
    with expect_error(CheckpointError, r"^the scheduler state of rank\(s\) \[1\] differs from rank 0's"):
        merge_plan_parts(parts)
    parts[1].scheduler_hash = 11
    assert not merge_plan_parts(parts).dataloader_per_rank
    parts[1].dataloader_hash = 13
    election = merge_plan_parts(parts)
    assert election.dataloader_per_rank and Election.from_bytes(election.to_bytes()) == election
    back = PlanPart.from_bytes(parts[1].to_bytes())
    assert (back.scheduler_hash, back.dataloader_hash) == (11, 13)


def test_differing_dataloader_states_are_stored_per_world_rank():
    """With per-rank dataloader states, every rank's result part carries its own, keyed by its world rank, and the
    header stores them all beside rank 0's one copy of the scheduler state."""
    election, r0, r1 = _big_mesh_save()
    election.dataloader_per_rank = True
    r0.dataloader, r1.dataloader = {"0": {"position": 4}}, {"1": {"position": 5}}
    assert ResultPart.from_bytes(r1.to_bytes()).dataloader == {"1": {"position": 5}}
    _, header = merge_results([r0, r1], election, "step_0000003")
    assert header["train_state"]["dataloader_ranks"] == {"0": {"position": 4}, "1": {"position": 5}}
    assert "dataloader" not in header["train_state"] and header["train_state"]["scheduler"] == {"m_last_step": 3}


def test_param_group_members_merge_across_ranks(expect_error):
    """Param group members are merged across the ranks, and a rank's optimizer is checked to hold a subset of the
    union."""
    # Each rank's optimizer holds the shared layer and its own stage; the index records the union of the lists.
    election, r0, r1 = _big_mesh_save()
    r0.scalars = {"optimizer/num_param_groups": 1, "optimizer/param_groups.0.params": "shared/w\nstage0/w"}
    r1.scalars = {"optimizer/num_param_groups": 1, "optimizer/param_groups.0.params": "shared/w\nstage1/w"}
    index, _ = merge_results([r0, r1], election, "step_0000001")
    assert index.scalars["optimizer/param_groups.0.params"] == "shared/w\nstage0/w\nstage1/w"

    class _Opt:
        def __init__(self, members):
            self.state = {"num_param_groups": 1, "param_groups.0.params": "\n".join(members)}

        def get_state_dict(self):
            return self.state

    check_param_groups(_Opt(["shared/w", "stage1/w"]), index.scalars, strict=True)  # each rank holds part of the union
    with expect_error(CheckpointError, "holds params the checkpoint's group 0 does not"):
        check_param_groups(_Opt(["shared/w", "stage2/w"]), index.scalars, strict=True)


def test_one_rank_sub_context_keeps_its_world_rank(monkeypatch):
    """A split launch can leave a process alone in its sub context: still the group of one, but under its MPI world
    rank, as capture_rng and restore_rng key its generators."""
    for name, value in (
        ("distributed_context_is_initialized", True),
        ("distributed_context_get_size", 1),
        ("distributed_context_get_rank", 0),
        ("distributed_context_world_rank", 3),
    ):
        monkeypatch.setattr(ttnn, name, lambda value=value: value, raising=False)
    group = _coordinator.current_group()
    assert isinstance(group.transport, LoopbackTransport)
    assert (group.rank, group.size, group.world_rank) == (0, 1, 3) and this_world_rank() == 3
    monkeypatch.setattr(ttnn, "distributed_context_is_initialized", lambda: False, raising=False)
    assert _coordinator.current_group().world_rank == 0 == this_world_rank()


def test_group_of_one_runs_phases_directly(expect_error):
    """The group of one runs its phases directly. A CheckpointError keeps its class, any other error is raised as a
    CheckpointError naming the phase with the error as its cause, as on many ranks, and an interrupt passes as is."""
    group = Group(LoopbackTransport())
    assert group.phase("x", lambda: 5) == 5 and group.gather(b"p") == [b"p"] and group.broadcast(b"q") == b"q"
    assert group.agree("x", 3) == [3] and group.allgather(4) == [4] and group.is_coordinator

    def raising(err):
        return lambda: (_ for _ in ()).throw(err)

    with expect_error(CheckpointError, r"^reading failed: ValueError: raw$") as raised:
        group.phase("reading", raising(ValueError("raw")))
    assert type(raised.value) is CheckpointError and type(raised.value.__cause__) is ValueError
    with expect_error(IncompleteCheckpointError, r"^no index$") as raised:
        group.phase("reading", raising(IncompleteCheckpointError("no index")))
    assert raised.value.__cause__ is None, "a CheckpointError propagates as raised"
    with expect_error(KeyboardInterrupt, None):
        group.phase("reading", raising(KeyboardInterrupt()))


def _four_rank_load(tmp_path, monkeypatch):
    """A device-free world of four in-process ranks loading one weight, saved as two column halves, as four row
    quarters, one per rank of a mesh they share: every rank reads both chunks in part. Returns the checkpoint, its
    directory, the saved array, the loading label, each rank's placed shards and the body that loads on a rank (with
    any further arguments of ``load``)."""
    shape = (1, 1, 8, 16)
    saved, loaded = world([R, S(3)], (1, 2)), world([S(2), R], (4, 1))
    src = arange(shape)
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (src, saved, Identity())})
    path = str(tmp_path / "step_0000001")
    ckpt.write(path)

    this = threading.local()
    shards = {}
    monkeypatch.setattr(_load, "current_group", lambda: this.group)
    monkeypatch.setattr(_load, "local_coords", lambda: [(this.group.rank, 0)])
    monkeypatch.setattr(_load, "mesh_device", lambda: SimpleNamespace(shape=(4, 1)))
    monkeypatch.setattr(_load, "reshard_fsdp_modules", lambda model: [])
    monkeypatch.setattr(
        _load, "collect_model", lambda model: [LiveTensor("w", None, info("w", loaded, shape, Identity()))]
    )
    monkeypatch.setattr(
        _load, "_place", lambda live, arrays, device, local: shards.__setitem__(this.group.rank, arrays)
    )
    monkeypatch.setattr(_load, "check_label_unchanged", lambda live, when: None)

    def body(group, fake, **kwargs):
        this.group = group
        return _load.load(path, model=object(), **kwargs)

    return ckpt, path, src, loaded, shards, body


def test_a_load_over_four_ranks_has_one_reader_check_each_chunk_they_all_read_in_part(tmp_path, monkeypatch):
    """Four ranks each load a row quarter of a weight saved as two column halves, so every rank reads both chunks in
    part, which the chunks' CRC32s cannot cover: the load elects one reader of each chunk to read it whole and check it.
    The device is faked (each rank's shards land in a dict). The shards come back exact, and a flipped byte in a chunk
    stops the load on every rank, naming the rank that checked it."""
    ckpt, path, src, loaded, shards, body = _four_rank_load(tmp_path, monkeypatch)
    _, errors = _run_ranks(4, body)
    assert errors == {}, errors
    expected = expected_shards(src, loaded)
    for rank in range(4):
        assert_shards_equal(shards[rank], {group: expected[group] for group in shards[rank]})
        assert len(shards[rank]) == 1, "each rank places its own quarter"

    record = ckpt.index.entries["w"].chunks[0]
    with open(os.path.join(path, ckpt.index.files[record.file].path), "r+b") as f:
        f.seek(len(ckpt.headers[record.file]) + record.data_offset + record.nbytes - 1)
        last = f.read(1)
        f.seek(-1, os.SEEK_CUR)
        f.write(bytes([last[0] ^ 0xFF]))
    _, errors = _run_ranks(4, body)
    assert set(errors) == {0, 1, 2, 3}, errors
    for rank, err in errors.items():
        assert type(err) is CheckpointError, (rank, repr(err))
        assert re.search(
            r"^reading and placing the shards failed on rank\(s\) \[(\d)\] of 4: rank \1: CheckpointError: .*chunk 0 .*"
            r"fails its CRC32 check",
            str(err),
        ), (rank, str(err))


def test_a_failure_while_placing_the_first_tensor_on_a_shared_mesh_says_to_restart(tmp_path, monkeypatch):
    """Ranks sharing a mesh that each fail while placing their first tensor have placed nothing, but a rank cannot see
    where the others stopped, and one that failed after its device allocated part of the tensor may hold buffers the
    others do not: every rank's error says to end the job and restart it."""
    _, _, _, _, _, body = _four_rank_load(tmp_path, monkeypatch)

    def fail(live, arrays, device, local):
        raise RuntimeError("simulated device out of memory")

    monkeypatch.setattr(_load, "_place", fail)
    _, errors = _run_ranks(4, body)
    assert set(errors) == {0, 1, 2, 3}, errors
    for rank, err in errors.items():
        assert "end the job and restart it instead of loading again" in str(err), (rank, str(err))
        assert "partly restored" not in str(err), "no rank replaced a tensor"


def test_a_strict_load_refuses_ranks_that_disagree_on_passing_a_scheduler(tmp_path, monkeypatch):
    """Rank 0 alone passing a scheduler stops every rank before anything is read: the checkpoint's one scheduler state
    is every rank's to restore."""
    _, _, _, _, shards, body = _four_rank_load(tmp_path, monkeypatch)
    _, errors = _run_ranks(4, lambda group, fake: body(group, fake, scheduler=object() if group.rank == 0 else None))
    assert set(errors) == {0, 1, 2, 3}, errors
    refused = "rank(s) [1, 2, 3] of 4 pass no scheduler while rank(s) [0] pass one"
    for rank, err in errors.items():
        assert type(err) is CheckpointError and refused in str(err), (rank, str(err))
    assert shards == {}, "no rank read or placed anything"


def test_a_tensor_counts_as_placed_once_installed(tmp_path, monkeypatch):
    """A tensor whose label check fails after it was installed is counted: every rank's error says one tensor was
    placed and the model is partly restored."""
    _, _, _, _, _, body = _four_rank_load(tmp_path, monkeypatch)

    def fail(live, when):
        raise RuntimeError("simulated label change")

    monkeypatch.setattr(_load, "check_label_unchanged", fail)
    _, errors = _run_ranks(4, body)
    assert set(errors) == {0, 1, 2, 3}, errors
    counted = "after 1 of 1 tensors were placed (the first: w); the model and optimizer are partly restored"
    for rank, err in errors.items():
        assert counted in str(err), (rank, str(err))


def test_a_check_that_fails_on_one_rank_stops_every_rank_before_any_device_work(tmp_path, monkeypatch):
    """Save and load reshard the FSDP modules, which is device work (a module left gathered frees its full weights),
    only once every rank has passed the checks a rank can fail alone: a header that has no JSON form on one rank, or
    a dataloader one rank asks for that the checkpoint holds no state for, stops every rank before any rank reshards,
    so the ranks never go on with different modules gathered and different device allocations."""
    load_body = _four_rank_load(tmp_path, monkeypatch)[-1]
    resharded = []
    monkeypatch.setattr(_load, "reshard_fsdp_modules", lambda model: resharded.append("load") or [])
    _, errors = _run_ranks(
        4, lambda group, fake: load_body(group, fake, dataloader=Loader() if group.rank == 1 else None, rng=False)
    )
    assert set(errors) == {0, 1, 2, 3}, errors
    for rank, err in errors.items():
        assert re.search(r"failed on rank\(s\) \[1\] of 4: rank 1: .*train_state/dataloader", str(err)), (rank, err)
    assert resharded == [], "a rank resharded its FSDP modules in a load although another rank's check had failed"

    this = threading.local()
    monkeypatch.setattr(_save, "current_group", lambda: this.group)
    monkeypatch.setattr(_save, "reshard_fsdp_modules", lambda model: resharded.append("save") or [])
    monkeypatch.setattr(_save, "collect_model", lambda model: [])
    target = str(tmp_path / "step_0000002")

    def save_body(group, fake):
        this.group = group
        return _save.save(target, model=object(), header={"step": float("nan") if group.rank == 1 else 1}, rng=False)

    _, errors = _run_ranks(4, save_body)
    assert set(errors) == {0, 1, 2, 3}, errors
    for rank, err in errors.items():
        assert re.search(r"failed on rank\(s\) \[1\] of 4: rank 1: .*header at /step", str(err)), (rank, err)
    assert resharded == [], "a rank resharded its FSDP modules in a save although another rank's check had failed"
    assert not os.path.exists(target)
