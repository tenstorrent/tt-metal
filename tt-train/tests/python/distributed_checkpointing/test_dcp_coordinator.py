# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The multi-rank coordinator without MPI or devices: the union plan and owner election over the ranks' parts, the
merge of the ranks' results into one index, and the phase protocol over an in-process transport (a failing rank
stops every rank with its number; a failing collective is fatal and ends the exchanges)."""

import json
import threading

import pytest

from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._coordinator import (
    Group,
    PlanPart,
    RankRecord,
    ResultPart,
    UnionPlan,
    merge_plan_parts,
    merge_results,
)
from ttml.distributed_checkpointing._errors import CheckpointError, CheckpointFatal
from ttml.distributed_checkpointing._index import ChunkRecord, Entry, FileRecord, data_file_name
from ttml.distributed_checkpointing._layouts import Identity
from ttml.distributed_checkpointing._planner import TensorInfo, create_save_plan
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords


def _label(placements, shape):
    return Label(tuple(placements), tuple(shape), tuple(dist_coords(shape)))


def _info(key, placements, mesh_shape, local_shape, role="param"):
    return TensorInfo(key, role, "BF16", "TILE", _label(placements, mesh_shape), tuple(local_shape), Identity())


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


def test_union_election_gives_every_box_one_owner_that_holds_it():
    union = merge_plan_parts(_big_mesh_parts())
    assert [r.rank for r in union.records] == [0, 1] and set(union.infos) == {"model/w", "model/b"}
    w = union.owners["model/w"]
    assert len(w) == 4, "a column-sharded weight has four unique boxes"
    for group, (rank, coord) in w.items():
        assert coord == (0, group[0]) and rank == (0 if coord[1] < 2 else 1), "each box is written where it lives"
    (bias_owner,) = union.owners["model/b"].values()
    assert bias_owner in {(0, (0, 0)), (0, (0, 1)), (1, (0, 2)), (1, (0, 3))}
    # The same union, serialized and back, elects the same owners on every rank.
    assert UnionPlan.from_bytes(union.to_bytes()).owners == union.owners


def test_each_rank_plans_only_its_owned_chunks_and_files():
    parts = _big_mesh_parts()
    union = merge_plan_parts(parts)
    plans = {}
    for part in parts:
        rank = part.record.rank
        plans[rank] = create_save_plan(
            part.infos, file_for_coord=lambda c, r=rank: data_file_name(r, 0, c), owner_of=union.owner_of(rank)
        )
    for rank, plan in plans.items():
        assert all(it.chunk.mesh_coord in set(parts[rank].record.local_coords) for its in plan.items.values() for it in its)
        assert [f.coord for f in plan.files] == sorted({it.chunk.mesh_coord for its in plan.items.values() for it in its})
        assert all(f.path.startswith(f"data/w{rank:03d}_") for f in plan.files)
    assert len(plans[0].items["model/w"]) == 2 and len(plans[1].items["model/w"]) == 2
    assert len(plans[0].items["model/b"]) + len(plans[1].items["model/b"]) == 1
    # A key another rank writes entirely has no entry in this rank's part.
    bias_writer = union.owners["model/b"][()][0]
    other = 1 - bias_writer
    assert "model/b" not in plans[other].entries() and "model/b" in plans[bias_writer].entries()


def test_the_ranks_must_cover_the_label_and_agree_on_its_description():
    infos = [_info("model/w", [R, S(3)], (1, 4), (1, 1, 32, 16))]
    with pytest.raises(CheckpointError, match="no rank holds"):
        merge_plan_parts([PlanPart(_record(0, [(0, 0), (0, 1)]), infos), PlanPart(_record(1, [(0, 2)]), infos)])
    different = [_info("model/w", [R, S(2)], (1, 4), (1, 1, 32, 16))]
    with pytest.raises(CheckpointError, match="describes it differently"):
        merge_plan_parts([PlanPart(_record(0, [(0, 0), (0, 1)]), infos), PlanPart(_record(1, [(0, 2), (0, 3)]), different)])


def test_multi_mesh_ranks_with_their_own_keys_and_one_shared_key():
    """Two ranks, one [1, 2] mesh each (the same coordinates): disjoint keys, plus a replicated key both hold."""
    stage0 = _info("model/stage0/w", [R, S(3)], (1, 2), (1, 1, 32, 32))
    stage1 = _info("model/stage1/w", [R, S(3)], (1, 2), (1, 1, 32, 32))
    shared = _info("model/norm", [R, R], (1, 2), (1, 1, 1, 64))
    parts = [
        PlanPart(_record(0, [(0, 0), (0, 1)], mesh_id=0, shape=(1, 2)), [stage0, shared]),
        PlanPart(_record(1, [(0, 0), (0, 1)], mesh_id=1, shape=(1, 2)), [stage1, shared]),
    ]
    union = merge_plan_parts(parts)
    assert {r for r, _ in union.owners["model/stage0/w"].values()} == {0}
    assert {r for r, _ in union.owners["model/stage1/w"].values()} == {1}
    assert len(union.owners["model/norm"]) == 1, "one box, held by four (rank, coordinate) pairs"


def _result(rank, files, entries, rng_text):
    return ResultPart(
        _record(rank, [(0, 2 * rank), (0, 2 * rank + 1)]),
        files,
        entries,
        {"format": 1, "ranks": {str(rank): {"ttml": rng_text}}},
        sum(f.nbytes for f in files),
        scalars={"optimizer/steps": 3} if rank == 0 else {},
        header={"step": 3} if rank == 0 else {},
        train_state={"scheduler": {"m_last_step": 3}} if rank == 0 else {},
    )


def _entry(key, chunks, global_shape=(1, 1, 32, 64)):
    return Entry(
        key=key,
        role="param",
        dtype="BF16",
        layout="TILE",
        global_shape=global_shape,
        canonical_shape=global_shape,
        transform=Identity(),
        label=_label([R, S(3)], (1, 4)),
        live_keys=(key,),
        chunks=chunks,
    )


def test_merge_results_builds_one_index_with_remapped_files_and_every_ranks_generators():
    union = merge_plan_parts(_big_mesh_parts())
    chunk = lambda col, file: ChunkRecord(Box((0, 0, 0, 16 * col), (1, 1, 32, 16)), file, 4096, 1024, (0, col), 7)
    r0 = _result(
        0,
        [FileRecord("data/w000_m0_d0-0.safetensors", 5120, 0, 0, (0, 0)), FileRecord("data/w000_m0_d0-1.safetensors", 5120, 0, 0, (0, 1))],
        {"model/w": _entry("model/w", [chunk(0, 0), chunk(1, 1)])},
        "gen0",
    )
    bias = ChunkRecord(Box((0, 0, 0, 0), (1, 1, 1, 64)), 1, 8192, 128, (0, 3), 9)
    r1 = _result(
        1,
        [FileRecord("data/w001_m0_d0-2.safetensors", 5120, 1, 0, (0, 2)), FileRecord("data/w001_m0_d0-3.safetensors", 5248, 1, 0, (0, 3))],
        {"model/w": _entry("model/w", [chunk(2, 0), chunk(3, 1)]), "model/b": _entry("model/b", [bias], (1, 1, 1, 64))},
        "gen1",
    )
    index, header = merge_results([r1, r0], union, "step_0000003")
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
    # The index must hold exactly the elected chunks.
    short = _result(1, r1.files, {"model/w": _entry("model/w", [chunk(2, 0)])}, "gen1")
    with pytest.raises(CheckpointError, match="chunk"):
        merge_results([r0, short], union, "x")
    # Parts travel as bytes without loss.
    assert ResultPart.from_bytes(r1.to_bytes()).to_bytes() == r1.to_bytes()


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
                    fake.cv.wait_for(lambda: fake.boxes[(source, rank)])
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

    threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
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
        merged = group.phase("merge", lambda: json.dumps([json.loads(p)["rank"] for p in parts]).encode() if group.is_coordinator else None)
        return json.loads(group.broadcast(merged))

    results, errors = _run_ranks(3, body)
    assert errors == {} and results == {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}


def test_a_failing_rank_stops_every_rank_and_names_itself():
    def body(group, fake):
        def work():
            if group.rank == 1:
                raise ValueError("disk full")

        group.phase("writing", work)

    results, errors = _run_ranks(3, body)
    assert set(errors) == {0, 1, 2}
    assert all(isinstance(e, CheckpointError) and "rank(s) [1]" in str(e) for e in errors.values())
    assert "disk full" in str(errors[1]) and "disk full" not in str(errors[0])


def test_a_failing_collective_is_fatal_and_ends_the_exchanges():
    def body(group, fake):
        if group.rank == 2:
            fake.die.add(2)
        try:
            group.allgather(1)
        except CheckpointFatal:
            with pytest.raises(CheckpointFatal, match="no further exchange"):
                group.allgather(2)
            raise

    fake_size = 3
    fake = _Fake(fake_size)
    errors = {}

    def run(rank):
        try:
            body(Group(fake.transport(rank)), fake)
        except BaseException as err:  # noqa: BLE001
            errors[rank] = err

    t2 = threading.Thread(target=run, args=(2,))
    t2.start()
    t2.join(timeout=10)
    assert isinstance(errors[2], CheckpointFatal) and "peer may have died" in str(errors[2])


def test_scalar_restore_keeps_each_optimizers_own_param_group_members():
    """On a multi-mesh world the index carries rank 0's optimizer scalars; another rank's optimizer holds other
    parameters, and the membership list must stay its own while steps and learning rates are restored."""
    from ttml.distributed_checkpointing._state_dict import is_structural_scalar, restore_scalars

    class _Opt:
        def __init__(self):
            self.state = {"steps": 0, "num_param_groups": 1, "param_groups.0.params": "stage1/w,shared/w", "param_groups.0.lr": 1e-3}

        def get_state_dict(self):
            return dict(self.state)

        def set_state_dict(self, state):
            assert state["param_groups.0.params"] == "stage1/w,shared/w", "membership must be this optimizer's own"
            self.state = dict(state)

    opt = _Opt()
    saved = {
        "optimizer/steps": 7,
        "optimizer/num_param_groups": 1,
        "optimizer/param_groups.0.params": "stage0/w,shared/w",
        "optimizer/param_groups.0.lr": 5e-4,
    }
    assert is_structural_scalar("optimizer/param_groups.0.params") and not is_structural_scalar("optimizer/steps")
    applied = restore_scalars(opt, saved)
    assert set(applied) == {"optimizer/steps", "optimizer/num_param_groups", "optimizer/param_groups.0.lr"}
    assert opt.state["steps"] == 7 and opt.state["param_groups.0.lr"] == 5e-4
    assert opt.state["param_groups.0.params"] == "stage1/w,shared/w"


def test_disagreeing_ranks_are_refused():
    def body(group, fake):
        group.agree("the checkpoint directory", group.rank)

    _, errors = _run_ranks(2, body)
    assert all("disagree" in str(e) for e in errors.values()) and set(errors) == {0, 1}
