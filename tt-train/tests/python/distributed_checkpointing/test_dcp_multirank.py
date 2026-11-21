# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two ranks co-owning one mesh: the collective save and load of ``ttml.distributed_checkpointing``.

Launch on one Blackhole host with two chips per rank (the ttnn dual-rank descriptor):

    tt-run --mesh-graph-descriptor tests/ttnn/distributed/config/bh_1x4_dual_rank_mesh_graph_descriptor.textproto \\
        --hosts "$(hostname)" --bare pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multirank.py

``--bare`` keeps the ranks on shared memory, as ``dcp_harness.tt_run_command`` explains.

Every rank writes only the shards of the two coordinates it holds; rank 0 merges the parts into one index; a load
on the same world restores every rank's shards bit for bit and each rank's own generators. Both ranks must see the
same checkpoint directory: ``DCP_SHARED_TMP`` names it (a shared filesystem when the ranks run on two hosts), the
system temporary directory otherwise. Without ``tt-run`` the module skips; ``test_dcp_cross_world`` runs it whole
under tt-run, which is how CI runs it.
"""

import gc
import os
import random
import shutil

import numpy as np
import pytest
import torch
import ttnn
from dcp_oracle import Loader
from dcp_harness import ranks_in_step, two_rank_world
from dcp_shards import assert_same_shards, model_shards, optimizer_shards
from tp_block import TPBlock, fresh_tp, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.common.schedulers import CosineAnnealingScheduler
from ttml.distributed_checkpointing import _load, _save

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

MESH = (1, 4)
DIM = 64
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


@pytest.fixture(scope="module")
def two_ranks():
    with two_rank_world(MESH, "multirank", "two ranks co-owning a [1, 4] mesh") as (rank, root):
        local = [
            tuple(c)
            for c in ttml.autograd.AutoContext.get_instance().get_device().get_view().get_local_mesh_coord_range()
        ]
        assert local == [(0, 2 * rank), (0, 2 * rank + 1)], f"rank {rank} holds {local}"
        yield rank, local, root


@pytest.fixture(autouse=True)
def _ranks_in_step(request, two_ranks):
    ranks_in_step(request.node.nodeid)


@pytest.fixture(autouse=True)
def ranks_in_lockstep():
    """Ranks that share a mesh must make the same device allocations and frees in the same order. A test can leave
    device tensors to the cyclic garbage collector, whose runs differ between the ranks (an exception's traceback
    holds the frames that raised it), so every test ends with a collection on each rank before the next one starts."""
    yield
    gc.collect()
    ttnn.distributed_context_barrier()


def test_ranks_write_their_coordinates(two_ranks):
    """Each rank writes the files of its coordinates, and rank 0 merges the index."""
    rank, local, root = two_ranks
    ctx = ttml.autograd.AutoContext.get_instance()
    ttml.init.manual_seed(0)  # the ranks build the same replicated weights, which then train alike
    model, params, opt = fresh_tp(DIM)
    sched = CosineAnnealingScheduler(opt, T_max=8)
    loader = Loader()
    ctx.set_seed(100 + rank)
    random.seed(200 + rank)
    np.random.seed(300 + rank)
    torch.manual_seed(400 + rank)
    for _ in range(2):
        train_step(model, opt, DIM)
        sched.step()
        loader.advance()
    before = model_shards(params, local)
    assert len(before) == len(params) and all(len(arrays) == len(local) for arrays in before.values())
    state_before, scalars_before = optimizer_shards(opt, local)
    assert len(state_before) == 2 * len(params) and all(len(arrays) == len(local) for arrays in state_before.values())
    assert scalars_before["steps"] == 2
    path = os.path.join(root, dcp.step_dir_name(2))

    result = dcp.save(path, model=model, optimizer=opt, scheduler=sched, dataloader=loader, header={"step": 2})

    assert dcp.is_complete(path)
    assert all(f"/w{rank:03d}_" in f for f in result.files), "this rank's files carry its world rank"
    assert result.keys == 3 * len(params), "every parameter and its two AdamW moments, counted over the whole world"
    index = dcp.read_index(path)
    assert index.world["group_size"] == 2 and [r["rank"] for r in index.world["ranks"]] == [0, 1]
    assert [tuple(map(tuple, r["local_coords"])) for r in index.world["ranks"]] == [((0, 0), (0, 1)), ((0, 2), (0, 3))]
    writers = {f.world_rank for f in index.files}
    assert writers == {0, 1}, "both ranks wrote files"
    for entry in index.entries.values():
        assert [tuple(c.box.offsets) for c in entry.chunks] == sorted({tuple(c.box.offsets) for c in entry.chunks})
    stored = dcp.read_train_state(path)
    assert set(stored["rng"]["ranks"]) == {"0", "1"}, "every rank's generators are in the header"
    assert stored["dataloader"] == {"position": 2}
    own_gen_after_save = ctx.get_generator_state()

    # The uninterrupted run continues one step on every rank.
    sched.step()
    loader.advance()
    expected = {"lrs": sched.get_last_lrs(), "python": random.random(), "numpy": float(np.random.random())}

    model2, params2, opt2 = fresh_tp(DIM)
    sched2 = CosineAnnealingScheduler(opt2, T_max=8)
    loader2 = Loader()
    ctx.set_seed(7)
    random.seed(0)
    np.random.seed(0)
    loaded = dcp.load(path, model=model2, optimizer=opt2, scheduler=sched2, dataloader=loader2, rng=True)

    assert loaded.missing == [] and loaded.unexpected == [] and loaded.header == {"step": 2}
    assert loaded.restored[:2] == ["scheduler", "dataloader"] and "rng.ttml" in loaded.restored
    assert_same_shards(model_shards(params2, local), before, "parameters")  # exactly the coordinates this rank holds
    state_after, scalars_after = optimizer_shards(opt2, local)
    assert_same_shards(state_after, state_before, "optimizer moments")
    assert scalars_after["steps"] == 2 and scalars_after == scalars_before
    assert ctx.get_generator_state() == own_gen_after_save, "this rank's own ttml generator came back"
    sched2.step()
    loader2.advance()
    assert sched2.get_last_lrs() == expected["lrs"] and loader2.position == 3
    assert random.random() == expected["python"] and float(np.random.random()) == expected["numpy"]

    # strict is judged over the union of the ranks: a model-only load of a model+optimizer checkpoint is fine, and
    # the optimizer entries are not reported as unexpected either way.
    model3, params3, opt3 = fresh_tp(DIM)
    plain = dcp.load(path, model=model3)
    assert plain.unexpected == [] and plain.missing == []
    ttnn.distributed_context_barrier()


def test_rank_failing_mid_save_stops_every_rank(two_ranks, expect_error):
    """A rank failing mid-save stops every rank and leaves no checkpoint."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "refused")
    # On rank 1 a model that disagrees with rank 0's description of the same keys: rank 0's merge refuses it, and
    # every rank raises with rank 0's message. Both ranks build it: ranks sharing a mesh must make the same device
    # allocations in the same order, or the buffers of every later tensor sit at different addresses on their devices.
    wider = TPBlock(DIM * 2)
    wider.parameters()  # materializes the weights
    other = wider if rank == 1 else model
    with expect_error(
        dcp.CheckpointError,
        r"merging the plan parts failed on rank\(s\) \[0\] of 2: rank 0: CheckpointError: .* rank 1 describes it",
    ):
        dcp.save(path, model=other)
    ttnn.distributed_context_barrier()
    assert not os.path.exists(path) and not any(n.startswith("refused.partial-") for n in os.listdir(root))
    ttnn.distributed_context_barrier()


def test_a_rank_called_with_a_bad_argument_stops_every_rank(two_ranks, expect_error):
    """A rank whose find_latest saw no checkpoint passes None: every rank raises, naming that rank, in the first
    phase of save and of load, and nothing is created."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "bad_argument") if rank == 0 else None
    with expect_error(dcp.CheckpointError, r"collecting the live tensors failed on rank\(s\) \[1\] of 2"):
        dcp.save(path, model=model)
    with expect_error(dcp.CheckpointError, r"checking the arguments failed on rank\(s\) \[1\] of 2"):
        dcp.load(path, model=model)
    ttnn.distributed_context_barrier()
    assert not os.path.exists(os.path.join(root, "bad_argument"))
    ttnn.distributed_context_barrier()


def test_ranks_whose_optimizers_differ_are_refused(two_ranks, expect_error):
    """The index holds one copy of the optimizer's scalars, rank 0's. Ranks whose learning rates differ, or of which
    one passes no optimizer, are refused on every rank, and nothing is written."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "optimizers_differ")
    if rank == 1:
        opt.set_group_lrs([5e-4])
    with expect_error(dcp.CheckpointError, r"the ranks disagree on the optimizer's scalars"):
        dcp.save(path, model=model, optimizer=opt)
    with expect_error(dcp.CheckpointError, r"the ranks disagree on whether an optimizer is saved"):
        dcp.save(path, model=model, optimizer=opt if rank == 0 else None)
    ttnn.distributed_context_barrier()
    assert not os.path.exists(path) and not any(n.startswith("optimizers_differ.partial-") for n in os.listdir(root))
    ttnn.distributed_context_barrier()


def test_truncated_file_on_one_rank_stops_all(two_ranks, expect_error):
    """Every rank opens and checks its data files while the load is planned, so a truncated file of rank 1's stops
    every rank before any rank places anything, whether or not rank 0 reads that file too (the election spreads
    replicated boxes over both ranks' files)."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    path = os.path.join(root, "truncated")
    dcp.save(path, model=model)
    ttnn.distributed_context_barrier()
    if rank == 0:  # one of rank 1's files, through the shared directory
        victim = next(f.path for f in dcp.read_index(path).files if f.world_rank == 1)
        os.truncate(os.path.join(path, victim), 100)
    ttnn.distributed_context_barrier()
    fresh, fresh_params, _ = fresh_tp(DIM)
    before = model_shards(fresh_params, local)
    with expect_error(
        dcp.IncompleteCheckpointError, r"planning the load failed on rank\(s\) \[(0, )?1\] of 2: .* is 100 bytes but"
    ):
        dcp.load(path, model=fresh)
    assert_same_shards(model_shards(fresh_params, local), before, "no rank placed anything")
    ttnn.distributed_context_barrier()


def test_a_directory_the_ranks_do_not_share_is_refused_before_any_write(two_ranks, monkeypatch, expect_error):
    """A checkpoint directory on each host's own disk: rank 1 does not see the staging directory rank 0 created, and
    must not make one of its own. Every rank raises before anything is written, and rank 1's disk is left as it was.
    Rank 1 stands in for the other host by resolving the staging directory under a directory of its own."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "not_shared")
    own_disk = os.path.join(root, "rank1_disk")
    if rank == 1:
        os.makedirs(own_disk)
        real_staging_dir = _save.staging_dir
        monkeypatch.setattr(
            _save, "staging_dir", lambda d, nonce: os.path.join(own_disk, os.path.basename(real_staging_dir(d, nonce)))
        )
    with expect_error(
        dcp.CheckpointError,
        r"creating the staging directory failed on rank\(s\) \[1\] of 2: rank 1: CheckpointError: .*, which rank 0 "
        r"created, does not exist on rank 1: the checkpoint directory must be on a filesystem every rank shares",
    ):
        dcp.save(path, model=model, optimizer=opt)
    monkeypatch.undo()
    ttnn.distributed_context_barrier()
    assert not os.path.exists(path) and not any(n.startswith("not_shared.partial-") for n in os.listdir(root))
    assert os.listdir(own_disk) == [], "rank 1 made no staging directory of its own"
    ttnn.distributed_context_barrier()
    if rank == 0:
        shutil.rmtree(own_disk)
    ttnn.distributed_context_barrier()


def test_ranks_reading_different_checkpoints_at_one_path_are_refused(two_ranks, monkeypatch, expect_error):
    """Hosts holding their own copies at one path agree on the directory but would load different checkpoints, so the
    ranks also agree on the index's id and creation time, and nothing is placed. Rank 1 stands in for a host holding an
    older copy by reading an index with another creation time."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "own_copies")
    dcp.save(path, model=model)
    if rank == 1:
        real_read_index = _load.read_index

        def older_copy(directory):
            index = real_read_index(directory)
            index.created = "2000-01-01T00:00:00Z"
            return index

        monkeypatch.setattr(_load, "read_index", older_copy)
    fresh, fresh_params, _ = fresh_tp(DIM)
    before = model_shards(fresh_params, local)
    with expect_error(dcp.CheckpointError, r"the ranks disagree on the checkpoint \(its id and creation time\): rank "):
        dcp.load(path, model=fresh)
    assert_same_shards(model_shards(fresh_params, local), before, "no rank placed anything")
    ttnn.distributed_context_barrier()


def test_a_rank_that_placed_its_tensors_says_so_when_another_rank_fails(two_ranks, monkeypatch, expect_error):
    """Rank 1 places every tensor, as rank 0 does (ranks sharing a mesh must make the same device allocations in the
    same order), but reports a failure at its last one. Rank 0's error then says that it placed all of its tensors and
    that its model is partly restored; rank 1's message counts its own tensors, once."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    path = os.path.join(root, "placed_on_rank0")
    dcp.save(path, model=model)
    fresh, fresh_params, _ = fresh_tp(DIM)
    n = len(fresh_params)
    if rank == 1:
        real_place = _load._place
        placed = []

        def place_then_fail_at_the_last(live, *args):
            real_place(live, *args)
            placed.append(live.key)
            if len(placed) == n:
                raise RuntimeError("simulated device out of memory")

        monkeypatch.setattr(_load, "_place", place_then_fail_at_the_last)
    pattern = (
        r"^reading and placing the shards failed on rank\(s\) \[1\] of 2: rank 1: CheckpointError: load failed at "
        rf".*\(RuntimeError: simulated device out of memory\) after {n - 1} of {n} tensors were placed"
    )
    try:
        with expect_error(dcp.CheckpointError, pattern) as raised:
            dcp.load(path, model=fresh)
        message = str(raised.value)
        if rank == 0:
            assert f"; rank 0 had placed {n} of {n} tensors, so the model and optimizer are partly restored" in message
        else:
            assert "rank 1 had placed" not in message, "a failing rank's own message already counts its tensors"
    finally:
        ttnn.distributed_context_barrier()


def test_differing_dataloaders_are_stored_per_rank(two_ranks, expect_error):
    """Data-parallel loaders keep a position (a seed, a shard) of their own: the ranks' differing states are stored
    per world rank, and each rank restores its own. A rank that saved without a dataloader has no state to restore,
    which a strict load with one refuses on every rank."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    loader = Loader()
    for _ in range(3 + rank):
        loader.advance()
    path = os.path.join(root, "per_rank_loaders")
    dcp.save(path, model=model, dataloader=loader)
    stored = dcp.read_train_state(path)
    assert stored["dataloader_ranks"] == {"0": {"position": 3}, "1": {"position": 4}} and "dataloader" not in stored
    resumed = Loader()
    assert "dataloader" in dcp.load(path, model=model, dataloader=resumed, rng=False).restored
    assert resumed.position == 3 + rank, "each rank restores its own"

    path = os.path.join(root, "one_loader")
    dcp.save(path, model=model, dataloader=loader if rank == 0 else None)
    assert dcp.read_train_state(path)["dataloader_ranks"] == {"0": {"position": 3}}
    with expect_error(
        dcp.MissingKeyError,
        r"reading the index and collecting the live tensors failed on rank\(s\) \[1\] of 2: .*train_state/dataloader",
    ):
        dcp.load(path, model=model, dataloader=Loader(), rng=False)
    ttnn.distributed_context_barrier()


def test_a_failure_on_one_rank_stops_every_rank_and_names_it(two_ranks, monkeypatch, expect_error):
    """The one-rank failures save and load must turn into the same error on every rank: a header that is not JSON
    on rank 1, a data file rank 1 cannot create (the staging directory is then removed), and ranks that load with
    different strictness."""
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "one_rank")
    with expect_error(
        dcp.CheckpointError, r"collecting the live tensors failed on rank\(s\) \[1\] of 2: rank 1: .*header at /step"
    ):
        dcp.save(path, model=model, header={"step": np.int64(1) if rank == 1 else 1})

    real_create = _save.create_file

    def create_file(file_path, header, data_bytes):
        if rank == 1:
            raise OSError("simulated EDQUOT")
        return real_create(file_path, header, data_bytes)

    monkeypatch.setattr(_save, "create_file", create_file)
    with expect_error(
        dcp.CheckpointError, r"writing the shards failed on rank\(s\) \[1\] of 2: rank 1: OSError: simulated EDQUOT"
    ):
        dcp.save(path, model=model)
    monkeypatch.undo()
    ttnn.distributed_context_barrier()
    assert not os.path.exists(path) and not any(n.startswith("one_rank.partial-") for n in os.listdir(root))
    ttnn.distributed_context_barrier()

    dcp.save(path, model=model)
    with expect_error(dcp.CheckpointError, r"the ranks disagree on strict and whether an optimizer is loaded: rank"):
        dcp.load(path, model=model, strict=rank == 0)
    ttnn.distributed_context_barrier()


def test_a_checkpoint_for_the_one_process_load(two_ranks):
    """The save half of a cross-world case: two ranks co-owning the [1, 4] mesh write a TPBlock checkpoint, so most
    keys' chunks come from both ranks' files, and dump their own shards next to it. ``test_dcp_cross_world`` runs this
    module with ``DCP_CROSS_WORLD_DIR`` set, then loads the checkpoint on one process."""
    out = os.environ.get("DCP_CROSS_WORLD_DIR")
    if not out:
        pytest.skip("run by test_dcp_cross_world with DCP_CROSS_WORLD_DIR set")
    rank, local, root = two_ranks
    ttml.init.manual_seed(0)
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    dcp.save(os.path.join(out, "ckpt"), model=model)
    coords = sorted(local)
    shards = model_shards(params, coords)
    assert len(shards) == len(params) and all(len(arrays) == len(coords) for arrays in shards.values())
    np.savez(
        os.path.join(out, f"shards_{rank}.npz"),
        **{
            f"{name}@{'-'.join(map(str, coord))}": array
            for name, arrays in shards.items()
            for coord, array in zip(coords, arrays)
        },
    )
    index = dcp.read_index(os.path.join(out, "ckpt"))
    assert {f.world_rank for f in index.files} == {0, 1}
    ttnn.distributed_context_barrier()
