# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two ranks co-owning one mesh: the collective save and load of ``ttml.distributed_checkpointing``.

Launch on one Blackhole host with two chips per rank (the ttnn dual-rank descriptor):

    tt-run --mesh-graph-descriptor tests/ttnn/distributed/config/bh_1x4_dual_rank_mesh_graph_descriptor.textproto \\
        --hosts "$(hostname)" pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multirank.py

Every rank writes only the shards of the two coordinates it holds; rank 0 merges the parts into one index; a load
on the same world restores every rank's shards bit for bit and each rank's own generators. Both ranks must see the
same checkpoint directory: ``DCP_SHARED_TMP`` names it (a shared filesystem when the ranks run on two hosts), the
system temporary directory otherwise. Without ``tt-run`` the module skips.
"""

import os
import random
import shutil
import tempfile

import numpy as np
import pytest
import torch
import ttnn
from dcp_shards import shard_arrays
from tp_block import TPBlock, fresh_tp, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.common.schedulers import CosineAnnealingScheduler

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

MESH = (1, 4)
DIM = 64
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def _local_shards(params, local):
    """Every parameter's shards at this rank's coordinates, keyed by name and coordinate."""
    out = {}
    for name, tensor in params.items():
        for coord, array in shard_arrays(tensor.get_value(NATIVE), local).items():
            out[(name, coord)] = array
    return out


@pytest.fixture(scope="module")
def two_ranks():
    if not ttnn.using_distributed_env():
        pytest.skip("needs two ranks co-owning a [1, 4] mesh; launch with tt-run")
    ttml.open_device_mesh(ttml.Mesh(MESH, ("dp", "tp")))
    try:
        size = int(ttnn.distributed_context_get_size())
        if size != 2:
            pytest.skip(f"needs exactly two ranks, launched with {size}")
        rank = int(ttnn.distributed_context_get_rank())
        local = [
            tuple(c)
            for c in ttml.autograd.AutoContext.get_instance().get_device().get_view().get_local_mesh_coord_range()
        ]
        assert local == [(0, 2 * rank), (0, 2 * rank + 1)], f"rank {rank} holds {local}"
        ctx = ttml.autograd.AutoContext.get_instance()
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True))
        root = os.path.join(os.environ.get("DCP_SHARED_TMP", tempfile.gettempdir()), f"dcp-multirank-{os.getuid()}")
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
            os.makedirs(root, exist_ok=True)
        ttnn.distributed_context_barrier()
        yield rank, local, root
        ttnn.distributed_context_barrier()
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
    finally:
        ttml.close_device_mesh()


class _Loader:
    def __init__(self) -> None:
        self.position = 0

    def advance(self) -> None:
        self.position += 1

    def get_state_dict(self) -> dict:
        return {"position": self.position}

    def set_state_dict(self, state: dict) -> None:
        self.position = int(state["position"])


def test_each_rank_writes_its_coordinates_and_rank_zero_merges_the_index(two_ranks):
    rank, local, root = two_ranks
    ctx = ttml.autograd.AutoContext.get_instance()
    model, params, opt = fresh_tp(DIM)
    sched = CosineAnnealingScheduler(opt, T_max=8)
    loader = _Loader()
    ctx.set_seed(100 + rank)
    random.seed(200 + rank)
    np.random.seed(300 + rank)
    torch.manual_seed(400 + rank)
    for _ in range(2):
        train_step(model, opt, DIM)
        sched.step()
        loader.advance()
    before = _local_shards(params, local)
    # Devices that hold the same replicated tensor need not agree across ranks (a TP world whose collectives do
    # not span its ranks lets them drift), and the checkpoint keeps the copy of the device elected to write it: a
    # loaded shard must equal one of the saving ranks' copies, so every rank dumps its copies for the others.
    np.savez(os.path.join(root, f"before-w{rank}.npz"), **{f"{n}|{c}": a for (n, c), a in before.items()})
    ttnn.distributed_context_barrier()
    copies: dict[str, list[np.ndarray]] = {}
    for r in range(2):
        with np.load(os.path.join(root, f"before-w{r}.npz")) as part:
            for key in part.files:
                copies.setdefault(key.split("|", 1)[0], []).append(part[key])
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
    loader2 = _Loader()
    ctx.set_seed(7)
    random.seed(0)
    np.random.seed(0)
    loaded = dcp.load(path, model=model2, optimizer=opt2, scheduler=sched2, dataloader=loader2, rng=True)

    assert loaded.missing == [] and loaded.unexpected == [] and loaded.header == {"step": 2}
    assert loaded.restored[:2] == ["dataloader", "scheduler"] and "rng.ttml" in loaded.restored
    after = _local_shards(params2, local)
    assert after.keys() == before.keys(), "this rank restores exactly the coordinates it holds"
    for (name, coord), array in after.items():
        assert any(np.array_equal(array, c) for c in copies[name]), f"{name} at {coord} matches no saved copy"
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


def test_a_rank_failing_mid_save_stops_every_rank_and_leaves_no_checkpoint(two_ranks):
    rank, local, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    path = os.path.join(root, "refused")
    if rank == 1:
        # A model that disagrees with rank 0's description of the same key: the merge refuses it on every rank.
        other = TPBlock(DIM * 2)
        other.parameters()  # materializes the weights
        with pytest.raises(dcp.CheckpointError):  # allow-pytest.raises: the fixture matches on device error text
            dcp.save(path, model=other)
    else:
        with pytest.raises(dcp.CheckpointError):  # allow-pytest.raises: the fixture matches on device error text
            dcp.save(path, model=model)
    ttnn.distributed_context_barrier()
    assert not os.path.exists(path) and not any(n.startswith("refused.partial-") for n in os.listdir(root))
    ttnn.distributed_context_barrier()
