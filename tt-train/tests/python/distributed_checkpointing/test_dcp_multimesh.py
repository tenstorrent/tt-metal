# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two ranks with one mesh each: the multi-mesh case of ``ttml.distributed_checkpointing``.

Each rank opens its own ``[1, 2]`` mesh (mesh ids 0 and 1) and holds a pipeline stage of its own plus a layer both
ranks replicate, as a pipeline-parallel run does. The collective save writes every stage from its rank and the
shared layer once, under one index; ``strict`` is judged over the union of the ranks, so each rank's load finds
nothing missing or unexpected although no rank holds every key. No collective spans the ranks, so the numbers are
the same as a single-rank run's. Launch on one Blackhole host:

    tt-run --mesh-graph-descriptor tt-train/tests/python/distributed_checkpointing/bh_two_1x2_meshes_one_host.textproto \\
        --hosts "$(hostname)" --bare pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multimesh.py

``--bare`` keeps the ranks on shared memory, as ``dcp_harness.tt_run_command`` explains.

Both ranks must see the same checkpoint directory: ``DCP_SHARED_TMP`` names it, the system temporary directory
otherwise. Without ``tt-run`` the module skips; ``test_dcp_cross_world`` runs it whole under tt-run, which is how CI
runs it.
"""

import os
import re

import pytest
import ttnn
from dcp_harness import ranks_in_step, two_rank_world
from dcp_shards import assert_same_shards, model_shards, optimizer_shards
from tp_block import TPBlock, make_adamw, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.distributed_checkpointing._state_dict import local_coords
from ttml.modules import AbstractModuleBase, LinearLayer

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

MESH = (1, 2)
DIM = 64
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


class _Pipeline(AbstractModuleBase):
    """A layer every rank replicates, then this rank's stage(s). The shared layer is frozen, so it stays an exact
    replica across the ranks while each rank's stage trains on weights drawn from ``stage_seed``."""

    def __init__(self, dim: int, stages: list[str], stage_seed: int) -> None:
        super().__init__()
        self.shared = LinearLayer(dim, dim)
        for tensor in self.shared.parameters().values():
            tensor.set_requires_grad(False)
        ttml.init.manual_seed(stage_seed)
        self._stages = list(stages)
        for name in stages:
            setattr(self, name, TPBlock(dim))

    def forward(self, x):
        x = self.shared(x)
        for name in self._stages:
            x = getattr(self, name)(x)
        return x


@pytest.fixture(scope="module")
def two_meshes():
    with two_rank_world(MESH, "multimesh", "two ranks with one [1, 2] mesh each") as (rank, root):
        local = local_coords()
        assert local == [(0, 0), (0, 1)], f"rank {rank} holds {local}"
        yield rank, local, root


@pytest.fixture(autouse=True)
def _ranks_in_step(request, two_meshes):
    ranks_in_step(request.node.nodeid)


def test_each_mesh_saves_its_own_stage_and_the_shared_layer_once(two_meshes, expect_error):
    rank, local, root = two_meshes
    ctx = ttml.autograd.AutoContext.get_instance()
    # The same seeds on both ranks for the shared layer, including ttml.init's own generator (seeded from OS entropy
    # otherwise), so it starts identical on both meshes and, frozen, stays so: the checkpoint keeps one copy of it,
    # which each rank gets back only because they agree. Each rank's stage is drawn from its own seed, so a stage
    # restored from the other rank's chunks would show.
    ctx.set_seed(11)
    ttml.init.manual_seed(11)
    model = _Pipeline(DIM, [f"stage{rank}"], stage_seed=100 + rank)
    params = model.parameters()
    opt = make_adamw(params)
    train_step(model, opt, DIM)
    before = model_shards(params, local)
    assert len(before) == len(params) and all(len(arrays) == len(local) for arrays in before.values())
    trainable = [name for name, tensor in params.items() if tensor.get_requires_grad()]
    assert len(trainable) == len(params) - 2, "everything but the shared layer's weight and bias"
    state_before, scalars_before = optimizer_shards(opt, local)
    assert len(state_before) == 2 * len(trainable) and all(
        len(arrays) == len(local) for arrays in state_before.values()
    )
    assert scalars_before["steps"] == 1
    path = os.path.join(root, dcp.step_dir_name(1))

    result = dcp.save(path, model=model, optimizer=opt, header={"step": 1})

    assert dcp.is_complete(path)
    index = dcp.read_index(path)
    assert [r["mesh_id"] for r in index.world["ranks"]] == [0, 1] and index.world["group_size"] == 2
    assert {f.mesh_id for f in index.files} == {0, 1}
    stages = {m.group(1) for k in index.entries for m in [re.search(r"/(stage\d+)/", k)] if m}
    assert stages == {"stage0", "stage1"}, "every rank's stage is in the one index"
    shared = [e for k, e in index.entries.items() if "/shared/" in k and e.role == "param"]
    assert shared and all(len(e.chunks) == 1 for e in shared), "a replicated layer is written once"
    assert result.keys == len(index.entries), "keys counts the whole world"

    ttml.init.manual_seed(11)
    model2 = _Pipeline(DIM, [f"stage{rank}"], stage_seed=200 + rank)
    params2 = model2.parameters()
    opt2 = make_adamw(params2)
    loaded = dcp.load(path, model=model2, optimizer=opt2)
    assert loaded.missing == [] and loaded.unexpected == [], "judged over the union of the ranks' keys"
    assert_same_shards(model_shards(params2, local), before, "parameters")
    state_after, scalars_after = optimizer_shards(opt2, local)
    assert_same_shards(state_after, state_before, "optimizer moments")
    # The scalars restored are the index's; this rank keeps its own param-group member list (its own stage).
    assert {k: v for k, v in scalars_after.items() if not k.endswith(".params")} == {
        k: v for k, v in scalars_before.items() if not k.endswith(".params")
    }

    # A rank whose model wants a key no rank saved fails strictly, and takes every rank with it.
    model3 = _Pipeline(DIM, [f"stage{rank}"] + (["extra"] if rank == 1 else []), stage_seed=300 + rank)
    with expect_error(
        dcp.MissingKeyError, r"planning the load failed on rank\(s\) \[1\] of 2: rank 1: MissingKeyError: .*extra"
    ):
        dcp.load(path, model=model3)
    ttnn.distributed_context_barrier()


def test_strict_refuses_entries_no_rank_holds_on_every_rank(two_meshes, expect_error):
    """strict judges the unexpected entries over the union of the ranks' live tensors. With rank 1 holding no stage,
    no rank holds stage1's entries: every rank refuses a strict load, and a lenient one lists them on every rank."""
    rank, local, root = two_meshes
    path = os.path.join(root, dcp.step_dir_name(1))  # the first test's checkpoint
    ttml.init.manual_seed(11)
    model = _Pipeline(DIM, ["stage0"] if rank == 0 else [], stage_seed=400 + rank)
    with expect_error(dcp.UnexpectedKeyError, r"have no live tensor: \['model/_Pipeline/stage1/"):
        dcp.load(path, model=model)
    lenient = dcp.load(path, model=model, strict=False)
    assert lenient.unexpected and all("/stage1/" in key for key in lenient.unexpected), lenient.unexpected
    ttnn.distributed_context_barrier()
