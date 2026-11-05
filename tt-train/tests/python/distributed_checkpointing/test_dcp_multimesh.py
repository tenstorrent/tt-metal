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
        --hosts "$(hostname)" pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multimesh.py

Both ranks must see the same checkpoint directory: ``DCP_SHARED_TMP`` names it, the system temporary directory
otherwise. Without ``tt-run`` the module skips.
"""

import os
import re
import shutil
import tempfile

import numpy as np
import pytest
import ttnn
from dcp_shards import local_coords, shard_arrays
from tp_block import TPBlock, make_adamw, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.modules import AbstractModuleBase, LinearLayer

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

MESH = (1, 2)
DIM = 64
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


class _Pipeline(AbstractModuleBase):
    """A layer every rank replicates, then this rank's stage(s)."""

    def __init__(self, dim: int, stages: list[str]) -> None:
        super().__init__()
        self.shared = LinearLayer(dim, dim)
        self._stages = list(stages)
        for name in stages:
            setattr(self, name, TPBlock(dim))

    def forward(self, x):
        x = self.shared(x)
        for name in self._stages:
            x = getattr(self, name)(x)
        return x


def _local_shards(params, local):
    out = {}
    for name, tensor in params.items():
        for coord, array in shard_arrays(tensor.get_value(NATIVE), local).items():
            out[(name, coord)] = array
    return out


@pytest.fixture(scope="module")
def two_meshes():
    if not ttnn.using_distributed_env():
        pytest.skip("needs two ranks with one [1, 2] mesh each; launch with tt-run")
    ttml.open_device_mesh(ttml.Mesh(MESH, ("dp", "tp")))
    try:
        size = int(ttnn.distributed_context_get_size())
        if size != 2:
            pytest.skip(f"needs exactly two ranks, launched with {size}")
        rank = int(ttnn.distributed_context_get_rank())
        device = ttml.autograd.AutoContext.get_instance().get_device()
        local = local_coords(device)
        assert local == [(0, 0), (0, 1)], f"rank {rank} holds {local}"
        ctx = ttml.autograd.AutoContext.get_instance()
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True))
        root = os.path.join(os.environ.get("DCP_SHARED_TMP", tempfile.gettempdir()), f"dcp-multimesh-{os.getuid()}")
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


def test_each_mesh_saves_its_own_stage_and_the_shared_layer_once(two_meshes):
    rank, local, root = two_meshes
    ctx = ttml.autograd.AutoContext.get_instance()
    # The same seeds on both ranks, including ttml.init's own generator (seeded from OS entropy otherwise): the
    # shared layer then starts identical on both meshes, and the stages, which differ only by name, too.
    ctx.set_seed(11)
    ttml.init.manual_seed(11)
    model = _Pipeline(DIM, [f"stage{rank}"])
    params = model.parameters()
    opt = make_adamw(params)
    train_step(model, opt, DIM)
    before = _local_shards(params, local)
    np.savez(os.path.join(root, f"before-w{rank}.npz"), **{f"{n}|{c}": a for (n, c), a in before.items()})
    ttnn.distributed_context_barrier()
    copies: dict[str, list[np.ndarray]] = {}
    for r in range(2):
        with np.load(os.path.join(root, f"before-w{r}.npz")) as part:
            for key in part.files:
                copies.setdefault(key.split("|", 1)[0], []).append(part[key])
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

    model2 = _Pipeline(DIM, [f"stage{rank}"])
    params2 = model2.parameters()
    opt2 = make_adamw(params2)
    loaded = dcp.load(path, model=model2, optimizer=opt2)
    assert loaded.missing == [] and loaded.unexpected == [], "judged over the union of the ranks' keys"
    after = _local_shards(params2, local)
    assert after.keys() == before.keys()
    for (name, coord), array in after.items():
        assert any(np.array_equal(array, c) for c in copies[name]), f"{name} at {coord} matches no saved copy"

    # A rank whose model wants a key no rank saved fails strictly, and takes every rank with it.
    model3 = _Pipeline(DIM, [f"stage{rank}"] + (["extra"] if rank == 1 else []))
    with pytest.raises(dcp.CheckpointError):  # allow-pytest.raises: the fixture matches on device error text
        dcp.load(path, model=model3)
    ttnn.distributed_context_barrier()
