# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: every rung of the L1 fallback ladder, forced on small shapes, plus output resharding.

The ladder (distributed_rms_norm_program_descriptor.blocking_options) only reaches its later rungs under L1 pressure
(wide HEIGHT_SHARDED rows, tall WIDTH_SHARDED shards, long prefill); each rung is pinned here by forcing it through
`first_fit`, so a regression in a rung shows up on a shape that runs in seconds.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import ttnn

import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

import importlib.util as _ilu
import pathlib as _pl

# shared acceptance helpers from the sibling test file (the op's test directory is not a package)
_spec = _ilu.spec_from_file_location("_drms_accept", _pl.Path(__file__).with_name("test_distributed_rms_norm.py"))
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
TORCH_DTYPE, _check, _gamma_to_mesh, _linear_meshes = (
    _acc.TORCH_DTYPE,
    _acc._check,
    _acc._gamma_to_mesh,
    _acc._linear_meshes,
)
_quantize, _reference, _stacked, _to_mesh = _acc._quantize, _acc._reference, _acc._stacked, _acc._to_mesh

EPS = 1e-6
ML = ttnn.TensorMemoryLayout


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


def _force(monkeypatch, **opt):
    """Make local_plan take exactly this ladder option (whatever the budget says)."""
    monkeypatch.setattr(pd, "first_fit", lambda base, sz, budget: (0, replace(base, **opt)))


def _run(
    mesh,
    shape,
    *,
    cluster_axis,
    memory_layout,
    gamma_layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    out_mc=None,
    acc=True,
):
    mesh_shape = tuple(mesh.shape)
    x = _quantize(_stacked(mesh_shape, shape, TORCH_DTYPE[dtype]), dtype)
    gamma = tg = None
    if gamma_layout is not None:
        gamma = _stacked(mesh_shape, (1, 1, 1, shape[-1]), torch.bfloat16)
        tg = _gamma_to_mesh(gamma, mesh, dtype=ttnn.bfloat16, layout=gamma_layout)
    if memory_layout == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=mesh)
    tx = _to_mesh(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    kwargs = {}
    if out_mc is not None:
        kwargs["memory_config"] = out_mc
    elif memory_layout != ML.INTERLEAVED:
        kwargs["memory_config"] = mc
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc)
    out = distributed_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=EPS,
        topology=ttnn.Topology.Linear,
        compute_kernel_config=cfg,
        **kwargs,
    )
    _check(out, _reference(x, gamma, cluster_axis, EPS), shape=shape, dtype=dtype, mesh_shape=mesh_shape)
    return out


_MESH = _linear_meshes()[:1]
pytestmark = pytest.mark.skipif(not _MESH, reason="distributed_rms_norm needs a multi-device system")

_C = pd.GAMMA_CHUNK
_W = pd.GAMMA_WAVE
_B = pd.GAMMA_BLOCK

# (id, memory layout, shape, γ layout, forced ladder option)
RUNGS = [
    ("chunk-resident", ML.INTERLEAVED, (1, 1, 640, 1792), ttnn.TILE_LAYOUT, dict(gamma_mode=_C)),
    (
        "chunk-stream-cap2",
        ML.INTERLEAVED,
        (1, 1, 640, 1792),
        ttnn.TILE_LAYOUT,
        dict(gamma_mode=_C, gamma_stream=True, cap=2),
    ),
    (
        "chunk-stream-rm-wpad",
        ML.INTERLEAVED,
        (1, 1, 64, 1000),
        ttnn.ROW_MAJOR_LAYOUT,
        dict(gamma_mode=_C, gamma_stream=True, cap=4),
    ),
    ("passes-pw1-xdepth2", ML.INTERLEAVED, (1, 1, 640, 1792), ttnn.TILE_LAYOUT, dict(PW=1, x_depth=2)),
    (
        "passes-pw2-wave-combine1",
        ML.INTERLEAVED,
        (1, 1, 2048, 1280),
        ttnn.TILE_LAYOUT,
        dict(PW=2, gamma_mode=_W, combine_depth=1),
    ),
    (
        "passes-pw1-chunkstream-wpad",
        ML.INTERLEAVED,
        (1, 1, 100, 1000),
        ttnn.ROW_MAJOR_LAYOUT,
        dict(PW=1, x_depth=2, gamma_mode=_C, gamma_stream=True, cap=8),
    ),
    ("passes-nogamma", ML.INTERLEAVED, (1, 1, 640, 1792), None, dict(PW=1, x_depth=1, cap=8)),
    (
        "ws-passes-chunkstream",
        ML.WIDTH_SHARDED,
        (1, 1, 640, 1792),
        ttnn.TILE_LAYOUT,
        dict(PW=1, gamma_mode=_C, gamma_stream=True),
    ),
    ("ws-passes-block-combine1", ML.WIDTH_SHARDED, (1, 1, 2048, 1280), ttnn.TILE_LAYOUT, dict(PW=2, combine_depth=1)),
    ("ws-wpad-passes", ML.WIDTH_SHARDED, (1, 1, 64, 1000), ttnn.ROW_MAJOR_LAYOUT, dict(PW=1, gamma_mode=_C)),
    (
        "hs-passes-chunkstream-rm",
        ML.HEIGHT_SHARDED,
        (1, 1, 640, 1792),
        ttnn.ROW_MAJOR_LAYOUT,
        dict(PW=1, gamma_mode=_C, gamma_stream=True, cap=4),
    ),
    (
        "hs-chunk-cap16",
        ML.HEIGHT_SHARDED,
        (1, 1, 32, 4096),
        ttnn.TILE_LAYOUT,
        dict(gamma_mode=_C, gamma_stream=True, cap=16),
    ),
    # Refinement 1a: x re-read (interleaved only) — Phase A streams x per square -> collapse round, the writer
    # re-reads it per (wave, chunk, tile-row) for the scaling pass
    (
        "reread-chunk",
        ML.INTERLEAVED,
        (1, 1, 640, 1792),
        ttnn.TILE_LAYOUT,
        dict(reread=True, gamma_mode=_C, sq_rows=1, x_depth=2),
    ),
    (
        "reread-sq2-depth1",
        ML.INTERLEAVED,
        (1, 1, 640, 1792),
        ttnn.TILE_LAYOUT,
        dict(reread=True, gamma_mode=_C, sq_rows=2, x_depth=1),
    ),
    (
        "reread-stream-rm-wpad-passes",
        ML.INTERLEAVED,
        (1, 1, 100, 1000),
        ttnn.ROW_MAJOR_LAYOUT,
        dict(reread=True, PW=1, gamma_mode=_C, gamma_stream=True, cap=4, sq_rows=1, x_depth=2),
    ),
    (
        "reread-nogamma-passes",
        ML.INTERLEAVED,
        (1, 1, 2048, 1280),
        None,
        dict(reread=True, PW=2, sq_rows=1, x_depth=2),
    ),
]


@pytest.mark.parametrize("mesh_shape,cluster_axis", _MESH, ids=lambda v: str(v))
@pytest.mark.parametrize("rung_id,memory_layout,shape,gamma_layout,opt", RUNGS, ids=[r[0] for r in RUNGS])
def test_ladder_rung(
    mesh_cache, monkeypatch, mesh_shape, cluster_axis, rung_id, memory_layout, shape, gamma_layout, opt
):
    mesh = mesh_cache.get(mesh_shape)
    _force(monkeypatch, **opt)
    _run(mesh, shape, cluster_axis=cluster_axis, memory_layout=memory_layout, gamma_layout=gamma_layout)


def _width_mc(mesh, shape, ncores):
    """A WIDTH_SHARDED L1 config on the first ncores of row 1.. (a placement different from auto_shard_config's)."""
    wt = shape[-1] // 32
    per = -(-wt // ncores)
    n = -(-wt // per)
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid.x, 1 + i // grid.x) for i in range(n)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    h = 1
    for d in shape[:-1]:
        h *= d
    return ttnn.MemoryConfig(
        ML.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, (h, per * 32), ttnn.ShardOrientation.ROW_MAJOR)
    )


@pytest.mark.parametrize("mesh_shape,cluster_axis", _MESH, ids=lambda v: str(v))
@pytest.mark.parametrize(
    "case",
    ["ws-to-other-ws", "interleaved-to-ws", "hs-to-interleaved", "ws-to-dram"],
)
def test_reshard_output(mesh_cache, mesh_shape, cluster_axis, case):
    mesh = mesh_cache.get(mesh_shape)
    shape = (1, 1, 32, 1024)
    if case == "ws-to-other-ws":
        out = _run(
            mesh, shape, cluster_axis=cluster_axis, memory_layout=ML.WIDTH_SHARDED, out_mc=_width_mc(mesh, shape, 4)
        )
        assert out.memory_config().memory_layout == ML.WIDTH_SHARDED
    elif case == "interleaved-to-ws":
        out = _run(
            mesh, shape, cluster_axis=cluster_axis, memory_layout=ML.INTERLEAVED, out_mc=_width_mc(mesh, shape, 8)
        )
        assert out.memory_config().memory_layout == ML.WIDTH_SHARDED
    elif case == "hs-to-interleaved":
        out = _run(
            mesh,
            (1, 1, 128, 1024),
            cluster_axis=cluster_axis,
            memory_layout=ML.HEIGHT_SHARDED,
            out_mc=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert out.memory_config().memory_layout == ML.INTERLEAVED
    else:
        out = _run(mesh, shape, cluster_axis=cluster_axis, memory_layout=ML.WIDTH_SHARDED, out_mc=ttnn.L1_MEMORY_CONFIG)
        assert out.memory_config().memory_layout == ML.INTERLEAVED


@pytest.mark.parametrize("mesh_shape,cluster_axis", _MESH, ids=lambda v: str(v))
@pytest.mark.parametrize(
    "memory_layout,shape",
    [
        (ML.WIDTH_SHARDED, (1, 1, 640, 1792)),
        (ML.INTERLEAVED, (1, 1, 2048, 1280)),
        (ML.HEIGHT_SHARDED, (1, 1, 1024, 512)),
    ],
    ids=["ws", "interleaved", "hs"],
)
def test_pass_regime_more_waves(mesh_cache, monkeypatch, mesh_shape, cluster_axis, memory_layout, shape):
    """The last ladder rung: up to WAVES_MAX_PASSES (> WAVES_MAX) smaller waves, one wave per pass (port own_stats
    semaphore ids beyond the Phase 0 count, next to the fabric connection's)."""
    mesh = mesh_cache.get(mesh_shape)
    monkeypatch.setattr(
        pd,
        "first_fit",
        lambda base, sz, budget: (0, replace(pd.rewave(base, pd.WAVES_MAX_PASSES), PW=1, combine_depth=1)),
    )
    _run(mesh, shape, cluster_axis=cluster_axis, memory_layout=memory_layout, gamma_layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("mesh_shape,cluster_axis", _MESH, ids=lambda v: str(v))
@pytest.mark.parametrize(
    "memory_layout,shape,gamma_layout",
    [
        (ML.HEIGHT_SHARDED, (1, 1, 32, 4096), ttnn.TILE_LAYOUT),
        (ML.INTERLEAVED, (1, 1, 640, 1792), ttnn.ROW_MAJOR_LAYOUT),
        (ML.WIDTH_SHARDED, (1, 1, 64, 1000), None),
    ],
    ids=["hs", "interleaved", "ws-wpad"],
)
def test_scratch_backed_cbs(mesh_cache, monkeypatch, mesh_shape, cluster_axis, memory_layout, shape, gamma_layout):
    """The last resort: no static CB region (the caller's tensors at the bottom of L1) -> every CB carved out of a
    scratch L1 buffer in the largest free hole."""
    mesh = mesh_cache.get(mesh_shape)
    monkeypatch.setattr(pd, "l1_budget", lambda device: 1024)
    _run(mesh, shape, cluster_axis=cluster_axis, memory_layout=memory_layout, gamma_layout=gamma_layout)
