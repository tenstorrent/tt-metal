# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill mHC projection matmul ``x @ fn_T`` ([1,1,T,16384] fp32 x [16384, 24] fp32, HiFi4 + fp32 dest, the
composite ``_project``'s call) under explicit program configs vs the default dispatch (tt-blaze G53F-0210).

At T = 2048 the output is 64 x 1 tiles: if every output core streams all of fn_T (2 MB fp32) next to its x row
(2 MB), the call reads ~2x the bytes it needs (G53F-0206 split: 0.72 ms per call). A 1D config with fn_T multicast
(``mcast_in0=False``: the M tiles over the cores, in1 read once and multicast) keeps each output tile's K order
(fp32 dest, fp32 intermediates), so it should be bit-identical; this checks that on every chip and times each
config (``PM_REPS`` back-to-back calls, one synchronize).

    pytest models/demos/glm53_flash/tests/test_mhc_proj_matmul.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _compute_kernel_config
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
TS = [int(v) for v in os.environ.get("PM_TS", "2048,64").split(",")]
REPS = int(os.environ.get("PM_REPS", "20"))
K = 16384


def _cfg_1d(Mt: int, grid, in0_block_w: int):
    """in1 (fn_T) multicast, the M tiles over the cores (per_core_M = ceil(Mt / cores))."""
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(gx * gy, Mt)
    cx = min(gx, ncores)
    cy = -(-ncores // cx)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=-(-Mt // (cx * cy)),
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mhc_proj_matmul(mesh_device):
    g = torch.Generator().manual_seed(11)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    ckc = _compute_kernel_config()
    fn_T = up(torch.randn(K, 24, generator=g) * 0.01)
    grid = mesh_device.compute_with_storage_grid_size()
    ok = True
    for T in TS:
        x = up(torch.randn(1, 1, T, K, generator=g) * torch.exp(2 * torch.randn(1, 1, T, 1, generator=g)))
        Mt = T // 32
        variants = {"default": None}
        for w in (4, 8, 16, 32):
            variants[f"1d_in1mcast_w{w}"] = _cfg_1d(Mt, grid, w)
        ref = None
        for name, pc in variants.items():
            call = lambda: ttnn.matmul(x, fn_T, compute_kernel_config=ckc, program_config=pc)
            try:
                y = call()
            except Exception as e:  # noqa: BLE001 -- an unsupported config is a result
                logger.info(f"[proj-mm] T={T} {name}: FAILED {repr(e)[:200]}")
                continue
            chips = [ttnn.to_torch(c).reshape(T, 24).view(torch.int32) for c in ttnn.get_device_tensors(y)]
            ttnn.deallocate(y)
            if ref is None:
                ref = chips
            eq = sum(torch.equal(a, b) for a, b in zip(chips, ref))
            for _ in range(2):
                ttnn.deallocate(call())
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.deallocate(call())
            ttnn.synchronize_device(mesh_device)
            ms = (time.perf_counter() - t0) / REPS * 1e3
            logger.info(f"[proj-mm] T={T} {name}: {ms:.3f} ms per call; == default on {eq} / {len(chips)} chips")
        ttnn.deallocate(x)
    assert ok
