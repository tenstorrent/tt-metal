# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scratch: per-core determinism sweep, using the gate's exact window() recipe.

TT_SWEEP_MODE=fullgrid -> ONE 11x10 window (positive control; must flag device 21)
TT_SWEEP_MODE=percore  -> every core as its own 1x1 window (di/dt negative control)

  TT_SWEEP_MODE=fullgrid pytest -q tests/didt/scratch_core_sweep.py -k galaxy -p no:randomly --timeout=0 -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

EMB_DIM = 7168
PER_CORE_M, PER_CORE_N = 10, 12
IN0_BLOCK_W = 8
MAX_GRID = (11, 10)
MESH_DEVICE_PARAMS = [pytest.param((8, 4), id="galaxy")]


def _shards(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
@pytest.mark.timeout(0)
def test_core_sweep(mesh_device):
    iters = int(os.environ.get("TT_SWEEP_ITERS", "10"))
    mode = os.environ.get("TT_SWEEP_MODE", "percore")
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    gx_max, gy_max = min(grid.x, MAX_GRID[0]), min(grid.y, MAX_GRID[1])
    ids = list(mesh_device.get_device_ids())

    compute_config = (ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def window(ox, oy, gx, gy):
        seq, hidden = gy * PER_CORE_M * 32, gx * PER_CORE_N * 32
        x, w = (
            ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for t in (torch.randn(1, 1, seq, EMB_DIM) * 0.02, torch.randn(EMB_DIM, hidden) * 0.02)
        )
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=IN0_BLOCK_W,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=PER_CORE_M,
            per_core_N=PER_CORE_N,
            transpose_mcast=False,
            fused_activation=None,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + gx - 1, oy + gy - 1))}
            ),
        )
        r2r, baseline = {}, None
        for i in range(iters):
            out = ttnn.matmul(x, w, program_config=pc, compute_kernel_config=compute_config)
            ttnn.synchronize_device(mesh_device)
            cur = _shards(out)
            ttnn.deallocate(out)
            if i == 0:
                baseline = cur
            else:
                for c in range(len(cur)):
                    if not torch.equal(baseline[c], cur[c]):
                        r2r[ids[c]] = r2r.get(ids[c], 0) + int((baseline[c] != cur[c]).sum())
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        return r2r

    if mode == "grid":  # TT_SWEEP_GRID=GXxGY -> one window of that size at (0,0)
        gx, gy = (int(v) for v in os.environ["TT_SWEEP_GRID"].split("x"))
        plan = [(0, 0, gx, gy)]
    elif mode == "ladder":  # TT_SWEEP_LADDER=7,8,9,10,11 -> gx x gy_max each
        plan = [(0, 0, int(gx), gy_max) for gx in os.environ["TT_SWEEP_LADDER"].split(",")]
    elif mode == "fullgrid":
        plan = [(0, 0, gx_max, gy_max)]
    else:
        plan = [(cx, cy, 1, 1) for cx in range(gx_max) for cy in range(gy_max)]

    logger.info(f"mode={mode}, {len(plan)} window(s), {iters} iters each, K={EMB_DIM}")
    failures, skipped = {}, []
    for ox, oy, gx, gy in plan:
        try:
            r2r = window(ox, oy, gx, gy)
        except Exception as e:
            skipped.append(((ox, oy), type(e).__name__))
            continue
        for dev, n in r2r.items():
            failures[(dev, (ox, oy))] = failures.get((dev, (ox, oy)), 0) + n
            logger.warning(f"  window {gx}x{gy} at ({ox},{oy}): device {dev} NOT deterministic, {n} elements")

    logger.info(f"skipped windows: {len(skipped)} -> {skipped[:12]}")
    devs = sorted({d for d, _ in failures})
    logger.success(f"mode={mode}: {len(failures)} (device,window) ND hits; devices={devs or 'NONE'}")
    assert not failures, f"non-deterministic: devices={devs}, hits={sorted(failures.items())[:15]}"
