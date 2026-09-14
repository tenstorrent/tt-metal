# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf guard set: one representative cell per distinct kernel path x layout x placement, each a single
op dispatch, correctness-gated. Profile the whole file in ONE run and read one DEVICE KERNEL DURATION
row per cell (in parametrization order, OP CODE == GenericOpDeviceOperation):

    scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_guard.py
    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/guard_report.py <csv>

Used by the perf tournament (changelog "Perf N") to find material regressions before a graduation.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

HIFI2_16 = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)
HIFI4_32 = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
HIFI4_16 = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False)

# id: (shape, x_dtype, x_layout, gamma_dtype|None, gamma_layout, compute cfg, shard=(shard_shape, grid)|None)
CELLS = {
    "R2_decode_7168_focus": (
        (1, 1, 32, 7168),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI2_16,
        None,
    ),
    "R3_decode_5120_sharded_8x4": (
        (1, 1, 32, 5120),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI2_16,
        ([32, 160], (8, 4)),
    ),
    "R3_decode_7168_sharded_7x4": (
        (1, 1, 32, 7168),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI2_16,
        ([32, 256], (7, 4)),
    ),
    "R3_2048_sharded_8_fp32dest": (
        (1, 1, 32, 2048),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI4_32,
        ([32, 256], (8, 1)),
    ),
    "R1_prefill_1024": (
        (1, 1, 8192, 1024),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI2_16,
        None,
    ),
    "R1_prefill_7168": (
        (1, 1, 8192, 7168),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI2_16,
        None,
    ),
    "R1_4d_no_gamma_fp32dest": ((2, 4, 128, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT, None, None, HIFI4_32, None),
    "R2_residency_64x12288_fp32dest": (
        (1, 1, 64, 12288),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        HIFI4_32,
        None,
    ),
    "R2_rm_x_rm_gamma": (
        (1, 1, 32, 7168),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        HIFI2_16,
        None,
    ),
    "R1_rm_x_rm_gamma_fp32_16bit": (
        (1, 1, 256, 1024),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        HIFI2_16,
        None,
    ),
    "R1_fp32_x_fp32_gamma": (
        (1, 1, 128, 4096),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        HIFI4_32,
        None,
    ),
    "R2_bf8b_x_bf8b_gamma": (
        (1, 1, 32, 4096),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        HIFI4_16,
        None,
    ),
}


def _torch_dtype(dt):
    return torch.float32 if dt == ttnn.float32 else torch.bfloat16


@pytest.mark.parametrize("cell", list(CELLS.keys()))
def test_rms_norm_perf_guard(device, cell):
    shape, x_dtype, x_layout, g_dtype, g_layout, cfg, shard = CELLS[cell]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_torch_dtype(x_dtype))
    xf = x.float()
    if g_dtype is not None:
        g = torch.randn(shape[-1], dtype=torch.float32).to(_torch_dtype(g_dtype))
        gf = g.float()
    else:
        g, gf = None, 1.0

    mem = ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        shard_shape, (gx, gy) = shard
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    ttnn_x = ttnn.from_torch(x, dtype=x_dtype, layout=x_layout, device=device, memory_config=mem)
    ttnn_g = None
    if g is not None:
        ttnn_g = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=g_dtype, layout=g_layout, device=device)

    # reference from the dtype-rounded device inputs
    xr = ttnn.to_torch(ttnn_x).float()
    gr = ttnn.to_torch(ttnn_g).float().reshape(-1) if ttnn_g is not None else 1.0
    expected = xr / torch.sqrt((xr * xr).mean(-1, keepdim=True) + 1e-6) * gr

    out = rms_norm(ttnn_x, gamma=ttnn_g, compute_kernel_config=ttnn.ComputeConfigDescriptor(**cfg))
    assert_with_pcc(expected, ttnn.to_torch(out).float(), 0.999)
