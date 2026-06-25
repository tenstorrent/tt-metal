# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Welford GroupNorm FP32 coverage (issue #44650).

The existing GroupNorm tests (test_group_norm.py, test_group_norm_DRAM.py) exercise welford only
in bf16. This file fills the FP32 gap on the **Welford** path (FP32 is gated to welford -- the
legacy path truncates to TF32 on SrcA and is validation-rejected). It covers FP32 input and FP32
gamma/beta, with bf16 controls, on both program factories:

  - sharded   (GroupNormShardedProgramFactory)        -- TILE and ROW_MAJOR output
  - interleaved (GroupNormMcast/NoMcastProgramFactory) -- TILE output only

FP32 requires fp32_dest_acc_en=True. Interleaved ROW_MAJOR output is unsupported for any dtype and
is intentionally not covered.
"""

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


N, C, H, W, NUM_GROUPS = 1, 320, 32, 32, 16


def _torch_ref(in_dtype):
    torch.manual_seed(0)
    x = torch.rand((N, C, H, W), dtype=torch.float32)
    w = torch.rand((C,), dtype=torch.float32)
    b = torch.rand((C,), dtype=torch.float32)
    ref = torch.nn.functional.group_norm(x, NUM_GROUPS, weight=w, bias=b).permute(0, 2, 3, 1).view(N, 1, W * H, C)
    return x, w, b, ref


def _compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # required for FP32 on the Welford path
        packer_l1_acc=False,
    )


@pytest.mark.parametrize("gb_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
def test_group_norm_sharded_welford_fp32(device, layout, in_dtype, gb_dtype):
    grid = ttnn.CoreGrid(y=1, x=8)
    x, w, b, ref = _torch_ref(in_dtype)

    xt = x.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    xt = ttnn.from_torch(xt, dtype=in_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    mask = ttnn.to_device(ttnn.create_group_norm_input_mask(C, NUM_GROUPS, grid.y, ttnn.DataType.BFLOAT8_B), device)
    gamma = ttnn.create_group_norm_weight_bias_rm(w, C, grid.y)
    beta = ttnn.create_group_norm_weight_bias_rm(b, C, grid.y)
    gt = ttnn.from_torch(
        gamma, dtype=gb_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bt = ttnn.from_torch(
        beta, dtype=gb_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    shard_shape = N * H * W // grid.x, C // grid.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec)
    xt = ttnn.to_memory_config(xt, mem)

    out = ttnn.group_norm(
        xt,
        num_groups=NUM_GROUPS,
        input_mask=mask,
        weight=gt,
        bias=bt,
        memory_config=mem,
        core_grid=grid,
        dtype=in_dtype,
        compute_kernel_config=_compute_kernel_config(device),
        use_welford=True,
        output_layout=layout,
        inplace=(layout == ttnn.ROW_MAJOR_LAYOUT),  # in-place only valid for sharded ROW_MAJOR
    )
    out = (
        ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))).float().reshape(ref.shape)
    )

    passing, pcc = comp_pcc(ref, out, pcc=0.999)
    assert passing, f"sharded welford {in_dtype} gamma={gb_dtype} {layout} PCC failed: {pcc}"


@pytest.mark.parametrize("gb_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_group_norm_interleaved_welford_fp32(device, in_dtype, gb_dtype):
    grid = ttnn.CoreGrid(y=1, x=8)
    x, w, b, ref = _torch_ref(in_dtype)

    xt = x.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    xt = ttnn.from_torch(
        xt, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    [gt, bt], mask = ttnn.dram_group_norm_params_from_torch(
        [w, b], C, NUM_GROUPS, device, core_grid=grid, return_mask=True, dtype=gb_dtype
    )

    out = ttnn.group_norm(
        xt,
        num_groups=NUM_GROUPS,
        input_mask=mask,
        weight=gt,
        bias=bt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid,
        dtype=in_dtype,
        compute_kernel_config=_compute_kernel_config(device),
        use_welford=True,
        num_out_blocks=1,
        inplace=False,
    )
    out = ttnn.to_torch(ttnn.from_device(out)).float().reshape(ref.shape)

    passing, pcc = comp_pcc(ref, out, pcc=0.999)
    assert passing, f"interleaved welford {in_dtype} gamma={gb_dtype} PCC failed: {pcc}"
