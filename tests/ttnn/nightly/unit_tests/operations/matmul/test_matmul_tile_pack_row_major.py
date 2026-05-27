# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


# Phase-2 targeted coverage for tile_pack_row_major on mcast factories.
# Configs intentionally pick subblocks that the legacy FATAL rejected
# (out_subblock_w != per_core_N AND out_subblock_h != 1) to confirm the gate
# is in place and the absolute-offset pack + row-group writer produce correct
# output.
#
# Multi-row subblocks (out_subblock_h > 1) are exercised together with
# in1_num_subblocks > 2 via the dedicated test_bug3_* case below: this was
# previously broken (PCC ~0.9) because the 2D / 1D mcast and non-mcast
# factories shared the out_cb and interm0_cb L1 region, and the row-major
# per-row-group reserve on out_cb would overwrite interm0's unconsumed
# partials. The factories now force separate regions when tile_pack_row_major
# is set.


def _dtype_pcc(dtype, k_tiles):
    # bfloat8_b accumulates more quantization error; widen PCC on large K.
    if dtype == ttnn.bfloat8_b and k_tiles > 16:
        return 0.97
    return 0.99


def _check_matmul(torch_out, out, in_dtype, k_tiles):
    # Mirror test_matmul.py's bf16/bf8 atol/rtol/Frobenius scaling so allclose +
    # Frobenius pass alongside PCC. ULP off — bfp/bf16 matmul output isn't ULP-stable.
    pcc = _dtype_pcc(in_dtype, k_tiles)
    k = k_tiles * 32
    if in_dtype == ttnn.bfloat8_b:
        assert_numeric_metrics(
            torch_out,
            out,
            atol=0.004 * k,
            rtol=0.624 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=pcc,
            check_ulp=False,
        )
    else:
        assert_numeric_metrics(
            torch_out,
            out,
            atol=0.004 * k,
            rtol=0.227 * k,
            frobenius_threshold=0.001 * k,
            pcc_threshold=pcc,
            check_ulp=False,
        )


def _input_a_block_sharded(m, k, grid_xy):
    return ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=grid_xy[1], x=grid_xy[0]),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


# -------- 2D mcast: no-bias, multi-row subblocks --------------------------


@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w",
    [
        (2, 2),  # multi-row; legacy FATAL rejected
        (4, 2),  # largest multi-row (8 DST tiles)
        (1, 4),  # h=1 fast path
    ],
    ids=["subblk_2x2", "subblk_4x2", "subblk_1x4"],
)
@pytest.mark.parametrize("out_sharded", [True], ids=["l1_sharded_out"])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["in1_bf16", "in1_bfp8"])
def test_mcast_2d_tile_pack_row_major_no_bias(device, out_subblock_h, out_subblock_w, out_sharded, in1_dtype):
    # M=256 (8 tiles) x K=256 (8 tiles) x N=256 (8 tiles), 2x2 grid.
    # Per-core: M=4, N=4 tiles. in0_block_w=4 → 2 K-block iterations.
    m_tiles, k_tiles, n_tiles = 8, 8, 8
    m, k, n = m_tiles * 32, k_tiles * 32, n_tiles * 32
    grid_xy = (2, 2)
    per_core_M = m_tiles // grid_xy[1]
    per_core_N = n_tiles // grid_xy[0]

    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, m, k).to(torch.float32)
    torch_b = torch.randn(1, 1, k, n).to(torch.float32)
    torch_out = torch_a @ torch_b

    a = ttnn.from_torch(
        torch_a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_a_block_sharded(m, k, grid_xy),
    )
    b = ttnn.from_torch(
        torch_b.to(torch.bfloat16),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_mem = (
        ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1)
        if out_sharded
        else ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
        tile_pack_row_major=True,
    )

    out_t = ttnn.matmul(
        a,
        b,
        program_config=program_config,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t)
    _check_matmul(torch_out, out, in1_dtype, k_tiles)


# -------- 2D mcast: FUSE_BIAS across all supported subblocks --------------


@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w",
    [
        (2, 2),  # multi-row; add_bias_bcast_rows row-major path
        (4, 2),  # largest multi-row
        (1, 4),  # h=1 fast path
        (1, 2),  # h=1, w < per_core_N
    ],
    ids=["subblk_2x2", "subblk_4x2", "subblk_1x4", "subblk_1x2"],
)
@pytest.mark.parametrize("out_sharded", [True], ids=["l1_sharded_out"])
def test_mcast_2d_tile_pack_row_major_fuse_bias(device, out_subblock_h, out_subblock_w, out_sharded):
    m_tiles, k_tiles, n_tiles = 8, 8, 8
    m, k, n = m_tiles * 32, k_tiles * 32, n_tiles * 32
    grid_xy = (2, 2)
    per_core_M = m_tiles // grid_xy[1]
    per_core_N = n_tiles // grid_xy[0]

    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, m, k).to(torch.float32)
    torch_b = torch.randn(1, 1, k, n).to(torch.float32)
    torch_bias = torch.randn(1, 1, 1, n).to(torch.float32)
    torch_out = torch_a @ torch_b + torch_bias

    a = ttnn.from_torch(
        torch_a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_a_block_sharded(m, k, grid_xy),
    )
    b = ttnn.from_torch(
        torch_b.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Logical shape [1, 1, 1, N] — ttnn pads M=1 up to TILE_HEIGHT internally.
    # Validation in matmul_device_operation.cpp (post main PR #42430) checks
    # dim_bias == 1 || dim_bias == dim_out on the LOGICAL shape, so an
    # expand to [1, 1, 32, N] would now fail (32 != 1, 32 != output M).
    bias = ttnn.from_torch(
        torch_bias.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_mem = (
        ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1)
        if out_sharded
        else ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
        tile_pack_row_major=True,
    )

    out_t = ttnn.linear(
        a,
        b,
        bias=bias,
        program_config=program_config,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t)
    _check_matmul(torch_out, out, ttnn.bfloat16, k_tiles)


# -------- 1D mcast_in0 ----------------------------------------------------


@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w",
    [
        (2, 2),  # in1_num_subblocks = 2, multi-row supported
        (4, 2),  # in1_num_subblocks = 2, multi-row supported
        (1, 4),  # h=1 fast path, full subblock
        (1, 2),  # h=1
    ],
    ids=["subblk_2x2", "subblk_4x2", "subblk_1x4", "subblk_1x2"],
)
@pytest.mark.parametrize("out_sharded", [True], ids=["l1_sharded_out"])
def test_mcast_1d_mcast_in0_tile_pack_row_major(device, out_subblock_h, out_subblock_w, out_sharded):
    # 1D mcast_in0: A is width-sharded, B comes via mcast.
    # out_sharded=True   → WIDTH_SHARDED L1 output (OUT_SHARDED short-wait path).
    # out_sharded=False  → DRAM-interleaved output (writer kernel's Phase-1
    #                      #ifdef TILE_PACK_ROW_MAJOR path, without OUT_SHARDED).
    # Shape: M=128 (4 tiles), K=256 (8 tiles), N=256 (8 tiles), grid (2, 1).
    # Per-core M=4, per_core_N=4 tiles. With out_subblock_w=2 → in1_num_subblocks=2.
    m_tiles, k_tiles, n_tiles = 4, 8, 8
    m, k, n = m_tiles * 32, k_tiles * 32, n_tiles * 32
    grid_xy = (2, 1)
    per_core_M = m_tiles
    per_core_N = n_tiles // grid_xy[0]
    in0_block_w = k_tiles // grid_xy[0]

    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, m, k).to(torch.float32)
    torch_b = torch.randn(1, 1, k, n).to(torch.float32)
    torch_out = torch_a @ torch_b

    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=grid_xy[1], x=grid_xy[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    a = ttnn.from_torch(
        torch_a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem,
    )
    b = ttnn.from_torch(
        torch_b.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_mem = (
        ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1)
        if out_sharded
        else ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        tile_pack_row_major=True,
    )

    out_t = ttnn.matmul(
        a,
        b,
        program_config=program_config,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t)
    _check_matmul(torch_out, out, ttnn.bfloat16, k_tiles)


# -------- Multi-row subblock + in1_num_subblocks > 2 regression guard -----
# Previously produced PCC ~0.9 because out_cb and interm0_cb shared the same
# L1 region in the mcast factories. Row-major's per-row-group reserve on
# out_cb landed on top of interm0's unconsumed subblocks. Fixed by gating
# the shared-region CB path on !tile_pack_row_major in 2D / 1D / non-mcast
# factories; these cases must stay green as a regression guard.


@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w, in1_num_subblocks_expected",
    [
        (2, 2, 4),  # multi-row + 4 N-subblocks
        (1, 2, 4),  # h=1 control
    ],
    ids=["multi_row_h2w2_sb4", "control_h1w2_sb4"],
)
def test_multi_row_wide_n_shared_cb_guard(device, out_subblock_h, out_subblock_w, in1_num_subblocks_expected):
    # Doubled N vs the regular 2D mcast tests to push in1_num_subblocks > 2.
    # Per-core: M=4, N=8 → in1_num_subblocks = 8 / out_subblock_w.
    m_tiles, k_tiles, n_tiles = 8, 8, 16
    m, k, n = m_tiles * 32, k_tiles * 32, n_tiles * 32
    grid_xy = (2, 2)
    per_core_M = m_tiles // grid_xy[1]
    per_core_N = n_tiles // grid_xy[0]
    assert per_core_N // out_subblock_w == in1_num_subblocks_expected

    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, m, k).to(torch.float32)
    torch_b = torch.randn(1, 1, k, n).to(torch.float32)
    torch_out = torch_a @ torch_b

    a = ttnn.from_torch(
        torch_a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_a_block_sharded(m, k, grid_xy),
    )
    b = ttnn.from_torch(
        torch_b.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
        tile_pack_row_major=True,
    )

    out_t = ttnn.matmul(
        a,
        b,
        program_config=program_config,
        memory_config=ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
        ),
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t)
    _check_matmul(torch_out, out, ttnn.bfloat16, k_tiles)


# -------- 1D mcast_in0: untilize_out (helper's last-K-block to interm + Phase-3) --
# Phase-A regression guard for the matmul_block helper. With untilize_out=True the
# kernel routes the last K-block through LastBlockTarget::Interm and runs a
# downstream reblock_and_untilize phase. Adding pack_untilize.h to the helper's .inl
# (Phase A introduced LastBlockTarget::OutWithUntilize, which lives in the same
# .inl) means a header-include or namespacing regression here would surface as a
# JIT-compile failure on this kernel; PCC drift on the un-untilized reference
# would surface a behavioral regression in the spill / reload / Phase-3 chain.
# Single subblock (out_subblock_h=1) keeps it compatible with the 1D factory's
# untilize_out constraints.


@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w",
    [
        (1, 4),
        (1, 2),
    ],
    ids=["subblk_1x4", "subblk_1x2"],
)
def test_mcast_1d_mcast_in0_tile_pack_row_major_untilize_out(device, out_subblock_h, out_subblock_w):
    m_tiles, k_tiles, n_tiles = 4, 8, 8
    m, k, n = m_tiles * 32, k_tiles * 32, n_tiles * 32
    grid_xy = (2, 1)
    per_core_M = m_tiles
    per_core_N = n_tiles // grid_xy[0]
    in0_block_w = k_tiles // grid_xy[0]

    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, m, k).to(torch.float32)
    torch_b = torch.randn(1, 1, k, n).to(torch.float32)
    torch_out = torch_a @ torch_b

    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=grid_xy[1], x=grid_xy[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    a = ttnn.from_torch(
        torch_a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem,
    )
    b = ttnn.from_torch(
        torch_b.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        untilize_out=True,
        tile_pack_row_major=True,
    )

    out_mem = ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1)
    out_t = ttnn.matmul(
        a,
        b,
        program_config=program_config,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t)
    _check_matmul(torch_out, out, ttnn.bfloat16, k_tiles)


# -------- 2D mcast: fused ReLU activation, no bias ------------------------


@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w",
    [
        (2, 2),
        (4, 2),
    ],
    ids=["subblk_2x2", "subblk_4x2"],
)
def test_mcast_2d_tile_pack_row_major_with_activation(device, out_subblock_h, out_subblock_w):
    m_tiles, k_tiles, n_tiles = 8, 8, 8
    m, k, n = m_tiles * 32, k_tiles * 32, n_tiles * 32
    grid_xy = (2, 2)
    per_core_M = m_tiles // grid_xy[1]
    per_core_N = n_tiles // grid_xy[0]

    torch.manual_seed(0)
    torch_a = torch.randn(1, 1, m, k).to(torch.float32)
    torch_b = torch.randn(1, 1, k, n).to(torch.float32)
    torch_out = torch.nn.functional.relu(torch_a @ torch_b)

    a = ttnn.from_torch(
        torch_a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_a_block_sharded(m, k, grid_xy),
    )
    b = ttnn.from_torch(
        torch_b.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_xy,
        in0_block_w=4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        fuse_batch=True,
        tile_pack_row_major=True,
    )

    out_t = ttnn.matmul(
        a,
        b,
        program_config=program_config,
        memory_config=ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
        ),
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t)
    _check_matmul(torch_out, out, ttnn.bfloat16, k_tiles)
