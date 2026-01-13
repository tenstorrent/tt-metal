# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc


def assert_quality(torch_output, tt_output):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    logger.info(f"PCC: {pcc_val:.7f}, Relative RMSE: {relative_rmse_val:.4f}")
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_linear_split(
    device,
    M,
    K,
    N,
    chunks,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    use_bias=False,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    dtype=ttnn.bfloat16,
    weight_dtype=None,
    bias_dtype=None,
    core_grid=None,
):
    """Test minimal_matmul_split with chunks"""
    assert N % chunks == 0, f"N={N} must be divisible by chunks={chunks}"
    assert chunks == 3, f"Only chunks=3 supported in this version, got chunks={chunks}"

    logger.info(f"Running test_linear_split with M={M}, K={K}, N={N}, chunks={chunks}")
    torch_dtype = torch.float32

    torch_input = torch.randn((M, K), dtype=torch_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype)
    bias_input = None
    if use_bias:
        bias_input = torch.randn((1, N), dtype=torch_dtype)

    # Expected output
    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input
        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

        # Split into chunks
        torch_chunks = torch.chunk(torch_output, chunks, dim=-1)

    # Prepare TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=weight_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(bias_input, dtype=bias_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)

    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)

    core_grid = core_grid or device.compute_with_storage_grid_size()

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

    tt_chunks = ttnn.experimental.minimal_matmul_split(
        tt_input,
        tt_weight,
        chunks=chunks,
        dim=-1,
        bias_tensor=tt_bias,
        fused_activation=activation_fn,
        compute_kernel_config=compute_config,
        config=matmul_config,
    )

    # Validate all chunks
    assert len(tt_chunks) == chunks, f"Expected {chunks} output tensors, got {len(tt_chunks)}"

    results = []
    for i in range(chunks):
        tt_out_i = ttnn.to_torch(tt_chunks[i])
        result_i = assert_quality(torch_chunks[i], tt_out_i)
        results.append(result_i)
        logger.info(f"Chunk {i}: PCC={result_i['pcc']:.7f}, RMSE={result_i['relative_rmse']:.4f}")

    # Return worst-case PCC/RMSE across all chunks
    return {
        "pcc": min(r["pcc"] for r in results),
        "relative_rmse": max(r["relative_rmse"] for r in results),
    }


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096 * 3)],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_split_basic(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear_split(
        device,
        M,
        K,
        N,
        chunks=3,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("use_bias", [True, False], ids=["with_bias", "without_bias"])
def test_linear_split_bias(device, use_bias):
    M, K, N = 512, 512, 512 * 3
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 1, 1, 1, 1, 1
    check_result = run_test_linear_split(
        device,
        M,
        K,
        N,
        chunks=3,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        use_bias=use_bias,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32],
    ids=["bf16", "bf8b", "fp32"],
)
def test_linear_split_dtypes(device, dtype):
    M, K, N = 256, 256, 256 * 3
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 1, 1, 1, 1, 1
    check_result = run_test_linear_split(
        device,
        M,
        K,
        N,
        chunks=3,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        dtype=dtype,
    )
    pcc_threshold = 0.999_500 if dtype == ttnn.bfloat16 else 0.99
    assert check_result["pcc"] > pcc_threshold
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("M", [32, 96, 320, 1024])
@pytest.mark.parametrize("K", [32, 96, 320, 1024])
@pytest.mark.parametrize("N", [96, 960, 3072])  # All divisible by 3 and tile-aligned after division
def test_linear_split_sizes(device, M, K, N):
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 8, 8, 8, 2, 2
    check_result = run_test_linear_split(
        device,
        M,
        K,
        N,
        chunks=3,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "core_grid",
    [ttnn.CoreCoord(2, 2), ttnn.CoreCoord(4, 4), ttnn.CoreCoord(2, 4), ttnn.CoreCoord(4, 2)],
    ids=["core_grid_2x2", "core_grid_4x4", "core_grid_2x4", "core_grid_4x2"],
)
def test_linear_split_core_grid(device, core_grid):
    M, K, N = 256, 256, 256 * 3
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 1, 1, 1, 1, 1
    check_result = run_test_linear_split(
        device,
        M,
        K,
        N,
        chunks=3,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        core_grid=core_grid,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


# Constraint validation tests
def test_invalid_chunks(device):
    """Should fail: chunks != 3"""
    with pytest.raises((RuntimeError, ValueError)):
        M, K, N = 256, 256, 512
        torch_input = torch.randn((M, K), dtype=torch.float32)
        weight_input = torch.randn((K, N), dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight_input, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.experimental.minimal_matmul_split(tt_input, tt_weight, chunks=2, dim=-1)


def test_invalid_dim(device):
    """Should fail: dim != -1"""
    with pytest.raises((RuntimeError, ValueError)):
        M, K, N = 256, 256, 768
        torch_input = torch.randn((M, K), dtype=torch.float32)
        weight_input = torch.randn((K, N), dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight_input, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.experimental.minimal_matmul_split(tt_input, tt_weight, chunks=3, dim=0)


def test_non_divisible_n(device):
    """Should fail: N not divisible by 3"""
    with pytest.raises((RuntimeError, ValueError)):
        M, K, N = 256, 256, 256  # 256 not divisible by 3
        torch_input = torch.randn((M, K), dtype=torch.float32)
        weight_input = torch.randn((K, N), dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight_input, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.experimental.minimal_matmul_split(tt_input, tt_weight, chunks=3, dim=-1)


def test_non_tile_aligned_chunk(device):
    """Should fail: N/chunks not tile-aligned"""
    with pytest.raises((RuntimeError, ValueError)):
        M, K, N = 256, 256, 99  # 99/3 = 33, not tile-aligned (not multiple of 32)
        torch_input = torch.randn((M, K), dtype=torch.float32)
        weight_input = torch.randn((K, N), dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight_input, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.experimental.minimal_matmul_split(tt_input, tt_weight, chunks=3, dim=-1)


# Skipped tests for future work
@pytest.mark.skip(reason="chunks != 3 not yet supported")
@pytest.mark.parametrize("chunks", [2, 4, 5, 8])
def test_linear_split_variable_chunks(device, chunks):
    """Future: support arbitrary chunk counts"""
    N = 256 * chunks  # Ensure divisible and tile-aligned
    M, K = 256, 256
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 1, 1, 1, 1, 1
    check_result = run_test_linear_split(
        device,
        M,
        K,
        N,
        chunks=chunks,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02
