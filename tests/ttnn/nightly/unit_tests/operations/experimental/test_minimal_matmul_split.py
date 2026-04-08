# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)


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
    assert (N // chunks) % 32 == 0, f"N/chunks={N // chunks} must be tile-aligned (divisible by 32)"

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


def test_linear_split_bias(device):
    M, K, N = 64, 256, 96 * 3
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
        use_bias=True,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32],
    ids=["bf16", "bf8b", "fp32"],
)
def test_linear_split_dtypes(device, dtype):
    M, K, N = 64, 256, 96 * 3
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
    """Should fail: chunks < 1"""
    with pytest.raises((RuntimeError, ValueError)):
        M, K, N = 256, 256, 512
        torch_input = torch.randn((M, K), dtype=torch.float32)
        weight_input = torch.randn((K, N), dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight_input, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.experimental.minimal_matmul_split(tt_input, tt_weight, chunks=0, dim=-1)


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


# Variable chunks tests (N-tensor support)
@pytest.mark.parametrize("chunks", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "M, K, N_per_chunk",
    [
        (256, 256, 32),  # N = 32 * chunks (1 tile per chunk)
        (32, 32, 2048),  # N = 2048 * chunks (64 tiles per chunk)
        (15, 32, 128),  # N = 128 * chunks (4 tiles per chunk)
        (15, 15, 96),  # N = 96 * chunks (3 tiles per chunk)
        (1024, 1024, 4096),  # N = 1024 * chunks (128 tiles per chunk)
    ],
)
def test_linear_split_variable_chunks(device, chunks, M, K, N_per_chunk):
    """Test variable chunk counts (2, 4, 6) with N-tensor support"""
    N = N_per_chunk * chunks  # Ensure divisible and tile-aligned
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 8, 8, 8, 2, 2
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


@pytest.mark.parametrize("chunks", [3])
def test_linear_split_variable_chunks_with_bias(device, chunks):
    """Test variable chunk counts with bias"""
    M, K, N = 256, 256, 64 * chunks
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 8, 8, 8, 2, 2
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
        use_bias=True,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


# Performance test utilities
def perf_model(M, K, N, core_count, fidelity_div):
    mm_flops = 2 * M * K * N
    core_flop_per_cycle = 2 * 8 * 16 * 16
    core_flop_per_cycle_with_fidelity = core_flop_per_cycle / fidelity_div
    chip_flop_per_cycle = core_flop_per_cycle_with_fidelity * core_count
    ideal_cycles = mm_flops / chip_flop_per_cycle
    ideal_ns = ideal_cycles
    if ttnn.device.is_blackhole():
        ideal_ns = ideal_ns / 1.3
    return ideal_ns


def post_process_ops_log(
    output_logs_subdir, float_columns=None, columns=None, sum_vals=True, op_name="", has_signposts=False
):
    filename = get_latest_ops_log_filename(output_logs_subdir)
    import pandas as pd

    df = pd.read_csv(filename)

    if has_signposts:
        # there are explicit start and stop points in the model we want to measure between
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
    if op_name != "":
        df = df[df["OP CODE"] == op_name]

    results = {}
    if float_columns:
        assert (
            type(float_columns) == list
        ), f"Bad columns name type, requested columns should be of type list but {type(float_columns)} was provided"
        for col in float_columns:
            df_filtered = df[df[col] != "-"]
            if sum_vals:
                results[col] = df_filtered[col].astype(float).sum()
            else:
                results[col] = df_filtered[col].astype(float).to_numpy()
    if columns:
        assert (
            type(columns) == list
        ), f"Bad columns name type, requested columns should be of type list but {type(columns)} was provided"
        for col in columns:
            df_filtered = df[df[col] != "-"]
            results[col] = df_filtered[col]
    else:
        results = df
    return results


@pytest.mark.parametrize(
    "chunks, shape",
    [
        (1, (4096, 4096, 4096)),
        (2, (4096, 4096, 4096)),
        (3, (4096, 4096, 6144)),
        (6, (4096, 4096, 6144)),
        (3, (9472, 5120, 3840)),
    ],
)
def test_run_performance_split(device, chunks, shape):
    core_grid = ttnn.CoreCoord(8, 8)
    M, K, N = shape
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 8, 8, 8, 2, 2
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
        core_grid=core_grid,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "chunks, shape",
    [
        (1, (4096, 4096, 4096)),
        (2, (4096, 4096, 4096)),
        (3, (4096, 4096, 6144)),
        (6, (4096, 4096, 6144)),
        (3, (9472, 5120, 3840)),
    ],
)
def test_performance_split(chunks, shape):
    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]
    command = f"pytest 'tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul_split.py::test_run_performance_split[chunks={chunks}-shape=({shape[0]}, {shape[1]}, {shape[2]})]'"

    run_device_profiler(
        command, "ttnn_minimal_matmul_split_performance", device_analysis_types=["device_kernel_duration"]
    )
    r = post_process_ops_log(
        "ttnn_minimal_matmul_split_performance",
        float_columns=float_cols,
        columns=cols,
        op_name="",
        sum_vals=False,
        has_signposts=False,
    )
    core_count = int(r["CORE COUNT"][0])
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())
    M, K, N = shape
    # M=4096, K=4096, N=4608 (full matmul size before split)
    expected_ns = perf_model(M, K, N, core_count, 2)

    util = expected_ns / duration_ns
    logger.info(f"Chunks: {chunks}, Utilization: {util*100:.1f}%")

    if ttnn.device.is_blackhole():
        expected_util = 0.895
    else:
        expected_util = 0.582

    tolerance = 0.04
    assert (
        util > expected_util - tolerance
    ), f"Utilization {util:.2f}% is less than expected {expected_util:.2f}% by more than {tolerance:.2f}%"
    assert (
        util < expected_util + tolerance
    ), f"Utilization {util:.2f}% is greater than expected {expected_util:.2f}% by more than {tolerance:.2f}%"
