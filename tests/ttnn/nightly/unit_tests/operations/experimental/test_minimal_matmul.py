# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
import os
import json
from itertools import product

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


def run_test_linear_impl(
    device,
    torch_input,
    weight_input,
    bias_input,
    tt_input,
    tt_weight,
    tt_bias,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    core_grid=None,
):
    core_grid = core_grid or device.compute_with_storage_grid_size()

    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)
    else:
        assert activation is None, f"Unsupported activation: {activation}"

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input

        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

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
    tt_output = ttnn.experimental.minimal_matmul(
        tt_input,
        tt_weight,
        bias_tensor=tt_bias,
        fused_activation=activation_fn,
        compute_kernel_config=compute_config,
        config=matmul_config,
    )
    tt_output = ttnn.to_torch(tt_output)
    check_result = assert_quality(torch_output, tt_output)
    return check_result


def run_test_linear(
    device,
    M,
    K,
    N,
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
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_dtype = torch.float32

    torch_input = torch.randn((M, K), dtype=torch_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype)
    bias_input = None
    if use_bias:
        bias_input = torch.randn((1, N), dtype=torch_dtype)

    # Prepare TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=weight_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(bias_input, dtype=bias_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)

    return run_test_linear_impl(
        device=device,
        torch_input=torch_input,
        weight_input=weight_input,
        bias_input=bias_input,
        tt_input=tt_input,
        tt_weight=tt_weight,
        tt_bias=tt_bias,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        activation=activation,
        math_fidelity=math_fidelity,
        fp32_acc=fp32_acc,
        core_grid=core_grid,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(
        device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(512, 512, 512, 1, 1, 1, 1, 1)],
)
@pytest.mark.parametrize("use_bias", [True, False], ids=["with_bias", "without_bias"])
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4],
    ids=["LoFi", "HiFi2", "HiFi4"],
)
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "fp16_acc"])
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32],
    ids=["bf16", "bf8b", "bf4b", "fp32"],
)
def test_linear_dtype_compute_config(
    device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    use_bias,
    math_fidelity,
    fp32_acc,
    dtype,
):
    check_result = run_test_linear(
        device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
        use_bias=use_bias,
        math_fidelity=math_fidelity,
        fp32_acc=fp32_acc,
        dtype=dtype,
    )

    PCC_THRESHOLD = 0.999_500
    RMSE_THRESHOLD = 0.02
    if dtype in [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32] and math_fidelity == ttnn.MathFidelity.LoFi:
        RMSE_THRESHOLD = 0.04
    if dtype == ttnn.bfloat4_b:
        PCC_THRESHOLD = 0.97
        RMSE_THRESHOLD = 0.26
    assert check_result["pcc"] > PCC_THRESHOLD
    assert check_result["relative_rmse"] < RMSE_THRESHOLD


@pytest.mark.parametrize("M", [32, 96, 320, 4096])
@pytest.mark.parametrize("K", [32, 96, 320, 4096])
@pytest.mark.parametrize("N", [32, 96, 320, 4096])
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_block_padding(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("M,K,N", [(255, 255, 255)])
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(1, 1, 1, 1, 1)],
)
def test_linear_tile_padding(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("act_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32], ids=["bf16", "bf8b", "fp32"])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32], ids=["bf16", "bf8b", "fp32"])
@pytest.mark.parametrize("bias_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32], ids=["bf16", "bf8b", "fp32"])
def test_linear_dtypes(device, act_dtype, weight_dtype, bias_dtype):
    M, K, N = 256, 256, 256
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 1, 1, 1, 1, 1
    check_result = run_test_linear(
        device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
        dtype=act_dtype,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "core_grid",
    [ttnn.CoreCoord(2, 2), ttnn.CoreCoord(4, 4), ttnn.CoreCoord(2, 4), ttnn.CoreCoord(4, 2)],
    ids=["core_grid_2x2", "core_grid_4x4", "core_grid_2x4", "core_grid_4x2"],
)
def test_linear_core_grid(device, core_grid):
    M, K, N = 256, 256, 256
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 1, 1, 1, 1, 1
    check_result = run_test_linear(
        device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, core_grid=core_grid
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
@pytest.mark.parametrize("B", [2, 3])
@pytest.mark.parametrize("T", [4, 5])
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_batch_broadcast(
    device, B, T, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w
):
    torch_input = torch.randn((B, T, M, K), dtype=torch.float32)
    weight_input = torch.randn((K, N), dtype=torch.float32)
    bias_input = torch.randn((1, N), dtype=torch.float32)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(bias_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    check_result = run_test_linear_impl(
        device=device,
        torch_input=torch_input,
        weight_input=weight_input,
        bias_input=bias_input,
        tt_input=tt_input,
        tt_weight=tt_weight,
        tt_bias=tt_bias,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    [
        (9472, 5120, 1280),
        (9472, 5120, 3456),
        (9472, 3456, 5120),
    ],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_padded_wan_shapes(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


def test_run_performance(device):
    core_grid = ttnn.CoreCoord(8, 8)
    M, K, N = 4096, 4096, 4096
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 8, 8, 8, 2, 2
    check_result = run_test_linear(
        device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, core_grid=core_grid
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


def test_performance():
    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]
    command = (
        f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py::test_run_performance"
    )

    run_device_profiler(command, "ttnn_minimal_matmul_performance", device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        "ttnn_minimal_matmul_performance",
        float_columns=float_cols,
        columns=cols,
        op_name="",
        sum_vals=False,
        has_signposts=False,
    )
    core_count = int(r["CORE COUNT"][0])
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())
    expected_ns = perf_model(4096, 4096, 4096, core_count, 2)

    util = expected_ns / duration_ns
    logger.info(f"Utilization: {util*100:.1f}%")

    if ttnn.device.is_blackhole():
        expected_util = 0.895
    else:
        expected_util = 0.582

    tolerance = 0.02
    assert (
        util > expected_util - tolerance
    ), f"Utilization {util:.1f}% is less than expected {expected_util:.1f}% by more than {tolerance:.1f}%"
    assert (
        util < expected_util + tolerance
    ), f"Utilization {util:.1f}% is greater than expected {expected_util:.1f}% by more than {tolerance:.1f}%"


TABLE_CONFIGS = [
    (512, 512, 512),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (1024, 1024, 1024),
    (1024, 1024, 2048),
    (1024, 2048, 2048),
    (2048, 2048, 2048),
    (2048, 2048, 3072),
    (2048, 3072, 3072),
    (3072, 3072, 3072),
    (3072, 3072, 4096),
    (3072, 4096, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
]


@pytest.mark.skip()
@pytest.mark.parametrize(
    "M, K, N",
    TABLE_CONFIGS,
)
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4],
    ids=["LoFi", "HiFi2", "HiFi4"],
)
@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["dtype_bf16", "dtype_bf8b", "dtype_bf4b"]
)
def test_perf_table_sweep(device, M, K, N, fp32_acc, math_fidelity, dtype):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_execution_dtype = torch.float32
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype).to(torch_execution_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype).to(torch_execution_dtype)

    # Prepare TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    core_grid = device.compute_with_storage_grid_size()
    subblocks = [(2, 2)] if fp32_acc else [(2, 4), (4, 2)]

    m_block_sizes = [2, 4, 8, 16]
    n_block_sizes = [2, 4, 8, 16]
    k_block_sizes = [2, 4, 8, 16]

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        m_block_sizes, k_block_sizes, n_block_sizes, subblocks
    ):
        if (M_block_size < subblock_h) or (N_block_size < subblock_w):
            continue
        if (M_block_size % subblock_h) != 0 or (N_block_size % subblock_w) != 0:
            continue
        logger.info(
            f"Running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
        )

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            check_result = assert_quality(torch_output, tt_output)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
            )


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


@pytest.mark.skip()
@pytest.mark.parametrize(
    "fidelity, dtype, fp32_acc",
    [
        ("HiFi2", "dtype_bf16", "fp32_acc"),
        ("HiFi2", "dtype_bf16", "bf16_acc"),
        ("HiFi4", "dtype_bf16", "bf16_acc"),
        ("HiFi2", "dtype_bf8b", "bf16_acc"),
        ("LoFi", "dtype_bf8b", "bf16_acc"),
        ("LoFi", "dtype_bf4b", "bf16_acc"),
    ],
    ids=[
        "HiFi2_bf16_fp32_acc",
        "HiFi2_bf16_bf16_acc",
        "HiFi4_bf16_bf16_acc",
        "HiFi2_bf8b_bf16_acc",
        "LoFi_bf8b_bf16_acc",
        "LoFi_bf4b_bf16_acc",
    ],
)
def test_create_perf_table(fidelity, dtype, fp32_acc):
    fidelity_div = {
        "HiFi2": 2,
        "HiFi4": 4,
        "LoFi": 1,
    }[fidelity]
    perf_results = []
    expected_results = []
    attrs_results = []
    subdir = "ttnn_linear_performance"
    for M, K, N in TABLE_CONFIGS:
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        cols = ["ATTRIBUTES"]
        command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py::test_perf_table_sweep[{dtype}-{fidelity}-{fp32_acc}-M={M}-K={K}-N={N}]"

        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        r = post_process_ops_log(
            subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
        )

        core_count = int(r["CORE COUNT"][0])
        duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())
        duration_arg_min = int(r["DEVICE KERNEL DURATION [ns]"].argmin())
        attrs = r["ATTRIBUTES"][duration_arg_min].split("'config': ")[1].split("'fused_")[0]

        expected_ns = perf_model(M, K, N, core_count, fidelity_div)

        perf_results.append(duration_ns)
        expected_results.append(expected_ns)
        attrs_results.append(attrs)

    # Pretty summary table
    config_details = f"DTYPE: {dtype}, FP32 ACC: {fp32_acc}, FIDELITY: {fidelity}"
    header = "| M, K, N | math util (%) | measured perf (ms) | attributes |"
    sep = "|---|---:|---:|---:|"
    print(config_details)
    print(header)
    print(sep)
    for idx in range(len(TABLE_CONFIGS)):
        M, K, N = TABLE_CONFIGS[idx]
        measured_ns = perf_results[idx]
        ideal_ns = expected_results[idx]
        attrs = attrs_results[idx]

        if measured_ns is None or ideal_ns is None or measured_ns == 0:
            measured_ms_str = "-"
            util_str = "-"
            attrs_str = "-"
        else:
            measured_ms = measured_ns / 1e6
            # Assume 1 cycle ≈ 1 ns for ideal estimate already returned from perf_model
            math_util = (ideal_ns / measured_ns) * 100.0
            attrs_str = attrs

            measured_ms_str = f"{measured_ms:.3f}"
            util_str = f"{math_util:.1f}"

        print(f"| ({M}, {K}, {N}) | {util_str} | {measured_ms_str} | {attrs_str} |")


# ---------------------------------------------------------------------------
# WAN2.2 matmul shape sweep (bf16, HiFi2, fp32_acc=True) using shapes from shape_trace_summary.json
# ---------------------------------------------------------------------------


def _load_wan_matmul_shapes() -> list[tuple[int, int, int]]:
    """
    Read matmul shapes from shape_trace_summary.json and return unique (M, K, N) triples.
    Expects entries like:
      { "input": [1, 1, M, K], "weight": [K, N] }
    """
    candidates: list[tuple[int, int, int]] = []
    # Try repo root; fall back to CWD
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
    path_candidates = [
        os.path.join(repo_root, "shape_trace_summary.json"),
        os.path.abspath("shape_trace_summary.json"),
    ]
    json_path = next((p for p in path_candidates if os.path.exists(p)), None)
    if json_path is None:
        logger.warning("shape_trace_summary.json not found; WAN matmul sweep will be empty")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matmuls = data.get("matmul", {})
    seen = set()
    for _, shapes in matmuls.items():
        for s in shapes:
            inp = s.get("input") or []
            w = s.get("weight") or []
            if not (isinstance(inp, list) and len(inp) >= 4 and isinstance(w, list) and len(w) >= 2):
                continue
            M = int(inp[-2])
            K_in = int(inp[-1])
            K_w = int(w[0])
            N = int(w[1])
            if K_in != K_w:
                # Skip inconsistent entries
                continue
            key = (M, K_in, N)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(key)
    return candidates


WAN_MATMUL_SHAPES = _load_wan_matmul_shapes()
WAN_MATMUL_IDS = [f"M{M}_K{K}_N{N}" for (M, K, N) in WAN_MATMUL_SHAPES]


@pytest.mark.timeout(0)
@pytest.mark.parametrize("M,K,N", WAN_MATMUL_SHAPES, ids=WAN_MATMUL_IDS)
def test_minimal_matmul_block_sweep_wan_shapes_bf16_hifi2_fp32acc(device, M, K, N):
    """
    Sweep block sizes for WAN matmul shapes under: dtype=bf16, math_fidelity=HiFi2, fp32_acc=True.
    For each viable block/subblock config, run minimal_matmul and verify numerical quality.
    """
    if not WAN_MATMUL_SHAPES:
        pytest.skip("No WAN matmul shapes loaded from shape_trace_summary.json")

    # Host tensors
    torch_input = torch.randn((M, K), dtype=torch.float32)
    weight_input = torch.randn((K, N), dtype=torch.float32)
    with torch.no_grad():
        torch_output = torch_input @ weight_input

    # TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    core_grid = device.compute_with_storage_grid_size()

    # Conservative sweep; expand if needed
    m_block_sizes = [2, 4, 8, 12, 16]
    n_block_sizes = [2, 4, 8, 12, 16]
    k_block_sizes = [2, 4, 8, 12, 16]
    subblocks = [(2, 2)]  # fp32_acc=True case

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        m_block_sizes, k_block_sizes, n_block_sizes, subblocks
    ):
        # Validity filters (match earlier sweep logic)
        if (M_block_size < subblock_h) or (N_block_size < subblock_w):
            continue
        if (M_block_size % subblock_h) != 0 or (N_block_size % subblock_w) != 0:
            continue

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            # Basic correctness gate
            res = assert_quality(torch_output, tt_output)
            assert res["pcc"] > 0.999_500
            assert res["relative_rmse"] < 0.02
        except Exception as e:
            # Don't fail the entire shape sweep on one bad config; log and continue
            logger.warning(
                f"[WAN sweep] M={M} K={K} N={N} failed for blocks "
                f"M={M_block_size} K={K_block_size} N={N_block_size} sub=({subblock_h},{subblock_w}): {e}"
            )


# @pytest.mark.skip()
@pytest.mark.timeout(0)
def test_create_wan_matmul_perf_table():
    """
    Profile all WAN matmul shapes by running the block-sweep test under tracy,
    then extract the best (min kernel duration) config and print a summary table.
    Fixed config: dtype=bf16, fidelity=HiFi2, fp32_acc=True.
    """
    if not WAN_MATMUL_SHAPES:
        pytest.skip("No WAN matmul shapes loaded from shape_trace_summary.json")

    subdir = "ttnn_minimal_matmul_wan_shapes"
    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]

    perf_results = []
    attrs_results = []
    shapes = []

    for (M, K, N), tid in zip(WAN_MATMUL_SHAPES, WAN_MATMUL_IDS):
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul.py::test_minimal_matmul_block_sweep_wan_shapes_bf16_hifi2_fp32acc[{tid}] -q -s"
        )
        run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
        r = post_process_ops_log(
            subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
        )
        core_count = int(r["CORE COUNT"][0])
        duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())
        duration_arg_min = int(r["DEVICE KERNEL DURATION [ns]"].argmin())
        attrs = r["ATTRIBUTES"][duration_arg_min]
        # Keep only the MinimalMatmulConfig attributes portion if present
        if "'config': " in attrs:
            attrs = attrs.split("'config': ")[1].split("'fused_")[0]

        shapes.append((M, K, N, core_count))
        perf_results.append(duration_ns)
        attrs_results.append(attrs)

    # Pretty summary table
    header = "| (M,K,N) | cores | measured (ms) | best config |"
    sep = "|---:|---:|---:|---|"
    print("WAN matmul sweep — bf16, HiFi2, fp32_acc=True")
    print(header)
    print(sep)
    for (M, K, N, cores), measured_ns, attrs in zip(shapes, perf_results, attrs_results):
        measured_ms = measured_ns / 1e6
        print(f"| ({M},{K},{N}) | {cores} | {measured_ms:.3f} | {attrs} |")
