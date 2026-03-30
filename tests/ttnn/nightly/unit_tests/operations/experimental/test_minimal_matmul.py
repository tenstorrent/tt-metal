# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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


OPTIMAL_CONFIGS = [
    # (M, K, N, M_block, K_block, N_block, subblock_h, subblock_w)
    # (512, 512, 512, 2, 8, 2, 1, 2),
    # (512, 5120, 2560, 2, 5, 8, 1, 4),
    # (1024, 1024, 1024, 8, 8, 4, 2, 2),
    (2048, 2048, 2048, 8, 8, 8, 1, 2),
    # (2368, 5120, 3840, 4, 8, 12, 1, 4),
    # (2368, 5120, 1280, 8, 4, 4, 1, 4),
    # (2368, 5120, 3456, 8, 2, 12, 2, 2),
    # (2368, 3456, 5120, 4, 4, 8, 1, 4),
    # (4096, 4096, 4096, 8, 4, 16, 1, 2),
    # (8192, 8192, 8192, 16, 4, 8, 2, 2),
    # (9472, 5120, 3840, 16, 4, 4, 1, 4),
    # (9472, 5120, 1280, 16, 8, 4, 1, 4),
    # (9472, 5120, 3456, 16, 8, 4, 1, 4),
    # (9472, 3456, 5120, 16, 3, 4, 1, 4),
    # (16384, 16384, 16384, 4, 16, 8, 2, 2),
]


@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    OPTIMAL_CONFIGS,
    ids=[f"M={c[0]}-K={c[1]}-N={c[2]}" for c in OPTIMAL_CONFIGS],
)
def test_run_perf_optimal(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    core_grid = ttnn.CoreCoord(11, 10)
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
        core_grid=core_grid,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.timeout(0)
def test_perf_report():
    fidelity_div = 2  # HiFi2
    subdir = "ttnn_minimal_matmul_optimal_perf"
    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]

    perf_results = []
    expected_results = []
    for M, K, N, *_ in OPTIMAL_CONFIGS:
        test_id = f"M={M}-K={K}-N={N}"
        command = (
            f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul.py::test_run_perf_optimal[{test_id}]"
        )
        run_device_profiler(
            command, subdir, device_analysis_types=["device_kernel_duration"], check_test_return_code=False
        )
        r = post_process_ops_log(subdir, float_columns=float_cols, op_name="", sum_vals=False, has_signposts=False)

        core_count = int(r["CORE COUNT"][0])
        duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())
        expected_ns = perf_model(M, K, N, core_count, fidelity_div)

        perf_results.append(duration_ns)
        expected_results.append(expected_ns)

    header = "| M, K, N | math util (%) | measured perf (ms) |"
    sep = "|---|---:|---:|"
    print("DTYPE: bf16, FP32 ACC: fp32, FIDELITY: HiFi2")
    print(header)
    print(sep)
    for idx, (M, K, N, M_b, K_b, N_b, sh, sw) in enumerate(OPTIMAL_CONFIGS):
        measured_ns = perf_results[idx]
        ideal_ns = expected_results[idx]
        if measured_ns is None or ideal_ns is None or measured_ns == 0:
            print(f"| ({M}, {K}, {N}) | - | - |")
        else:
            measured_ms = measured_ns / 1e6
            math_util = (ideal_ns / measured_ns) * 100.0
            print(f"| ({M}, {K}, {N}) | {math_util:.1f} | {measured_ms:.3f} |")


def test_run_perf_profiled(device):
    core_grid = ttnn.CoreCoord(11, 10)
    M, K, N = 2048, 2048, 2048
    M_block_size, K_block_size, N_block_size, subblock_h, subblock_w = 8, 8, 8, 1, 2
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
        core_grid=core_grid,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.timeout(0)
def test_profile_analysis():
    import os
    import pandas as pd
    from collections import defaultdict

    subdir = "ttnn_minimal_matmul_profile"
    command = (
        "pytest tests/ttnn/nightly/unit_tests/operations/experimental/" "test_minimal_matmul.py::test_run_perf_profiled"
    )
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"], check_test_return_code=False)

    tt_metal_home = os.environ.get("TT_METAL_HOME", os.getcwd())
    profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR", os.path.join(tt_metal_home, "generated", "profiler"))
    csv_path = os.path.join(profiler_dir, subdir, ".logs", "profile_log_device.csv")

    with open(csv_path, "r") as f:
        first_line = f.readline()
    freq_mhz = 1.0
    if "CHIP_FREQ[MHz]:" in first_line:
        freq_mhz = float(first_line.split("CHIP_FREQ[MHz]:")[1].strip().split(",")[0].strip())

    df = pd.read_csv(csv_path, skiprows=1, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    custom_zones = [
        "in0-dram-read",
        "in0-recv-wait",
        "in0-mcast",
        "in0-write-out",
        "in1-dram-read",
        "in1-recv-wait",
        "in1-mcast",
        "in1-write-out",
        "compute-wait",
        "compute-matmul",
    ]
    kernel_zones = ["BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"]
    all_zones = custom_zones + kernel_zones

    df_filtered = df[df["zone name"].str.strip().isin(all_zones)].copy()
    df_filtered["zone name"] = df_filtered["zone name"].str.strip()
    df_filtered["type"] = df_filtered["type"].str.strip()
    df_filtered["RISC processor type"] = df_filtered["RISC processor type"].str.strip()

    # compute-wait (cb_wait_front) only blocks the unpacker (TRISC_0)
    df_filtered = df_filtered[
        ~((df_filtered["zone name"] == "compute-wait") & (df_filtered["RISC processor type"] != "TRISC_0"))
    ]

    core_zone_durations = defaultdict(lambda: defaultdict(list))

    for (core_x, core_y, risc, zone), group in df_filtered.groupby(
        ["core_x", "core_y", "RISC processor type", "zone name"]
    ):
        begins = group[group["type"] == "ZONE_START"]["time[cycles since reset]"].values
        ends = group[group["type"] == "ZONE_END"]["time[cycles since reset]"].values
        n_pairs = min(len(begins), len(ends))
        for i in range(n_pairs):
            duration = ends[i] - begins[i]
            if duration > 0:
                core_zone_durations[(core_x, core_y, risc)][zone].append(duration)

    core_type_zone_totals = defaultdict(lambda: defaultdict(list))

    for (core_x, core_y, risc), zones_data in core_zone_durations.items():
        zone_names = set(zones_data.keys())
        if "in0-dram-read" in zone_names:
            core_type = "in0 injector"
        elif "in0-recv-wait" in zone_names:
            core_type = "in0 receiver"
        elif "in1-dram-read" in zone_names:
            core_type = "in1 injector"
        elif "in1-recv-wait" in zone_names:
            core_type = "in1 receiver"
        elif "compute-matmul" in zone_names:
            core_type = "compute"
        else:
            core_type = f"other ({risc})"

        for zone, durations in zones_data.items():
            total = sum(durations)
            core_type_zone_totals[core_type][zone].append(total)

    print("\n" + "=" * 100)
    print("KERNEL PROFILING BREAKDOWN: 2048x2048x2048 matmul (M_block=8, K_block=8, N_block=8)")
    print("=" * 100)

    for core_type in sorted(core_type_zone_totals.keys()):
        zones_data = core_type_zone_totals[core_type]
        n_cores = 0
        print(f"\n--- {core_type} ---")
        print(f"{'Zone':<25} {'Avg Cycles':>12} {'Min Cycles':>12} {'Max Cycles':>12} {'Avg us':>10} {'Count':>6}")
        print("-" * 80)
        for zone in all_zones:
            if zone in zones_data:
                totals = zones_data[zone]
                n_cores = max(n_cores, len(totals))
                avg_cycles = sum(totals) / len(totals)
                min_cycles = min(totals)
                max_cycles = max(totals)
                avg_us = avg_cycles / freq_mhz
                print(
                    f"{zone:<25} {avg_cycles:>12.0f} {min_cycles:>12.0f} {max_cycles:>12.0f} {avg_us:>10.1f} {len(totals):>6}"
                )
        print(f"  (across {n_cores} cores)")

    print("\n" + "=" * 100)
    print("SUMMARY: Average total cycles per core type")
    print("=" * 100)
    summary_header = f"{'Core Type':<30}"
    for zone in custom_zones:
        short = zone.replace("compute-", "cmp-").replace("write-out", "wr")
        summary_header += f" {short:>14}"
    summary_header += f" {'KERNEL':>14}"
    print(summary_header)
    print("-" * len(summary_header))

    for core_type in sorted(core_type_zone_totals.keys()):
        zones_data = core_type_zone_totals[core_type]
        row = f"{core_type:<30}"
        for zone in custom_zones:
            if zone in zones_data:
                avg = sum(zones_data[zone]) / len(zones_data[zone])
                row += f" {avg:>14.0f}"
            else:
                row += f" {'-':>14}"
        kernel_zone = None
        for kz in kernel_zones:
            if kz in zones_data:
                kernel_zone = kz
                break
        if kernel_zone:
            avg = sum(zones_data[kernel_zone]) / len(zones_data[kernel_zone])
            row += f" {avg:>14.0f}"
        else:
            row += f" {'-':>14}"
        print(row)

    print()


TABLE_CONFIGS = [
    # (512, 512, 512),
    # # (512, 1024, 1024),
    # # (512, 1024, 2048),
    # (1024, 1024, 1024),
    # # (1024, 1024, 2048),
    # # (1024, 2048, 2048),
    (2048, 2048, 2048),
    # # (2048, 2048, 3072),
    # # (2048, 3072, 3072),
    # # (3072, 3072, 3072),
    # # (3072, 3072, 4096),
    # # (3072, 4096, 4096),
    # (4096, 4096, 4096),
    # (8192, 8192, 8192),
    # (16384, 16384, 16384),
]


# @pytest.mark.skip()
@pytest.mark.timeout(0)
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

    core_grid = ttnn.CoreCoord(11, 10)
    subblocks = [(1, 2), (2, 2)] if fp32_acc else [(2, 4), (4, 2)]

    m_block_sizes = [2, 4, 8, 16]
    n_block_sizes = [2, 4, 8, 16]
    k_block_sizes = [2, 4, 8, 16]

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        m_block_sizes, k_block_sizes, n_block_sizes, subblocks
    ):
        # TILE_SIZE = 32
        # if M_block_size * core_grid.x * TILE_SIZE > M:
        #     continue
        # if N_block_size * core_grid.y * TILE_SIZE > N:
        #     continue
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
            # tt_output = ttnn.to_torch(tt_output)
            # check_result = assert_quality(torch_output, tt_output)
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


# @pytest.mark.skip()
@pytest.mark.timeout(0)
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


def test_open_device(device):
    breakpoint()
