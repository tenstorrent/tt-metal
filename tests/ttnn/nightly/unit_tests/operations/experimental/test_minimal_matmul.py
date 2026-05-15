# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

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


# Wan2.2 AGMM-equivalent shapes: the local matmul each device performs after the
# K-axis all-gather. Matches the SHAPES list in models/tt_dit/utils/sweep_mm_block_sizes.py
# (K-fractured 4-way; cluster_size=4 is used only for K_block candidate generation
# so the block search space matches what the AGMM sweep explores).
TABLE_CONFIGS = [
    # (M, K, N, use_bias, activation, cluster_size_for_candidate_gen)
    (3072, 5120, 3840, True, None, 4),
    (3072, 5120, 1280, True, None, 4),
    (3072, 5120, 3456, True, "gelu", 4),
]


def _shape_id(M, K, N, activation):
    return f"M{M}_K{K}_N{N}_{'gelu' if activation else 'plain'}"


TABLE_CONFIG_IDS = [_shape_id(M, K, N, a) for M, K, N, _, a, _ in TABLE_CONFIGS]


@pytest.mark.timeout(0)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance sweep — skip on CI")
@pytest.mark.parametrize(
    "M, K, N, use_bias, activation, cluster_size",
    TABLE_CONFIGS,
    ids=TABLE_CONFIG_IDS,
)
def test_perf_table_sweep(device, M, K, N, use_bias, activation, cluster_size):
    """Sweep all (M_block, K_block, N_block, subblock) combos for one (shape, bias, activation).

    Block candidates match the AGMM Linear sweep (models/tt_dit/utils/sweep_mm_block_sizes.py):
      - M_block / N_block: even sizes in [2,16] union divisors of M_per_core / N_per_core.
      - K_block: divisors of K_per_device at >= K_BLOCK_MIN (K_per_device = K_tiles / cluster_size,
        matching the AGMM 4-way K-fracture so the search space is identical).
      - subblock: pick_subblock — prefers (2,2) when fp32 dest acc and both blocks even.
    L1 budget filter and subblock picker are also reused from the AGMM sweep helpers.
    """
    from models.tt_dit.utils.sweep_mm_block_sizes import (
        get_mn_block_candidates,
        get_k_block_candidates,
        pick_subblock,
        estimate_l1_kb,
        L1_BUDGET_KB,
    )

    M_tiles, K_tiles, N_tiles = M // 32, K // 32, N // 32
    core_grid = device.compute_with_storage_grid_size()
    M_per_core = (M_tiles + core_grid.x - 1) // core_grid.x
    N_per_core = (N_tiles + core_grid.y - 1) // core_grid.y
    K_per_device = K_tiles // cluster_size

    m_cands = get_mn_block_candidates(M_per_core)
    k_cands = get_k_block_candidates(K_per_device)
    n_cands = get_mn_block_candidates(N_per_core)

    use_case = "plain_gelu" if activation == "gelu" else "plain"

    print(f"\n=== {_shape_id(M, K, N, activation)} ===", flush=True)
    print(f"  per_core: M={M_per_core}  K_per_device={K_per_device}  N={N_per_core}", flush=True)
    print(f"  M blocks ({len(m_cands)}): {m_cands}", flush=True)
    print(f"  K blocks ({len(k_cands)}): {k_cands}", flush=True)
    print(f"  N blocks ({len(n_cands)}): {n_cands}", flush=True)

    combos = []
    for m_blk in m_cands:
        for k_blk in k_cands:
            for n_blk in n_cands:
                if estimate_l1_kb(m_blk, k_blk, n_blk, use_case) > L1_BUDGET_KB:
                    continue
                sb_h, sb_w = pick_subblock(m_blk, n_blk)
                combos.append((m_blk, k_blk, n_blk, sb_h, sb_w))

    print(f"  combos to measure: {len(combos)} (post-L1 filter)", flush=True)

    # Build tensors once per shape (huge speedup for the sweep vs rebuilding per combo).
    torch_input = torch.randn((M, K), dtype=torch.float32)
    weight_input = torch.randn((K, N), dtype=torch.float32)
    bias_input = torch.randn((1, N), dtype=torch.float32) if use_bias else None

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = (
        ttnn.from_torch(bias_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) if use_bias else None
    )

    activation_fn = (ttnn.UnaryOpType.GELU, False) if activation == "gelu" else None
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    failed = 0
    for m_blk, k_blk, n_blk, sb_h, sb_w in combos:
        try:
            matmul_config = ttnn.MinimalMatmulConfig(
                M_block_size=m_blk,
                K_block_size=k_blk,
                N_block_size=n_blk,
                subblock_h=sb_h,
                subblock_w=sb_w,
                compute_with_storage_grid_size=core_grid,
            )
            ttnn.experimental.minimal_matmul(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=activation_fn,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            failed += 1
    if failed:
        logger.info(f"Failed combos: {failed}")


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


@pytest.mark.timeout(0)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance sweep — skip on CI")
def test_create_perf_table():
    """Per shape, run test_perf_table_sweep via run_device_profiler and report top combos.

    Mirrors the AGMM sweep orchestrator (models/tt_dit/utils/sweep_mm_block_sizes.py
    test_mm_sweep): one profiler subprocess per shape, parse ops log, sort by kernel
    duration. Outputs the best block-size combo per shape so it can be compared to
    the AGMM Linear sweep results on the same shapes.
    """
    fidelity_div = 2  # HiFi2 (matches AGMM default in run_test_linear / sweep)
    perf_results = []
    expected_results = []
    attrs_results = []
    summary_rows = []
    for M, K, N, use_bias, activation, _cluster_size in TABLE_CONFIGS:
        shape_id = _shape_id(M, K, N, activation)
        subdir = f"ttnn_minimal_matmul_sweep_{shape_id}"
        command = (
            f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py"
            f"::test_perf_table_sweep[{shape_id}] -s"
        )

        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        r = post_process_ops_log(
            subdir,
            float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"],
            columns=["ATTRIBUTES"],
            op_name="",
            sum_vals=False,
            has_signposts=False,
        )

        durations = r["DEVICE KERNEL DURATION [ns]"]
        attrs_col = r["ATTRIBUTES"]
        if len(durations) == 0:
            print(f"\n=== {shape_id} ===  no measurements", flush=True)
            perf_results.append(None)
            expected_results.append(None)
            attrs_results.append(None)
            continue

        core_count = int(r["CORE COUNT"][0])
        # numpy/pandas ordering
        import numpy as np

        order = np.argsort(durations)
        best_idx = int(order[0])
        duration_ns = int(durations[best_idx])
        try:
            best_attrs_full = attrs_col.iloc[best_idx]
        except AttributeError:
            best_attrs_full = attrs_col[best_idx]
        # Trim attrs to just the MinimalMatmulConfig portion if possible
        try:
            best_attrs = best_attrs_full.split("'config': ")[1].split("'fused_")[0]
        except Exception:
            best_attrs = str(best_attrs_full)[:200]

        expected_ns = perf_model(M, K, N, core_count, fidelity_div)

        print(f"\n=== {shape_id} ===  ({'bias' if use_bias else 'no_bias'}, {activation or 'plain'})", flush=True)
        print(f"  Best: {duration_ns} ns", flush=True)
        print("  Top 5:", flush=True)
        for rank, idx in enumerate(order[:5], 1):
            ns = int(durations[idx])
            try:
                a_full = attrs_col.iloc[idx]
            except AttributeError:
                a_full = attrs_col[idx]
            try:
                a = a_full.split("'config': ")[1].split("'fused_")[0]
            except Exception:
                a = str(a_full)[:200]
            print(f"    #{rank}: {ns} ns  {a}", flush=True)

        perf_results.append(duration_ns)
        expected_results.append(expected_ns)
        attrs_results.append(best_attrs)
        summary_rows.append((M, K, N, use_bias, activation, duration_ns, expected_ns, best_attrs))

    # Pretty summary table
    print("\n| M, K, N | bias | act | math util (%) | measured (ms) | best config |", flush=True)
    print("|---|---|---|---:|---:|---:|", flush=True)
    for M, K, N, use_bias, activation, measured_ns, ideal_ns, attrs in summary_rows:
        if measured_ns is None or measured_ns == 0:
            print(f"| ({M}, {K}, {N}) | {use_bias} | {activation or '-'} | - | - | - |", flush=True)
            continue
        measured_ms = measured_ns / 1e6
        math_util = (ideal_ns / measured_ns) * 100.0
        print(
            f"| ({M}, {K}, {N}) | {use_bias} | {activation or '-'} | {math_util:.1f} | {measured_ms:.3f} | {attrs} |",
            flush=True,
        )
