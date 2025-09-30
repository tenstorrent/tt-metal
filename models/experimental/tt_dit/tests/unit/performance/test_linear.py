"""
This file sweeps all DiT matmul shapes.
All DiT matmuls have the following configuration:
- interleaved activation, weight, output
- BF16 for all inputs and output
- HiFi2 fidelity and FP32 accumulation.

We run `ttnn.linear` with a core_grid parameter, no custom program config, and expect all of these shapes to
1. pass the test
2. have math utilization of > 60%
"""

import pytest
import torch
import ttnn

from ....utils.tensor import bf16_tensor
from ....utils.check import assert_quality

# from models.perf.device_perf_utils import run_device_perf

from tracy.process_model_log import (
    post_process_ops_log,
    run_device_profiler,
)


def run_test_linear(device, M, K, N):
    torch_dtype = torch.float32

    torch_model = torch.nn.Linear(K, N, bias=True).to(dtype=torch_dtype)
    torch_model.eval()

    torch_input = torch.randn((M, K), dtype=torch_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    # ttnn.linear expects weight shaped (K, N)
    torch_weight_t = torch_model.weight.transpose(0, 1)
    tt_weight = bf16_tensor(torch_weight_t, device=device)
    tt_bias = None
    if torch_model.bias is not None:
        tt_bias = bf16_tensor(torch_model.bias.reshape(1, -1), device=device)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,  # NOTE: True can improve correctness, tiny performance cost
    )
    core_grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=core_grid.y, x=core_grid.x)

    tt_output = ttnn.linear(
        tt_input, tt_weight, bias=tt_bias, compute_kernel_config=compute_config, core_grid=core_grid
    )

    # Compare outputs
    tt_output = ttnn.to_torch(tt_output)
    check_result = assert_quality(torch_output, tt_output)

    return check_result


PARAMS = [
    (11264, 3072, 4608),
    (11264, 3072, 8192),
    (11264, 4096, 3072),
    (5632, 3072, 2304),
    (5632, 3072, 4096),
    (5632, 2048, 3072),
    (37888, 5120, 1280),
    (37888, 5120, 3456),
    (37888, 3456, 5120),
    (9472, 5120, 1280),
    (9472, 5120, 3456),
    (9472, 3456, 5120),
    (9472, 5120, 2560),
    (9472, 5120, 6912),
    (9472, 6912, 5120),
]
PARAM_IDS = [
    "mochi_qkv_2x4sp1tp0",
    "mochi_ff1_2x4sp1tp0",
    "mochi_ff2_2x4sp1tp0",
    "mochi_qkv_4x8sp1tp0",
    "mochi_ff1_4x8sp1tp0",
    "mochi_ff2_4x8sp1tp0",
    "wan_qkv_2x4sp0tp1",
    "wan_ff1_2x4sp0tp1",
    "wan_ff2_2x4sp0tp1",
    "wan_qkv_4x8sp1tp0",
    "wan_ff1_4x8sp1tp0",
    "wan_ff2_4x8sp1tp0",
    "wan_qkv_2x8sp1tp0",
    "wan_ff1_2x8sp1tp0",
    "wan_ff2_2x8sp1tp0",
]


@pytest.mark.parametrize(
    "M, K, N",
    PARAMS,
    ids=PARAM_IDS,
)
def test_linear(device, M, K, N):
    check_result = run_test_linear(device, M, K, N)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


def perf_model(M, K, N, core_count):
    mm_flops = 2 * M * K * N
    core_flop_per_cycle = 2 * 8 * 16 * 16
    core_flop_per_cycle_hifi2 = core_flop_per_cycle / 2
    chip_flop_per_cycle = core_flop_per_cycle_hifi2 * core_count
    ideal_cycles = mm_flops / chip_flop_per_cycle
    return ideal_cycles


def test_summarize_linear_performance():
    perf_results = []
    expected_results = []
    subdir = "ttnn_linear_performance"

    for idx, param_id in enumerate(PARAM_IDS):
        M, K, N = PARAMS[idx]

        # Run with tracy to get profiler results
        cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        command = f"pytest models/experimental/tt_dit/tests/unit/performance/test_linear.py::test_linear -k {param_id}"

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(subdir, cols, op_name="", has_signposts=False)

            core_count = int(r["CORE COUNT"])
            duration_ns = int(r["DEVICE KERNEL DURATION [ns]"])
            expected_ns = perf_model(M, K, N, core_count)

            perf_results.append(duration_ns)
            expected_results.append(expected_ns)
        except Exception as e:
            print(f"Error running test_linear_performance for {param_id}. Try running it alone to see the error.")
            perf_results.append(None)
            expected_results.append(None)
            continue

    # Pretty summary table
    header = "| M, K, N | measured perf (ms) | math util (%) | expected perf at 60% util (ms) |"
    sep = "|---|---:|---:|---:|"
    print(header)
    print(sep)
    for idx in range(len(PARAM_IDS)):
        M, K, N = PARAMS[idx]
        measured_ns = perf_results[idx]
        ideal_ns = expected_results[idx]

        if measured_ns is None or ideal_ns is None or measured_ns == 0:
            measured_ms_str = "-"
            util_str = "-"
            expected60_ms_str = "-"
        else:
            measured_ms = measured_ns / 1e6
            # Assume 1 cycle ≈ 1 ns for ideal estimate already returned from perf_model
            math_util = (ideal_ns / measured_ns) * 100.0
            expected60_ms = (ideal_ns / 0.60) / 1e6

            measured_ms_str = f"{measured_ms:.3f}"
            util_str = f"{math_util:.1f}"
            expected60_ms_str = f"{expected60_ms:.3f}"

        print(f"| ({M}, {K}, {N}) | {measured_ms_str} | {util_str} | {expected60_ms_str} |")
