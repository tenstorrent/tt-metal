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


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    # relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    return {
        "pcc": pcc_val,
        # "relative_rmse": relative_rmse_val,
    }


def run_test_moe(device, M, K, N, check_accuracy, w0_dtype, math_fidelity, fp32_dest_acc_en, dump_outputs):
    logger.info(
        f"Running test_moe with M={M}, K={K}, N={N}, check_accuracy={check_accuracy}, "
        f"w0_dtype={w0_dtype}, math_fidelity={math_fidelity}, fp32_dest_acc_en={fp32_dest_acc_en}"
    )

    core_grid = (4, 4)
    num_cores = core_grid[0] * core_grid[1]
    in0_dtype = ttnn.bfloat8_b

    if check_accuracy:
        torch_input = torch.randn((M, K), dtype=torch.bfloat16)
        torch_w0 = torch.randn((K, N), dtype=torch.bfloat16)
        torch_w1 = torch.randn((K, N), dtype=torch.bfloat16)
        torch_w2 = torch.randn((N, K), dtype=torch.bfloat16)

    # Create HEIGHT_SHARDED memory config for input
    # Each core (expert) gets a copy of the original (M, K) input
    input_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K),  # Shard shape - each core gets (M, K)
        core_grid=ttnn.CoreGrid(x=core_grid[0], y=core_grid[1]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create WIDTH_SHARDED memory config for output
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N // num_cores),
        core_grid=ttnn.CoreGrid(x=core_grid[0], y=core_grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Prepare TT tensors
    if check_accuracy:
        # Replicate input 64 times so each core gets a copy
        torch_input_replicated = torch_input.repeat(num_cores, 1)  # Shape: (64*M, K)
        tt_input = ttnn.from_torch(
            torch_input_replicated,
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight0_1_shard_memory_config,
        )
        tt_weight0 = ttnn.from_torch(torch_w0, dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight1 = ttnn.from_torch(torch_w1, dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight2 = ttnn.from_torch(torch_w2, dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        # Output is sharded (32, 2048) with each core having one tile (32x32)
    else:
        # Replicate empty input 64 times for performance testing
        tt_input = ttnn.empty(
            (num_cores * M, K),
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_weight0 = ttnn.empty((K, N), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight1 = ttnn.empty((K, N), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight2 = ttnn.empty((N, K), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        # Output is sharded (32, 2048) with each core having one tile (32x32)

    tt_output = ttnn.empty(
        (M, N),
        dtype=in0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_sharded_mem_config,
    )

    if check_accuracy:
        with torch.no_grad():
            torch_w0_output = torch.nn.functional.silu(torch_input @ torch_w0)
            torch_w1_output = torch_input @ torch_w1
            torch_w2_input = torch_w0_output * torch_w1_output
            torch_output = torch_w2_input @ torch_w2

    tt_output = ttnn.experimental.moe(
        tt_input,
        w0_tensor=tt_weight0,
        w1_tensor=tt_weight1,
        w2_tensor=tt_weight2,
        output_tensor=tt_output,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    tt_output = ttnn.to_torch(tt_output)

    if check_accuracy and dump_outputs:
        var2filename = {
            torch_w0_output: "torch_w0_output.txt",
            torch_w1_output: "torch_w1_output.txt",
            torch_w2_input: "torch_w2_input.txt",
            tt_output: "tt_output.txt",
        }
        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

            # torch.save(var, filename)

    if check_accuracy:
        return get_accuracy_metrics(torch_w2_input, tt_output)
    return {}


SHAPE2TIME = {
    (32, 7168, 2048): 360.0,
}

# Configuration: (w0_dtype, math_fidelity, fp32_dest_acc_en, id_string)
COMPUTE_CONFIGS = [
    # (ttnn.bfloat16, ttnn.MathFidelity.LoFi, True, "bf16_lofi"),
    # (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True, "bf16_hifi"),
    # (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, True, "bf8b_lofi"),
    # (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, "bf8b_hifi"),
    # (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, "bf4b_lofi"),
    # (ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, True, "bf4b_hifi"),
    (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False, "bf4b_lofi_no_fp32acc"),
]


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize(
    "w0_dtype, math_fidelity, fp32_dest_acc_en",
    [(cfg[0], cfg[1], cfg[2]) for cfg in COMPUTE_CONFIGS],
    ids=[cfg[3] for cfg in COMPUTE_CONFIGS],
)
@pytest.mark.parametrize(
    "check_accuracy",
    [True],
    ids=[
        "acc_check",
    ],
)
@pytest.mark.parametrize(
    "dump_outputs",
    [False],
    ids=[
        "no_dump_outputs",
    ],
)
def test_moe(device, M, K, N, check_accuracy, w0_dtype, math_fidelity, fp32_dest_acc_en, dump_outputs):
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
        check_accuracy,
        w0_dtype,
        math_fidelity,
        fp32_dest_acc_en,
        dump_outputs,
    )

    # if check_accuracy:
    # assert accuracy_metrics["pcc"] > 0.999_500
    # assert accuracy_metrics["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", ["acc_check", "no_acc_check"])
@pytest.mark.parametrize("dump_outputs", ["dump_outputs", "no_dump_outputs"])
def test_moe_performance(M, K, N, check_accuracy, dump_outputs):
    # pytest -k matches test IDs, not parameter values. The test ID format is:
    # test_moe[dispatch_col-{M}-{K}-{N}-{config_id}-{check_accuracy_id}-{dump_outputs_id}]
    # So we match the numeric values and the ID strings
    command = f'pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe.py::test_moe -k "{M} and {K} and {N} and {check_accuracy} and {dump_outputs}"'

    run_device_profiler(command, "ttnn_moe_performance", device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log("ttnn_moe_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = int(r["DEVICE KERNEL DURATION [ns]"].min()) / 1000.0
    logger.info(f"Performance: {duration_us} us")
    duration_us = int(r["DEVICE KERNEL DURATION [ns]"].min()) / 1000.0
    logger.info(f"Performance: {duration_us} us")

    assert (
        duration_us < SHAPE2TIME[(M, K, N)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N)]} us"
        duration_us < SHAPE2TIME[(M, K, N)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
