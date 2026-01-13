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
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_moe(device, M, K, N, E, check_accuracy, dump_outputs):
    logger.info(
        f"Running test_moe with M={M}, K={K}, N={N}, E={E}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # We can restrict this to just the cores that are used, replicating it over all cores for now.
    in0_core_grid = device.compute_with_storage_grid_size()
    in0_num_cores = in0_core_grid.x * in0_core_grid.y

    # output is L1 sharded, the exact number of cores are kind of flexible
    # It just needs to divide the output shape (M, N) evenly.oo
    out_core_grid = ttnn.CoreGrid(x=8, y=8)
    out_num_cores = out_core_grid.x * out_core_grid.y

    in0_dtype = ttnn.bfloat8_b
    w0_dtype = ttnn.bfloat4_b

    if check_accuracy:
        torch_input = torch.randn((M, K), dtype=torch.bfloat16)
        torch_w0 = torch.randn((E, K, N), dtype=torch.bfloat16)
        torch_w1 = torch.randn((E, K, N), dtype=torch.bfloat16)
        torch_w2 = torch.randn((E, N, K), dtype=torch.bfloat16)

    # Each core (expert) gets a copy of the original (M, K) input
    input_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M * in0_num_cores, K),
        core_grid=ttnn.CoreGrid(x=in0_core_grid.x, y=in0_core_grid.y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Create WIDTH_SHARDED memory config for output
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N),
        core_grid=ttnn.CoreGrid(x=out_core_grid.x, y=out_core_grid.y),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Prepare TT tensors
    if check_accuracy:
        # Replicate input 64 times so each core gets a copy
        torch_input_replicated = torch_input.repeat(in0_num_cores, 1)  # Shape: (64*M, K)
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
            (in0_num_cores * M, K),
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_weight0 = ttnn.empty((E, K, N), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight1 = ttnn.empty((E, K, N), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight2 = ttnn.empty((E, N, K), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
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
        num_experts=E,
    )
    tt_output = ttnn.to_torch(tt_output)

    if check_accuracy and dump_outputs:
        torch.set_printoptions(profile="full")

        var2filename = {
            torch_w0_output: "torch_w0_output.txt",
            torch_w1_output: "torch_w1_output.txt",
            torch_w2_input: "torch_w2_input.txt",
            tt_output: "tt_output.txt",
        }
        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    if check_accuracy:
        return get_accuracy_metrics(torch_w2_input, tt_output)
    return {}


SHAPE2TIME = {
    (32, 7168, 2048): 102.0,
}


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
    "M, K, N, E",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe(device, M, K, N, E, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
        E,
        check_accuracy,
        dump_outputs,
    )

    if check_accuracy:
        assert accuracy_metrics["pcc"] > 0.999_500
        assert accuracy_metrics["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N, E",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe_performance(M, K, N, E, check_accuracy, dump_outputs):
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe.py::test_moe[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}-M={M}-K={K}-N={N}-E={E}-dispatch_row]"
    run_device_profiler(command, "ttnn_moe_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("ttnn_moe_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = int(r["DEVICE KERNEL DURATION [ns]"].min()) / 1000.0
    logger.warning(f"Performance: {duration_us} us")

    bytes_per_tile = 512 + 64
    total_tiles_0_1 = 224 * 64
    total_tiles_2 = 224 * 64
    total_tiles_per_core = 2 * total_tiles_0_1 + total_tiles_2
    total_bytes = total_tiles_per_core * bytes_per_tile
    bandwidth = total_bytes / (duration_us * 1000)
    logger.warning(f"Bandwidth: {bandwidth} GB/s")

    assert (
        duration_us < SHAPE2TIME[(M, K, N)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
