# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

# Disable per module device fixture


from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    logger.info(f"PCC: {pcc_val:.7f}, Relative RMSE: {relative_rmse_val:.4f}")
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_moe(device, M, K, N, check_accuracy):
    logger.info(f"Running test_moe with M={M}, K={K}, N={N}, check_accuracy={check_accuracy}")

    if check_accuracy:
        torch_input = torch.randn((M, K), dtype=torch.float32)
        torch_w0 = torch.randn((K, 12 * 5 * 32), dtype=torch.float32)
        torch_w1 = torch.randn((K, 12 * 5 * 32), dtype=torch.float32)
        torch_w2 = torch.randn((N, 12 * 18 * 32), dtype=torch.float32)

    # Generate shard specs for weights
    shard_grid_all_12_dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})
    shard_grid_first_4_dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_grid_last_8_dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(11, 0))})

    weight0_1_shard_spec = ttnn.ShardSpec(shard_grid_all_12_dram, [K, 5 * 32], ttnn.ShardOrientation.ROW_MAJOR)
    weight0_1_shard_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, weight0_1_shard_spec
    )

    weight2_shard_spec = ttnn.ShardSpec(shard_grid_all_12_dram, [N, 18 * 32], ttnn.ShardOrientation.ROW_MAJOR)
    weight2_shard_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, weight2_shard_spec
    )

    # Create HEIGHT_SHARDED memory config for input
    # Each core (expert) gets a copy of the original (M, K) input
    input_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K),  # Shard shape - each core gets (M, K)
        core_grid=ttnn.CoreGrid(x=8, y=8),  # 64 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Prepare TT tensors
    if check_accuracy:
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight0 = ttnn.from_torch(
            torch_w0,
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight0_1_shard_memory_config,
        )
        tt_weight1 = ttnn.from_torch(
            torch_w1,
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight0_1_shard_memory_config,
        )
        tt_weight2 = ttnn.from_torch(
            torch_w2,
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight2_shard_memory_config,
        )
        tt_output = ttnn.empty((M, K), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        tt_input = ttnn.empty((M, K), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight0 = ttnn.empty(
            (K, 12 * 5 * 32),
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight0_1_shard_memory_config,
        )
        tt_weight1 = ttnn.empty(
            (K, 12 * 5 * 32),
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight0_1_shard_memory_config,
        )
        tt_weight2 = ttnn.empty(
            (N, 12 * 18 * 32),
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight2_shard_memory_config,
        )
        tt_output = ttnn.empty((M, K), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

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
    )
    tt_output = ttnn.to_torch(tt_output)
    if check_accuracy:
        return get_accuracy_metrics(torch_output, tt_output)
    return {}


SHAPE2TIME = {
    (32, 7168, 2048): 360_000,
    (32, 7168, 2048): 360.0,
}


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            },
            id="dispatch_col",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["default", "no_check"])
def test_moe(device, M, K, N, check_accuracy):
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
        check_accuracy,
    )

    if check_accuracy:
        assert accuracy_metrics["pcc"] > 0.999_500
        assert accuracy_metrics["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", ["default", "no_check"])
def test_moe_performance(M, K, N, check_accuracy):
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe.py::test_moe[{check_accuracy}-M={M}-K={K}-N={N}-dispatch_col]"

    run_device_profiler(command, "ttnn_moe_performance", device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log("ttnn_moe_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = int(r["DEVICE KERNEL DURATION [ns]"].min()) / 1000.0
    logger.info(f"Performance: {duration_us} us")

    assert (
        duration_us < SHAPE2TIME[(M, K, N)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
