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
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    logger.info(f"PCC: {pcc_val:.7f}, Relative RMSE: {relative_rmse_val:.4f}")
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_moe(device, M, K, N):
    logger.info(f"Running test_moe with M={M}, K={K}, N={N}")

    torch_input = torch.randn((M, K), dtype=torch.float32)
    torch_w0 = torch.randn((K, N), dtype=torch.float32)
    torch_w1 = torch.randn((K, N), dtype=torch.float32)
    torch_w2 = torch.randn((N, K), dtype=torch.float32)

    # Prepare TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight0 = ttnn.from_torch(torch_w0, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight1 = ttnn.from_torch(torch_w1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight2 = ttnn.from_torch(torch_w2, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn.empty((M, K), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        w0_output = torch.nn.functional.silu(torch_input @ torch_w0)
        w1_output = torch_input @ torch_w1
        w2_input = w0_output * w1_output
        torch_output = w2_input @ torch_w2

    tt_output = ttnn.experimental.moe(
        tt_input,
        w0_tensor=tt_weight0,
        w1_tensor=tt_weight1,
        w2_tensor=tt_weight2,
        output_tensor=tt_output,
    )
    tt_output = ttnn.to_torch(tt_output)
    return get_accuracy_metrics(torch_output, tt_output)


TABLE_CONFIGS = [
    (32, 7168, 2048),
]


@pytest.mark.parametrize(
    "M, K, N",
    TABLE_CONFIGS,
)
def test_moe(device, M, K, N):
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
    )
    # assert accuracy_metrics["pcc"] > 0.999_500
    # assert accuracy_metrics["relative_rmse"] < 0.02


@pytest.mark.skip()
@pytest.mark.parametrize(
    "M, K, N",
    TABLE_CONFIGS,
)
def test_moe_performance(M, K, N):
    float_cols = ["DEVICE KERNEL DURATION [ns]"]
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe.py::test_moe[M={M}-K={K}-N={N}]"

    run_device_profiler(command, "ttnn_moe_performance", device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log("ttnn_moe_performance", float_columns=float_cols)
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
