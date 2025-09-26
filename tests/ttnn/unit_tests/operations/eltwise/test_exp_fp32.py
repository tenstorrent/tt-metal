# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import math
from tests.ttnn.utils_for_testing import assert_with_ulp
import matplotlib.pyplot as plt
import os
import pandas as pd

import models.utility_functions as util
from models.common.utility_functions import comp_ulp_check


def plot_graphs_exp_FP64(input_tensor, golden, result, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    x_vals = input_tensor.cpu().numpy()
    # ULP
    abs_error = torch.abs(golden - result)
    ulp_spacing = util.ulp(golden.to(torch.float32)).to(torch.float64)  # <-- FP64 spacing
    ulp_error = abs_error / ulp_spacing

    y_vals = ulp_error.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, y_vals, s=1, alpha=0.6)
    plt.xlabel("BF16-->FP64 Input Value")
    plt.ylabel("ULP Error")
    # plt.ylim(0, 10)
    # plt.xlim(0, 5)
    plt.title("exp FP32 vs FP64 Golden: ULP Error per Input Value")
    plt.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, "exp_FP64_ulp_errors.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"ULP scatter plot saved to: {out_file}")

    golden_vals = golden.to(torch.float32).cpu().numpy()
    result_vals = result.to(torch.float32).cpu().numpy()

    # TTNN vs Torch
    plt.figure(figsize=(12, 6))

    plt.scatter(x_vals, golden_vals, s=12, c="blue", label="PyTorch Golden", alpha=0.6, marker="x")
    plt.scatter(x_vals, result_vals, s=12, c="orange", label="TTNN Result", alpha=0.6, marker="o")

    plt.xlabel("Input")
    plt.ylabel("exp Output")
    plt.title("TTNN FP3 2vs PyTorch FP64")
    # plt.ylim(0, 5)
    # plt.xlim(0, 5)
    plt.legend()
    save_path = os.path.join(out_dir, "ttnn_vs_torch_exp.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")


def plot_graphs_exp_fp32(input_tensor, golden, result, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    x_vals = input_tensor.cpu().numpy()
    # ULP
    abs_error = torch.abs(golden - result)
    ulp_spacing = util.ulp(golden)  # <-- FP32 spacing
    ulp_error = abs_error / ulp_spacing

    y_vals = ulp_error.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, y_vals, s=1, alpha=0.6)
    plt.xlabel("BF16-->FP32 Input Value")
    plt.ylabel("ULP Error")
    # plt.ylim(0, 5)
    # plt.xlim(0, 5)
    plt.title("exp FP32 vs Golden: ULP Error per Input Value")
    plt.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, "exp_fp32_ulp_errors.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"ULP scatter plot saved to: {out_file}")

    golden_vals = golden.to(torch.float32).cpu().numpy()
    result_vals = result.to(torch.float32).cpu().numpy()

    # TTNN vs Torch
    plt.figure(figsize=(12, 6))

    plt.scatter(x_vals, golden_vals, s=12, c="blue", label="PyTorch Golden", alpha=0.6, marker="x")
    plt.scatter(x_vals, result_vals, s=12, c="orange", label="TTNN Result", alpha=0.6, marker="o")

    plt.xlabel("Input")
    plt.ylabel("exp Output")
    plt.title("TTNN vs PyTorch")
    # plt.ylim(0, 5)
    # plt.xlim(0, 5)
    plt.legend()
    save_path = os.path.join(out_dir, "ttnn_vs_torch_exp.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")


def test_exp_arange_masking(device):
    # Exp Working range for fp32 - Overflow from 88.7(inf), Underflow till -87.3(<0)
    low = -87.3
    high = 88.7

    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)
    # input_tensor = input_tensor.to(torch.float64)

    # masking to working range
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    # Mask NaN
    mask = torch.isnan(input_tensor)
    input_tensor[mask] = 1.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    comp_ulp_check(input_tensor, golden, result, 1, allow_nonfinite=True)
    plot_graphs_exp_fp32(input_tensor, golden, result)
    # plot_graphs_exp_FP64(input_tensor, golden, result)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 5, 5])),),
)
@pytest.mark.parametrize(
    "input_val",
    [
        (-0.0),
        (0.0),
        (1.0),
        (-1.0),
        (float("-inf")),
        (float("inf")),
        (float("nan")),
    ],
)
def test_exp_fill_val_bf16(input_shapes, input_val, device):
    torch_input = torch.ones(input_shapes, dtype=torch.float64) * input_val

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    torch.set_printoptions(precision=10)
    print("Input : ", torch_input)
    print("Torch : ", golden)
    print("TTNN : ", result)
    assert_with_ulp(golden, result, 1)
