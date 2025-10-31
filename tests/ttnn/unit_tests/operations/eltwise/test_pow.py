# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_allclose
from models.common.utility_functions import comp_ulp_check, debug_nonfinite_positions
import matplotlib.pyplot as plt
import os
import pandas as pd

import models.common.utility_functions as util


def plot_graphs_pow_FP64(input_tensor, golden, result, base, exponent, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # Format base and exponent for title
    base_str = "X" if base == "x" else str(base)
    exp_str = "X" if exponent == "x" else str(exponent)
    pow_label = f"POW({base_str}, {exp_str})"

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
    plt.title(f"{pow_label} - FP32 vs FP64 Golden: ULP Error per Input Value")
    plt.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, "pow_FP64_ulp_errors.png")
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
    plt.ylabel("pow Output")
    plt.title(f"{pow_label} - TTNN FP32 vs PyTorch FP64")
    # plt.ylim(0, 5)
    # plt.xlim(0, 5)
    plt.legend()
    save_path = os.path.join(out_dir, "ttnn_vs_torch_pow.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")


def plot_graphs_pow_fp32(input_tensor, golden, result, base, exponent, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # Format base and exponent for title
    base_str = "X" if base == "x" else str(base)
    exp_str = "X" if exponent == "x" else str(exponent)
    pow_label = f"POW({base_str}, {exp_str})"

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
    # plt.ylim(-1, 10)
    plt.title(f"{pow_label} - FP32 vs Golden: ULP Error per Input Value")
    plt.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, "pow_fp32_ulp_errors.png")
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
    plt.ylabel("pow Output")
    plt.title(f"{pow_label} - TTNN vs PyTorch")
    # plt.ylim(0, 5)
    # plt.xlim(0, 5)
    plt.legend()
    save_path = os.path.join(out_dir, "ttnn_vs_torch_pow.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")


@pytest.mark.parametrize(
    "base, exponent",
    [
        # ("x", 0.999),  # pow(x, 0.999)
        # (0.999, "x"),  # pow(0.999, x)
        (2.5, "x"),  # pow(2, x)
        # ("x", 2),  # pow(x, 2)
        # ("x", 3),  # pow(x, 3)
        # ("x", 9),  # pow(x, 10)
        # ("x", 2.56),  # pow(x, 10)
    ],
)
def test_pow_fp32(base, exponent, device):
    # pow Working range for fp32 - Overflow from 88.7(inf), Underflow till -87.3(<0)
    # low = -5.0
    # high = 5.0

    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # # # masking to working range
    # mask = (input_tensor >= low) & (input_tensor <= high)
    # input_tensor = input_tensor[mask]

    # Mask NaN
    mask = torch.isnan(input_tensor)
    input_tensor[mask] = 1.0

    if base == "x":
        base_tensor = input_tensor
    else:
        base_tensor = torch.full(input_tensor.shape, base, dtype=torch.float32)

    if exponent == "x":
        exponent_tensor = input_tensor
    else:
        exponent_tensor = torch.full(input_tensor.shape, exponent, dtype=torch.float32)

    # Convert to ttnn tensors
    tt_base = ttnn.from_torch(
        base_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_exponent = ttnn.from_torch(
        exponent_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compute golden result
    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(base_tensor, exponent_tensor, device=device)

    # Compute ttnn result
    tt_result = ttnn.pow(tt_base, tt_exponent)
    result = ttnn.to_torch(tt_result)
    debug_nonfinite_positions(input_tensor, exponent_tensor, golden, result)
    comp_ulp_check(base_tensor, exponent_tensor, golden, result, 1, allow_nonfinite=True)
    plot_graphs_pow_fp32(input_tensor, golden, result, base, exponent)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)
