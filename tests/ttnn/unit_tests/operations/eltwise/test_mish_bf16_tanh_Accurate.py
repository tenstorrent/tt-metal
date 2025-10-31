import torch
import matplotlib.pyplot as plt
import os
import ttnn
import pandas as pd
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import comp_ulp_check

import models.utility_functions as util


def test_mish_arange(device):
    # Generate all possible bit pattersn for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    mask = (
        torch.isnan(input_tensor) | torch.isinf(input_tensor) | (input_tensor == 3.3895313892515355e38)
    )  # Input -inf -> Golden: nan, Result: -3.076e+35
    input_tensor[mask] = 1.0

    # softplus Working range
    low = -20.0
    high = float("inf")
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.mish)
    golden = golden_function(input_tensor, device=device)

    tt_exp = ttnn.exp(tt_in)
    tt_log1p = ttnn.log1p(tt_exp)
    tt_tanh = ttnn.tanh(tt_log1p, fast_and_approximate_mode=False)
    tt_result = ttnn.multiply(tt_in, tt_tanh)
    result = ttnn.to_torch(tt_result)
    comp_ulp_check(input_tensor, golden, result, 1, allow_nonfinite=True)
    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


def test_mish_arange_plot(device, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16).to(torch.float32)

    mask = torch.isnan(input_tensor) | torch.isinf(input_tensor)  # Input -inf -> Golden: nan, Result: -3.076e+35
    input_tensor[mask] = 1.0

    # softplus Working range
    low = -87.0
    high = float("inf")
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    x_vals = input_tensor.cpu().numpy()

    golden_function = ttnn.get_golden_function(ttnn.mish)
    golden = golden_function(input_tensor, device=device)

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_exp = ttnn.exp(tt_in)
    tt_log1p = ttnn.log1p(tt_exp)
    tt_tanh = ttnn.tanh(tt_log1p, fast_and_approximate_mode=False)
    tt_result = ttnn.multiply(tt_in, tt_tanh)
    result = ttnn.to_torch(tt_result)

    # ULP
    abs_error = torch.abs(golden - result)
    ulp_spacing = util.ulp(golden.to(torch.bfloat16)).to(torch.float32)
    ulp_error = abs_error / ulp_spacing

    y_vals = ulp_error.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, y_vals, s=1, alpha=0.6)
    plt.xlabel("BF16 Input Value (converted to FP32)")
    plt.ylabel("ULP Error")
    # plt.ylim(-0.5, 10)
    # plt.xlim(-21, 1)
    plt.title("mish BF16 vs Golden: ULP Error per Input Value")
    plt.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, "mish_bf16_ulp_errors.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"ULP scatter plot saved to: {out_file}")

    df = pd.DataFrame({"input_value": x_vals, "ulp_error": y_vals})
    csv_file = os.path.join(out_dir, "mish_bf16_ulp_errors.csv")
    df.to_csv(csv_file, index=False)
    print(f"ULP error data saved to: {csv_file}")

    golden_vals = golden.to(torch.float32).cpu().numpy()
    result_vals = result.to(torch.float32).cpu().numpy()

    # TTNN vs Torch
    plt.figure(figsize=(12, 6))

    plt.scatter(x_vals, golden_vals, s=12, c="blue", label="PyTorch Golden", alpha=0.6, marker="x")
    plt.scatter(x_vals, result_vals, s=12, c="orange", label="TTNN Result", alpha=0.6, marker="o")

    plt.xlabel("Input (bfloat16 values)")
    plt.ylabel("mish Output")
    plt.title("TTNN vs PyTorch (mish outputs for all bf16 values)")
    # plt.ylim(-0.5, 3)
    # plt.xlim(-21, 3)
    plt.legend()
    save_path = os.path.abspath("ttnn_vs_torch_mish.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")
