import torch
import matplotlib.pyplot as plt
import os
import ttnn
import pandas as pd
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import comp_ulp_check

import models.utility_functions as util


def test_softmax_arange(device):
    # Generate all possible bit pattersn for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # Mask NaN, special values where softmax has ULP>1 (Covered in atol test below).
    mask = torch.isnan(input_tensor)
    input_tensor[mask] = 1.0

    # Exp Working range - Overflow from 88.5(inf), Underflow till -87(<0) to avoid AssertionError: Tensors are not finite at the same positions
    low = -87.0
    high = 88.5
    # masking to working range
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.softmax)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.softmax(tt_in)
    result = ttnn.to_torch(tt_result)
    comp_ulp_check(input_tensor, golden, result, 1, allow_nonfinite=True)
    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


def test_softmax_arange_plot(device, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16).to(torch.float32)

    # Mask NaNs
    mask = (torch.isnan(input_tensor)) | (input_tensor == 3.3895313892515355e38)
    input_tensor[mask] = 1.0

    # Exp Working range - Overflow from 88.5(inf), Underflow till -87(<0) to avoid AssertionError: Tensors are not finite at the same positions
    low = -87.0
    high = 88.5
    # masking to working range
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    x_vals = input_tensor.cpu().numpy()

    golden_function = ttnn.get_golden_function(ttnn.softmax)
    golden = golden_function(input_tensor, device=device)

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.softmax(tt_in)
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
    # plt.ylim(0, 40)
    # plt.xlim(-5, 1.2e-38)
    plt.title("softmax BF16 vs Golden: ULP Error per Input Value")
    plt.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, "softmax_bf16_ulp_errors.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"ULP scatter plot saved to: {out_file}")

    df = pd.DataFrame({"input_value": x_vals, "ulp_error": y_vals})
    csv_file = os.path.join(out_dir, "softmax_bf16_ulp_errors.csv")
    df.to_csv(csv_file, index=False)
    print(f"ULP error data saved to: {csv_file}")

    golden_vals = golden.to(torch.float32).cpu().numpy()
    result_vals = result.to(torch.float32).cpu().numpy()

    # TTNN vs Torch
    plt.figure(figsize=(12, 6))

    plt.scatter(x_vals, golden_vals, s=12, c="blue", label="PyTorch Golden", alpha=0.6, marker="x")
    plt.scatter(x_vals, result_vals, s=12, c="orange", label="TTNN Result", alpha=0.6, marker="o")

    plt.xlabel("Input (bfloat16 values)")
    plt.ylabel("softmax Output")
    plt.title("TTNN vs PyTorch (softmax outputs for all bf16 values)")
    # plt.ylim(-1, 0)
    # plt.xlim(-5, 1.2e-38)
    plt.legend()
    save_path = os.path.abspath("ttnn_vs_torch_softmax.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")
