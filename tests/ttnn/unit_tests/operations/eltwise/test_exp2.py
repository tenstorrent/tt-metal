# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import matplotlib.pyplot as plt
import os
from tests.ttnn.utils_for_testing import assert_with_ulp


def analyze_extreme_differences(input_tensor, calculated_tensor, expected_tensor, low, high):
    shape_str = str(list(input_tensor.shape))
    output_file = f"exp2_plot_shape_{shape_str.replace(' ', '')}_range_{low}_to_{high}.txt"
    assert input_tensor.shape == calculated_tensor.shape == expected_tensor.shape, "Tensors must have the same shape"

    extreme_input_values = []

    with open(output_file, "w") as f:
        difference = expected_tensor - calculated_tensor
        min_val = torch.min(difference)
        max_val = torch.max(difference)
        min_indices = torch.nonzero(difference == min_val)
        max_indices = torch.nonzero(difference == max_val)

        torch_inp = ttnn.to_torch(input_tensor)

        def write_details_at_indices(indices, label):
            for idx in indices:
                idx_tuple = tuple(idx.tolist())

                input_val = torch_inp[idx_tuple].item()
                extreme_input_values.append(input_val)

                f.write(f"{label} at index {idx_tuple}:\n")
                f.write(f"  Input tensor value     : {input_tensor[idx_tuple]}\n")
                f.write(f"  Calculated tensor value: {calculated_tensor[idx_tuple].item()}\n")
                f.write(f"  Expected tensor value  : {expected_tensor[idx_tuple].item()}\n")
                f.write(
                    f"  Abs Difference          : {abs((expected_tensor[idx_tuple] - calculated_tensor[idx_tuple]).item())}\n"
                )
                f.write("=" * 50 + "\n")

        f.write(f"\nMinimum difference: {min_val.item()}\n")
        write_details_at_indices(min_indices, "Minimum difference")

        f.write(f"\nMaximum difference: {max_val.item()}\n")
        write_details_at_indices(max_indices, "Maximum difference")
        f.write(f"\nInput values throwing max, min difference: {extreme_input_values}\n")

    print("\nInput values throwing max, min difference:", extreme_input_values)
    print("File : ", output_file)


def plot_input_vs_output(input_tensor, golden, calculated, low, high, test_name):
    shape_str = str(list(input_tensor.shape))
    input_flat = input_tensor.flatten().cpu().numpy()
    golden_flat = golden.flatten().cpu().numpy()
    calculated_flat = calculated.flatten().to(torch.float32).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(input_flat, golden_flat, label="Golden (FP32)", linewidth=2)
    plt.plot(input_flat, calculated_flat, label="TTNN (BF16)", linestyle="--")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"{test_name} | Shape: {shape_str} | Range: [{low}, {high}]")
    plt.legend()
    plt.grid(True)

    filename = f"exp2_plot_{test_name}_shape_{shape_str.replace(' ', '')}_range_{low}_to_{high}.png"
    filepath = os.path.join("plots", filename)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filepath)
    plt.close()

    print(f"--> Plot saved: {filepath}")


@pytest.mark.parametrize(
    "low, high, step",
    [
        (0.0, 10.1, 0.1),
        (10.0, 20.1, 0.1),
        (20.0, 30.1, 0.1),
        (30.0, 40.1, 0.1),
        (40.0, 50.1, 0.1),
        (50.0, 60.1, 0.1),
        (60.0, 70.1, 0.1),
        (70.0, 80.1, 0.1),
        (80.0, 90.1, 0.1),
        (90.0, 100.1, 0.1),
        (100.0, 110.1, 0.1),
        (110.0, 120.1, 0.1),
        (120.0, 127.1, 0.1),
    ],
)
def test_exp2(low, high, step, device):
    ttnn.set_printoptions(profile="full")

    input_tensor = torch.arange(low, high, step, dtype=torch.float32)  # to detect fractional differences

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    plot_input_vs_output(input_tensor, golden, result, low, high, "test_exp2")
    analyze_extreme_differences(tt_in, result, golden, low, high)
    assert_with_ulp(golden, result, 0)


def test_exp2_failedq(device):
    ttnn.set_printoptions(profile="full")
    input_values = [
        127.0,
        126.5,
        9.625,
        9.875,
        70.0,
        69.5,
        80.0,
        79.5,
        90.0,
        89.5,
        100.0,
        99.5,
        110.0,
        109.5,
        19.75,
        29.75,
        40.0,
        50.0,
        60.0,
    ]
    input_tensor = torch.tensor(input_values, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 0)


# @pytest.mark.parametrize(
#     "input_shapes",
#     (
#         (torch.Size([1, 1, 32, 32])),
#         (torch.Size([1, 2, 64, 120])),
#         (torch.Size([1, 3, 320, 320])),
#     ),
# )
# @pytest.mark.parametrize(
#     "low, high",
#     [
#         (-5, 5),
#         (-100, 100),
#     ],
# )
# def test_exp2_ULP(input_shapes, low, high, device):
#     num_elements = torch.prod(torch.tensor(input_shapes)).item()
#     torch_input = torch.linspace(high, low, num_elements, dtype=torch.float32)  # to detect fractional differences
#     torch_input = torch_input[:num_elements].reshape(input_shapes)

#     golden_function = ttnn.get_golden_function(ttnn.exp2)
#     golden = golden_function(torch_input, device=device)

#     tt_in = ttnn.from_torch(
#         torch_input,
#         dtype=ttnn.bfloat16,
#         device=device,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     )

#     tt_result = ttnn.exp2(tt_in)
#     result = ttnn.to_torch(tt_result)
#     plot_input_vs_output(torch_input, golden, result, low, high, "test_exp2")
#     # analyze_extreme_differences(tt_in, result, golden, low, high)
#     assert_with_ulp(golden, result, 0)
