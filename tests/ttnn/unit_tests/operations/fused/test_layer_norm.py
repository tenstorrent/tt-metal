# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.unit_testing.misc.test_scaled_dot_product_attention import fa_rand


def create_normal_tensor_with_outliers_width_axis(
    height,
    width,
    mean_range=(0.0, 0.0),  # Range of means across width
    std_range=(1.0, 1.0),  # Range of standard deviations across width
    outlier_fraction=0.05,
    outlier_magnitude=5.0,
    device="cpu",
    dtype=torch.float32,
    seed=None,
):
    """
    Create a 2D PyTorch tensor where each column (width axis) has its own normal distribution with outliers.

    Args:
        height (int): Number of rows in the tensor
        width (int): Number of columns in the tensor
        mean_range (tuple): (min_mean, max_mean) - range of means across width axis
        std_range (tuple): (min_std, max_std) - range of standard deviations across width axis
        outlier_fraction (float): Fraction of values that should be outliers per column
        outlier_magnitude (float): How many standard deviations away outliers should be
        device (str or torch.device): Device to create tensor on
        dtype (torch.dtype): Data type of the tensor
        seed (int, optional): Random seed for reproducibility

    Returns:
        torch.Tensor: 2D tensor with column-wise normal distributions and outliers
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create empty tensor
    tensor = torch.zeros(height, width, device=device, dtype=dtype)

    # Generate parameters for each column
    min_mean, max_mean = mean_range
    min_std, max_std = std_range

    # Linear interpolation for means and stds across width
    means = torch.linspace(min_mean, max_mean, width, device=device, dtype=dtype)
    stds = torch.linspace(min_std, max_std, width, device=device, dtype=dtype)

    # Fill each column with its own normal distribution
    for col in range(width):
        col_mean = means[col].item()
        col_std = stds[col].item()

        # Generate normal distribution for this column
        tensor[:, col] = torch.normal(col_mean, col_std, size=(height,), device=device, dtype=dtype)

        # Add outliers for this column
        num_outliers = int(height * outlier_fraction)
        if num_outliers > 0:
            # Select random positions in this column for outliers
            outlier_indices = torch.randperm(height, device=device)[:num_outliers]

            # Generate outlier values for this column
            outlier_signs = torch.randint(0, 2, (num_outliers,), device=device) * 2 - 1
            outlier_signs = outlier_signs.to(dtype=dtype)

            # Add randomness to outlier magnitude
            magnitude_variation = torch.rand(num_outliers, device=device, dtype=dtype) * 2.0
            actual_magnitudes = outlier_magnitude + magnitude_variation

            outlier_values = col_mean + outlier_signs * actual_magnitudes * col_std
            tensor[outlier_indices, col] = outlier_values

    return tensor


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
def test_layer_norm(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.layer_norm(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [256])
def test_layer_norm_fa_rand(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = fa_rand(h, w)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.layer_norm(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
def test_layer_norm_mean_var(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = create_normal_tensor_with_outliers_width_axis(
        h, w, mean_range=(100.0, 100.0), std_range=(1.0, 1.0)
    )
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.layer_norm(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    torch.set_printoptions(profile="full")

    with open("Welfords.txt", "w") as f:
        f.write(str(output_tensor))
    with open("torch_tensor_output.txt", "w") as f:
        f.write(str(torch_output_tensor))

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_layer_norm_with_weight_and_bias(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_layer_norm_with_weight_bias_and_residual_input(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.layer_norm(input_tensor, residual_input_tensor=residual_input_tensor, weight=weight, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [512])
def test_layer_norm_with_tile_layout(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.ones(w, dtype=torch.bfloat16)
    torch_bias = torch.zeros(w, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor,
        (w,),
        torch_weight,
        torch_bias,
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    weight = ttnn.from_torch(torch_weight)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device)

    bias = ttnn.from_torch(torch_bias)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
    bias = ttnn.to_device(bias, device)

    output_tensor = ttnn.layer_norm(
        input_tensor,
        weight=weight,
        bias=bias,
    )

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
@pytest.mark.parametrize("d_type", [torch.bfloat16, torch.float32])
def test_large_layer_norm(device, h, w, d_type):
    torch.manual_seed(0)

    base_row = torch.arange(w, dtype=torch.float64).unsqueeze(0)  # shape: [1, 4096]

    # Create a column vector with exponents from 0 to rows-1
    exponents = torch.arange(h, dtype=torch.float64).unsqueeze(1)  # shape: [2048, 1]

    # Compute the multipliers as powers of 10 (10^0, 10^1, ..., 10^(rows-1))
    multipliers = 10**exponents  # shape: [2048, 1]

    # Multiply the multipliers with the base_row using broadcasting.
    # Each row becomes 10 times the previous row.
    tensor = multipliers * base_row  # shape: [2048, 4096]
    torch_input_tensor = torch.rand((h, w), dtype=d_type)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.layer_norm(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
def test_large_layer_norm_with_weight_and_bias(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
def test_large_layer_norm_with_weight(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], weight=torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.layer_norm(input_tensor, weight=weight)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
def test_large_layer_norm_with_bias(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.layer_norm(input_tensor, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)
