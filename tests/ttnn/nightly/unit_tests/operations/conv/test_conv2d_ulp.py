# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_ulp
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_dtype,ulp_threshold",
    [
        (ttnn.bfloat16, 1),
        (ttnn.float32, 30000),
    ],
)
@pytest.mark.parametrize("input_channels", [1024])
def test_conv_ulp(device, input_dtype, ulp_threshold, input_channels):
    """Test 3x3 convolution with bfloat16 and fp32 inputs with fp32 accumulation enabled"""
    # These params will produce 96,K x K,96 matmul as a part of conv2d computation
    # where K = input_channels * Kh * Kw = input_channels * 3 * 3 = input_channels * 9
    batch_size = 1
    out_channels = 96
    input_height = 12
    input_width = 8
    kernel_size = 3
    stride = 1
    padding = 1

    # Determine torch dtype based on input_dtype - all tensors use same dtype
    if input_dtype == ttnn.bfloat16:
        torch_dtype = torch.bfloat16
    else:  # ttnn.float32
        torch_dtype = torch.float32

    # Generate random inputs - all with matching dtype
    torch.manual_seed(42)
    torch_input_nchw = torch.rand(batch_size, input_channels, input_height, input_width, dtype=torch_dtype)
    torch_weight = torch.rand(out_channels, input_channels, kernel_size, kernel_size, dtype=torch_dtype)
    torch_bias = torch.rand(1, 1, 1, out_channels, dtype=torch_dtype)

    # Convert input to NHWC for ttnn
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))

    # Compute reference output with torch - all tensors use same dtype
    torch_output = torch.nn.functional.conv2d(
        torch_input_nchw,
        torch_weight,
        bias=torch_bias.reshape(-1),
        stride=stride,
        padding=padding,
    )

    # Convert to ttnn tensors - all use the same dtype
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=input_dtype, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=input_dtype)
    tt_bias = ttnn.from_torch(torch_bias, dtype=input_dtype)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,  # Enable fp32 accumulation
        packer_l1_acc=True,
    )

    # Run conv2d
    [tt_output, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=input_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        compute_config=compute_config,
        return_output_dim=True,
    )

    # Convert output back to torch and compare
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.reshape(batch_size, out_height, out_width, out_channels)
    tt_output_torch = torch.permute(tt_output_torch, (0, 3, 1, 2))

    assert_with_ulp(torch_output, tt_output_torch, ulp_threshold)
