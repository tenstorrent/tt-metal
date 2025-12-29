# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("in_channels", [128])
@pytest.mark.parametrize("out_channels", [128])
@pytest.mark.parametrize("input_height", [124])
@pytest.mark.parametrize("input_width", [108])
@pytest.mark.parametrize("kernel_size", [[2, 2]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[0, 0]])
@pytest.mark.parametrize("output_padding", [[0, 0]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])
def test_convt2d_sanity(
    device,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
):
    torch.manual_seed(0)

    # Create torch input tensor: [batch_size, in_channels, input_height, input_width]
    torch_input_tensor = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.float32)

    # Create torch weight tensor: [in_channels, out_channels, kernel_h, kernel_w]
    torch_weight_tensor = torch.randn((in_channels, out_channels, kernel_size[0], kernel_size[1]), dtype=torch.float32)

    # Preprocess the input tensor
    # permute: [0, 2, 3, 1] -> [batch_size, input_height, input_width, in_channels]
    torch_input_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    # reshape: [1, 1, batch_size*input_height*input_width, in_channels]
    total_elements = batch_size * input_height * input_width
    torch_input_reshaped = torch.reshape(torch_input_permuted, (1, 1, total_elements, in_channels))

    # Convert to ttnn tensors
    input_tensor = ttnn.from_torch(
        torch_input_reshaped, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=device
    )

    weight_tensor = ttnn.from_torch(
        torch_weight_tensor, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=None
    )

    # ttnn conv_transpose2d
    output_tensor = ttnn.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        dtype=ttnn.DataType.FLOAT32,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
