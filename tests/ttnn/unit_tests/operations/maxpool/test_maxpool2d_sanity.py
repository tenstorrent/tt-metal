# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("batch_size", [6])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("input_h", [464])
@pytest.mark.parametrize("input_w", [800])
@pytest.mark.parametrize("kernel_size", [[3, 3]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("ceil_mode", [False])
def test_maxpool2d_sanity(
    device, batch_size, channels, input_h, input_w, kernel_size, stride, padding, dilation, ceil_mode
):
    torch.manual_seed(0)

    # Create torch input tensor
    torch_input_tensor = torch.randn((batch_size, channels, input_h, input_w), dtype=torch.float32)

    # Preprocess the tensor
    # permute: [0, 2, 3, 1] -> [batch_size, input_h, input_w, channels]
    torch_input_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    # reshape: [1, 1, batch_size*input_h*input_w, channels]
    total_elements = batch_size * input_h * input_w
    torch_input_reshaped = torch.reshape(torch_input_permuted, (1, 1, total_elements, channels))

    # Convert to ttnn tensor
    input_tensor = ttnn.from_torch(
        torch_input_reshaped, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=device
    )

    # ttnn max_pool2d
    output_tensor = ttnn.max_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
    )
