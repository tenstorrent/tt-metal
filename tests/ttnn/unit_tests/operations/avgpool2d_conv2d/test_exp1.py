# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import tests.ttnn.unit_tests.operations.avgpool2d_conv2d.utils as utils

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("in_channels", [360])
@pytest.mark.parametrize("out_channels", [480])
@pytest.mark.parametrize("input_h", [27])
@pytest.mark.parametrize("input_w", [39])
@pytest.mark.parametrize("kernel_size", [[3, 3]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[1, 1, 1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])
def test_conv2d(
    batch_size, in_channels, out_channels, input_h, input_w, kernel_size, stride, padding, dilation, groups
):
    device = utils.DeviceGetter.get_device((1, 1))
    
    # Create torch tensors 
    torch_input_tensor = torch.ones((batch_size,1,input_h*input_w, in_channels), dtype=torch.bfloat16)  # [1, 1, 1053, 360]
    torch_weight_tensor = torch.ones((out_channels, in_channels, kernel_size[0], kernel_size[1]), dtype=torch.bfloat16)  # [480, 360, 3, 3]
    torch_bias_tensor = torch.ones((out_channels,), dtype=torch.bfloat16)  # [480]
    
    # Bias processing: [out_channels] -> [1, 1, 1, out_channels] using torch
    torch_bias_reshaped = torch.reshape(torch_bias_tensor, (1, 1, 1, out_channels))  # [1, 1, 1, 480]
    
    # Convert to ttnn tensors 
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.Layout.ROW_MAJOR, device=device)
    weight_tensor = ttnn.from_torch(torch_weight_tensor, layout=ttnn.Layout.ROW_MAJOR, device=device)
    bias_tensor_reshaped = ttnn.from_torch(torch_bias_reshaped, layout=ttnn.Layout.ROW_MAJOR, device=device)


    # ttnn conv2d
    output_tensor = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=27,
        input_width=39,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_tensor=bias_tensor_reshaped,
        conv_config=ttnn.Conv2dConfig(
            config_tensors_in_dram=True, enable_kernel_stride_folding=False
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )


