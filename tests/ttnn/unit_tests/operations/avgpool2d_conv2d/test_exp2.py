# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import tests.ttnn.unit_tests.operations.avgpool2d_conv2d.utils as utils


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("in_channels", [360])
@pytest.mark.parametrize("out_channels", [480])
@pytest.mark.parametrize("input_h", [28])
@pytest.mark.parametrize("input_w", [40])
@pytest.mark.parametrize("kernel_size", [[3, 3]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[1, 1, 1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])
def test_avgpool2d_conv2d(
    batch_size, in_channels, out_channels, input_h, input_w, kernel_size, stride, padding, dilation, groups
):
    device = utils.DeviceGetter.get_device((1, 1))

    # Create torch tensors 
    torch_input_tensor = torch.ones((batch_size, in_channels, input_h, input_w), dtype=torch.bfloat16)  # [1, 360, 28, 40]
    torch_weight_tensor = torch.ones((out_channels, in_channels, kernel_size[0], kernel_size[1]), dtype=torch.bfloat16)  # [480, 360, 3, 3]
    torch_bias_tensor = torch.ones((out_channels,), dtype=torch.bfloat16)  # [480]

    # ALL PREPROCESSING WITH TORCH OPERATIONS 
    # Step 1: Permute [batch_size, channels, H, W] -> [batch_size, H, W, channels]
    torch_input_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # [1, 28, 40, 360]
    
    # Step 2: Reshape [batch_size, H, W, channels] -> [1, 1, batch_size*H*W, channels]
    total_elements = batch_size * input_h * input_w  # 1 * 28 * 40 = 1120
    torch_input_reshaped = torch.reshape(torch_input_permuted, (1, 1, total_elements, in_channels))  # [1, 1, 1120, 360]
    
    # Step 3: Process bias tensor with torch operations
    # Bias processing: [out_channels] -> [1, 1, 1, out_channels] using torch
    torch_bias_reshaped = torch.reshape(torch_bias_tensor, (1, 1, 1, out_channels))  # [1, 1, 1, 480]

    # Convert to ttnn tensors 
    input_tensor = ttnn.from_torch(torch_input_reshaped, layout=ttnn.Layout.ROW_MAJOR, device=device)
    weight_tensor = ttnn.from_torch(torch_weight_tensor, layout=ttnn.Layout.ROW_MAJOR, device=device)
    bias_tensor = ttnn.from_torch(torch_bias_reshaped, layout=ttnn.Layout.ROW_MAJOR, device=device)

    # TTNN OPERATION 1: avg_pool2d 
    output_avgpool = ttnn.avg_pool2d(
        input_tensor,
        batch_size,
        input_h,
        input_w,
        in_channels,
        [2, 2],
        [1, 1],
        [0, 0],
        False,
        True,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        applied_shard_scheme=None,
        compute_kernel_config=None,
        reallocate_halo_output=False,
    )  # [1, 1, 1053, 360] where 1053 = 27 * 39
    ttnn.deallocate(input_tensor, False)

    # TTNN OPERATION 2: conv2d
    output_conv = ttnn.conv2d(
        input_tensor=output_avgpool,
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
        bias_tensor=bias_tensor,
        conv_config=ttnn.Conv2dConfig(
            config_tensors_in_dram=True, enable_kernel_stride_folding=False
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )  




