# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# ResNet building blocks for PResNet-50.

import math
import ttnn

def _out_spatial(in_h, in_w, kernel, stride, padding):
    h = math.floor((in_h + 2 * padding[0] - kernel[0]) / stride[0]) + 1
    w = math.floor((in_w + 2 * padding[1] - kernel[1]) / stride[1]) + 1
    return h, w


def conv_block(x, params, device, kernel_size, stride, padding,
               input_height, input_width, activation=None):

    conv_config = ttnn.Conv2dConfig(
        deallocate_activation=False,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if activation == "relu" else None,
        reshard_if_not_optimal=True,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2, 
        math_approx_mode=False,                
    )

    out = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=params.weight,
        bias_tensor=params.bias,
        device=device,
        in_channels=x.shape[-1],
        out_channels=params.weight.shape[0],
        batch_size=x.shape[0],
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_config=conv_config,
        compute_config=compute_config,  
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_h, out_w = _out_spatial(input_height, input_width, kernel_size, stride, padding)
    return out, (out_h, out_w)


def residual_block(x, params, device, stride=1, input_height=56, input_width=56):
    # branch2a: 1x1, relu fused
    out, (h, w) = conv_block(
        x, params.conv1, device,
        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
        input_height=input_height, input_width=input_width,
        activation="relu",
    )
    # branch2b: 3x3, stride applied, relu fused
    out, (h, w) = conv_block(
        out, params.conv2, device,
        kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1),
        input_height=h, input_width=w,
        activation="relu",
    )
    # branch2c: 1x1, no activation - add happens after
    out, (h, w) = conv_block(
        out, params.conv3, device,
        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
        input_height=h, input_width=w,
    )

    # shortcut path
    if hasattr(params, "shortcut"):
        skip = x
        
        # ResNet-VD downsample: if stride=2, we Average Pool before the 1x1 Conv
        if stride == 2:
            skip = ttnn.avg_pool2d(
                skip,
                batch_size=skip.shape[0],
                input_h=input_height, input_w=input_width,
                channels=skip.shape[-1],
                kernel_size=[2, 2], stride=[2, 2], padding=[0, 0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            shortcut_stride = 1
            sh, sw = input_height // 2, input_width // 2
        else:
            shortcut_stride = 1
            sh, sw = input_height, input_width

        skip, _ = conv_block(
            skip, params.shortcut, device,
            kernel_size=(1, 1), stride=(shortcut_stride, shortcut_stride), padding=(0, 0),
            input_height=sh, input_width=sw,
        )
    else:
        skip = x

    out  = ttnn.to_layout(out,  ttnn.TILE_LAYOUT)
    skip = ttnn.to_layout(skip, ttnn.TILE_LAYOUT)

    # add + relu in L1
    result = ttnn.add(out, skip, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)
    ttnn.deallocate(skip)
    result = ttnn.relu(result, memory_config=ttnn.L1_MEMORY_CONFIG)
    result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)

    return result, h, w