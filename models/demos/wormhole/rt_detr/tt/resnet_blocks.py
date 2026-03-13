# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn


def conv_block(x, params, device, activation=None, **kwargs):
    # Configured memory to optimally leverage SRAM / cache mapping on conv.
    return ttnn.conv2d(
        input_tensor=x,
        weight_tensor=params.weight,
        bias_tensor=params.bias,
        device=device,
        activation=activation,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        **kwargs,
    )


def residual_block(x, params, device, stride=1, input_height=56, input_width=56):
    # Fused ReLU activation built directly into Convolution dispatch
    out, (h, w) = conv_block(
        x,
        params.conv1,
        device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        input_height=input_height,
        input_width=input_width,
        return_output_dim=True,
        activation="relu",
    )

    out, (h, w) = conv_block(
        out,
        params.conv2,
        device,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=(1, 1),
        input_height=h,
        input_width=w,
        return_output_dim=True,
        activation="relu",
    )

    out, (h, w) = conv_block(
        out,
        params.conv3,
        device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        input_height=h,
        input_width=w,
        return_output_dim=True,
    )

    # skip path
    if hasattr(params, "downsample"):
        skip, _ = conv_block(
            x,
            params.downsample[0],
            device,
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding=(0, 0),
            input_height=input_height,
            input_width=input_width,
            return_output_dim=True,
        )
    else:
        skip = x

    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    skip = ttnn.to_layout(skip, ttnn.TILE_LAYOUT)

    added = ttnn.add(out, skip, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.relu(added, memory_config=ttnn.L1_MEMORY_CONFIG), h, w
