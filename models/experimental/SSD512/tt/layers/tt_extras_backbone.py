# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    BlockShardedStrategyConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
    WidthSliceStrategyConfiguration,
)


def extras_backbone(cfg, input_channels=1024, batch_norm=False, device=None):
    """Build extras layers using TTNN operations."""
    layers = []
    in_channels = input_channels
    flag = False

    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                # Stride=2 downsampling layer
                if k + 1 >= len(cfg):
                    raise ValueError(f"'S' marker at index {k} but no next element in cfg")
                out_channels = cfg[k + 1]
                kernel_size = (1, 3)[flag]  # Alternating: 1 or 3

                layers.append(
                    {
                        "type": "conv",
                        "config": {
                            "kernel_size": (kernel_size, kernel_size),
                            "stride": (2, 2),
                            "padding": (1, 1),
                            "dilation": (1, 1),
                            "groups": 1,
                        },
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                    }
                )
                layers.append(
                    {
                        "type": "relu",
                    }
                )
                flag = not flag  # Toggle for next layer
            else:
                # Regular layer (stride=1, no padding)
                out_channels = v
                kernel_size = (1, 3)[flag]

                layers.append(
                    {
                        "type": "conv",
                        "config": {
                            "kernel_size": (kernel_size, kernel_size),
                            "stride": (1, 1),
                            "padding": (0, 0),
                            "dilation": (1, 1),
                            "groups": 1,
                        },
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                    }
                )
                layers.append(
                    {
                        "type": "relu",
                    }
                )
                flag = not flag

        in_channels = v

    if len(cfg) == 13:
        layers.append(
            {
                "type": "conv",
                "config": {
                    "kernel_size": (4, 4),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                },
                "in_channels": in_channels,
                "out_channels": 256,
            }
        )
        layers.append(
            {
                "type": "relu",
            }
        )

    return layers


def create_extras_layers_with_weights(layers_config, device=None, dtype=ttnn.bfloat8_b):
    """
    Create extras layers with initialized weights and biases.
    """
    from models.experimental.SSD512.common import create_conv2d_weights_and_bias

    layers_with_weights = []

    for layer in layers_config:
        if layer["type"] == "conv":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["config"]["kernel_size"]

            # Use common function to create weights and bias
            weight_ttnn, bias_ttnn = create_conv2d_weights_and_bias(
                in_channels, out_channels, kernel_size, device=device, dtype=dtype, init_method="kaiming_normal"
            )

            layer_with_weights = layer.copy()
            layer_with_weights["weight"] = weight_ttnn
            layer_with_weights["bias"] = bias_ttnn
            layers_with_weights.append(layer_with_weights)
        else:
            # For relu layers, just copy the config
            layers_with_weights.append(layer.copy())

    return layers_with_weights


def apply_extras_backbone(input_tensor, layers_with_weights, device=None, dtype=ttnn.bfloat8_b, memory_config=None):
    """
    Apply extras layers to input tensor using TTNN operations."""
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    if isinstance(input_tensor, torch.Tensor):
        x_torch = input_tensor.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = ttnn.from_torch(
            x_torch,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
    else:
        x = input_tensor

    if isinstance(input_tensor, torch.Tensor):
        batch_size = 1
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]
        current_channels = input_tensor.shape[1]
    else:
        shape = x.shape
        batch_size = 1
        input_height = shape[1]
        input_width = shape[2]
        current_channels = shape[3]

    # Track current dimensions
    current_h = input_height
    current_w = input_width
    current_c = current_channels

    for layer_idx, layer in enumerate(layers_with_weights):
        if layer["type"] == "conv":
            weight = layer["weight"]
            bias = layer.get("bias", None)
            config = layer["config"]
            input_height = current_h
            input_width = current_w
            in_channels = current_c
            out_channels = layer["out_channels"]
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            padding = config["padding"]
            dilation = config["dilation"]
            groups = config["groups"]

            tensor_size_estimate = batch_size * input_height * input_width * in_channels
            use_l1_for_this_layer = (
                input_height <= 128 and input_width <= 128 and tensor_size_estimate <= 2 * 1024 * 1024
            )

            is_1x1_conv = kernel_size == (1, 1) or (kernel_size[0] == 1 and kernel_size[1] == 1)

            use_dram_slicing = (
                not use_l1_for_this_layer
                and (tensor_size_estimate > 1024 * 1024 or input_height > 64 or input_width > 64)
                and not is_1x1_conv
            )

            if is_1x1_conv and stride == (1, 1) and padding == (0, 0):
                if isinstance(weight, torch.Tensor):
                    weight_torch = weight
                else:
                    weight_torch = ttnn.to_torch(weight)

                weight_2d_torch = weight_torch.reshape(out_channels, in_channels).permute(1, 0)

                layer_memory_config = ttnn.L1_MEMORY_CONFIG if use_l1_for_this_layer else memory_config

                if device is not None:
                    weight_2d = ttnn.from_torch(
                        weight_2d_torch,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=layer_memory_config,
                    )
                else:
                    weight_2d = ttnn.from_torch(weight_2d_torch, device=None, dtype=dtype, layout=ttnn.TILE_LAYOUT)

                if x.layout != ttnn.TILE_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

                x_flat = ttnn.reshape(x, (batch_size * current_h * current_w, in_channels))

                out_flat = ttnn.matmul(x_flat, weight_2d, memory_config=ttnn.L1_MEMORY_CONFIG)

                output_tensor = ttnn.reshape(out_flat, (batch_size, current_h, current_w, out_channels))

                if bias is not None:
                    if isinstance(bias, torch.Tensor):
                        bias_torch = bias
                    else:
                        bias_torch = ttnn.to_torch(bias)

                    bias_reshaped = bias_torch.reshape((out_channels,))

                    if device is not None:
                        bias_1d = ttnn.from_torch(
                            bias_reshaped,
                            device=device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=layer_memory_config,
                        )
                    else:
                        bias_1d = ttnn.from_torch(bias_reshaped, device=None, dtype=dtype, layout=ttnn.TILE_LAYOUT)
                    output_tensor = ttnn.add(output_tensor, bias_1d, memory_config=layer_memory_config)

                output_height = current_h
                output_width = current_w
            else:
                if isinstance(weight, ttnn.Tensor):
                    weight_ttnn = weight
                    if bias is not None:
                        bias_ttnn = bias if isinstance(bias, ttnn.Tensor) else None
                        if bias_ttnn is None:
                            bias_torch = bias if isinstance(bias, torch.Tensor) else ttnn.to_torch(bias)
                            bias_reshaped = bias_torch.reshape((1, 1, 1, -1))
                            bias_ttnn = ttnn.from_torch(bias_reshaped, dtype=ttnn.float32)
                    else:
                        bias_ttnn = None
                else:
                    weight_torch = weight if isinstance(weight, torch.Tensor) else ttnn.to_torch(weight)
                    bias_torch = None
                    if bias is not None:
                        bias_torch = bias if isinstance(bias, torch.Tensor) else ttnn.to_torch(bias)
                    weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
                        weight_torch, bias_torch
                    )

                estimated_memory_bytes = input_height * input_width * in_channels

                if use_dram_slicing:
                    slice_count = max(1, (batch_size * input_height * input_width) // (1024))
                    slice_count = min(slice_count, 64)
                    slice_strategy = WidthSliceStrategyConfiguration(num_slices=slice_count)
                    sharding_strategy = AutoShardedStrategyConfiguration()
                    enable_act_double_buffer = False
                    deallocate_activation = False
                elif use_l1_for_this_layer and estimated_memory_bytes < 1024 * 1024:
                    if in_channels > 256 or out_channels > 512:
                        sharding_strategy = AutoShardedStrategyConfiguration()
                        slice_strategy = None
                        enable_act_double_buffer = False
                        deallocate_activation = False
                    elif input_height <= 32:
                        act_block_h = 32
                        enable_act_double_buffer = True
                        sharding_strategy = BlockShardedStrategyConfiguration(
                            act_block_h_override=act_block_h,
                            act_block_w_div=1,
                            reshard_if_not_optimal=True,
                        )
                        slice_strategy = L1FullSliceStrategyConfiguration()
                        deallocate_activation = False
                    else:
                        if in_channels <= 128 and out_channels <= 256:
                            act_block_h = 32
                            enable_act_double_buffer = False
                            sharding_strategy = BlockShardedStrategyConfiguration(
                                act_block_h_override=act_block_h,
                                act_block_w_div=1,
                                reshard_if_not_optimal=True,
                            )
                            slice_strategy = L1FullSliceStrategyConfiguration()
                            deallocate_activation = False
                        else:
                            sharding_strategy = AutoShardedStrategyConfiguration()
                            slice_strategy = None
                            enable_act_double_buffer = False
                            deallocate_activation = False
                else:
                    sharding_strategy = AutoShardedStrategyConfiguration()
                    slice_strategy = None
                    enable_act_double_buffer = False
                    deallocate_activation = False

                conv_config = Conv2dConfiguration(
                    input_height=input_height,
                    input_width=input_width,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    batch_size=batch_size,
                    kernel_size=kernel_size,
                    weight=weight_ttnn,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias_ttnn,
                    activation_dtype=dtype,
                    weights_dtype=dtype,
                    output_dtype=dtype,
                    sharding_strategy=sharding_strategy,
                    slice_strategy=slice_strategy,
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                    enable_act_double_buffer=enable_act_double_buffer,
                    enable_weights_double_buffer=False,
                    deallocate_activation=deallocate_activation,
                    reallocate_halo_output=True,
                    config_tensors_in_dram=not use_l1_for_this_layer,
                )

                conv_layer = TtConv2d(conv_config, device)
                output_tensor = conv_layer(x)

                padding_h, padding_w = padding
                dilation_h, dilation_w = dilation
                stride_h, stride_w = stride
                kernel_h, kernel_w = kernel_size

                output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
                output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

            if output_tensor.is_sharded():
                output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config)
            x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])

            current_h = output_height
            current_w = output_width
            current_c = out_channels

        elif layer["type"] == "relu":
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.relu(x)

    return x


# configuration dictionaries
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
}


def build_extras_backbone(size=512, input_channels=1024, device=None):
    """
    Build extras backbone for specified input size.
    """
    if size not in [512]:
        raise ValueError(f"Size must be 512, got {size}")

    cfg = extras[str(size)]
    layers_config = extras_backbone(cfg, input_channels=input_channels, device=device)

    return layers_config
