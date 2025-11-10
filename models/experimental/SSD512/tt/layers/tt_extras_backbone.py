# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


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
            layer_memory_config = ttnn.L1_MEMORY_CONFIG

            is_1x1_conv = kernel_size == (1, 1) or (kernel_size[0] == 1 and kernel_size[1] == 1)

            use_dram_slicing = (
                not use_l1_for_this_layer
                and (tensor_size_estimate > 1024 * 1024 or input_height > 64 or input_width > 64)
                and not is_1x1_conv
            )

            if use_dram_slicing:
                conv_bias = None
                slice_count = max(1, (batch_size * input_height * input_width) // (1024))
                slice_count = min(slice_count, 64)
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
                if hasattr(x, "memory_config") and x.memory_config() is not None:
                    if x.memory_config().is_sharded():
                        x = ttnn.sharded_to_interleaved(x, memory_config=memory_config)

                if x.layout != ttnn.TILE_LAYOUT:
                    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
                    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

                if hasattr(weight, "memory_config") and weight.memory_config() is not None:
                    if weight.memory_config().is_sharded():
                        weight = ttnn.sharded_to_interleaved(weight, memory_config=memory_config)

                if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
                    weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)

                slice_config = ttnn.Conv2dSliceConfig(
                    slice_type=ttnn.Conv2dDRAMSliceWidth,
                    num_slices=slice_count,
                )
            else:
                conv_bias = bias
                slice_config = ttnn.Conv2dL1FullSliceConfig

            compute_config = None
            if device is not None:
                compute_config = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                    math_approx_mode=False,
                )

            estimated_memory_bytes = input_height * input_width * in_channels  # 1 byte per element, bfloat8_b

            if use_l1_for_this_layer and not use_dram_slicing and estimated_memory_bytes < 1024 * 1024:
                if in_channels > 256 or out_channels > 512:
                    use_l1_for_this_layer = True
                    layer_memory_config = memory_config
                    conv_config = ttnn.Conv2dConfig(
                        weights_dtype=dtype,
                        shard_layout=None,
                        deallocate_activation=False,
                        enable_act_double_buffer=False,
                        enable_weights_double_buffer=False,
                        reshard_if_not_optimal=False,
                    )
                elif input_height <= 32:
                    act_block_h = 32
                    enable_double_buffer = True
                    conv_config = ttnn.Conv2dConfig(
                        weights_dtype=dtype,
                        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        deallocate_activation=False,
                        enable_act_double_buffer=enable_double_buffer,
                        enable_weights_double_buffer=False,
                        reshard_if_not_optimal=True,
                        act_block_h_override=act_block_h,
                        act_block_w_div=1,
                    )
                else:
                    # For 64x64 or larger, only use L1 if channels are small
                    if in_channels <= 128 and out_channels <= 256:
                        act_block_h = 32
                        enable_double_buffer = False
                        conv_config = ttnn.Conv2dConfig(
                            weights_dtype=dtype,
                            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            deallocate_activation=False,
                            enable_act_double_buffer=enable_double_buffer,
                            enable_weights_double_buffer=False,
                            reshard_if_not_optimal=True,
                            act_block_h_override=act_block_h,
                            act_block_w_div=1,
                        )
                    else:
                        use_l1_for_this_layer = True
                        layer_memory_config = memory_config
                        conv_config = ttnn.Conv2dConfig(
                            weights_dtype=dtype,
                            shard_layout=None,
                            deallocate_activation=False,
                            enable_act_double_buffer=False,
                            enable_weights_double_buffer=False,
                            reshard_if_not_optimal=False,
                        )
            else:
                use_l1_for_this_layer = True
                layer_memory_config = memory_config
                conv_config = ttnn.Conv2dConfig(
                    weights_dtype=dtype,
                    shard_layout=None,
                    deallocate_activation=False,
                    enable_act_double_buffer=False,
                    enable_weights_double_buffer=False,
                    reshard_if_not_optimal=False,
                )

            conv2d_kwargs = {
                "input_tensor": x,
                "weight_tensor": weight,
                "bias_tensor": conv_bias,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
                "batch_size": batch_size,
                "input_height": input_height,
                "input_width": input_width,
                "device": device,
                "return_output_dim": True,
                "return_weights_and_bias": True,
                "dtype": dtype,
                "slice_config": slice_config,
                "compute_config": compute_config,
                "conv_config": conv_config,
            }

            if layer_memory_config != ttnn.DRAM_MEMORY_CONFIG:
                conv2d_kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG

            output_tensor, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
                **conv2d_kwargs
            )

            output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
            if slice_config != ttnn.Conv2dL1FullSliceConfig and output_tensor.layout != ttnn.TILE_LAYOUT:
                output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

            if bias is not None and use_dram_slicing:
                output_tensor = output_tensor.reshape([batch_size, output_height, output_width, out_channels])
                bias_torch = ttnn.to_torch(bias)
                bias_expanded = bias_torch.expand(batch_size, output_height, output_width, out_channels)
                bias_full = ttnn.from_torch(
                    bias_expanded,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=layer_memory_config,
                )

                output_tensor = ttnn.add(output_tensor, bias_full, memory_config=layer_memory_config)

            layer["weight"] = prepared_weight
            if prepared_bias is not None and not use_dram_slicing:
                layer["bias"] = prepared_bias

            if bias is None or not use_dram_slicing:
                x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])
            else:
                x = output_tensor

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
