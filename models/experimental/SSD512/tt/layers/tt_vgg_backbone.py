# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def vgg_backbone(cfg, input_channels=3, batch_norm=False, device=None):
    """
    Build VGG backbone layers using TTNN operations.
    """
    layers = []
    in_channels = input_channels

    # Build layers from configuration
    for v in cfg:
        if v == "M":
            # Standard MaxPool2d: kernel_size=2, stride=2, padding=0, ceil_mode=False
            layers.append(
                {
                    "type": "pool",
                    "config": {
                        "kernel_size": (2, 2),
                        "stride": (2, 2),
                        "padding": (0, 0),
                        "dilation": (1, 1),
                        "ceil_mode": False,
                    },
                }
            )
        elif v == "C":
            # MaxPool2d with ceil_mode=True: kernel_size=2, stride=2, padding=0, ceil_mode=True
            layers.append(
                {
                    "type": "pool",
                    "config": {
                        "kernel_size": (2, 2),
                        "stride": (2, 2),
                        "padding": (0, 0),
                        "dilation": (1, 1),
                        "ceil_mode": True,
                    },
                }
            )
        else:
            # Standard 3x3 Conv2d: kernel_size=3, stride=1, padding=1, dilation=1
            layers.append(
                {
                    "type": "conv",
                    "config": {
                        "kernel_size": (3, 3),
                        "stride": (1, 1),
                        "padding": (1, 1),
                        "dilation": (1, 1),
                        "groups": 1,
                    },
                    "in_channels": in_channels,
                    "out_channels": v,
                }
            )
            layers.append(
                {
                    "type": "relu",
                }
            )
            in_channels = v

    # Add pool5: kernel_size=3, stride=1, padding=1, ceil_mode=False
    layers.append(
        {
            "type": "pool",
            "config": {
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
                "ceil_mode": False,
            },
        }
    )

    # Add conv6: kernel_size=3, stride=1, padding=6, dilation=6
    layers.append(
        {
            "type": "conv",
            "config": {
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (6, 6),
                "dilation": (6, 6),
                "groups": 1,
            },
            "in_channels": 512,
            "out_channels": 1024,
        }
    )
    layers.append(
        {
            "type": "relu",
        }
    )

    # Add conv7: kernel_size=1, stride=1, padding=0, dilation=1
    layers.append(
        {
            "type": "conv",
            "config": {
                "kernel_size": (1, 1),
                "stride": (1, 1),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
            },
            "in_channels": 1024,
            "out_channels": 1024,
        }
    )
    layers.append(
        {
            "type": "relu",
        }
    )

    return layers


def create_vgg_layers_with_weights(layers_config, device=None, dtype=ttnn.bfloat16):
    """
    Create VGG layers with initialized weights and biases.
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
            layers_with_weights.append(layer.copy())

    return layers_with_weights


def apply_vgg_backbone(
    input_tensor, layers_with_weights, device=None, dtype=ttnn.bfloat8_b, memory_config=None, return_sources=None
):
    """
    Apply VGG backbone layers to input tensor using TTNN operations.
    """
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    # Verify input tensor shape before conversion
    if isinstance(input_tensor, torch.Tensor):
        expected_shape = (1, 3, 512, 512)
        if input_tensor.shape != expected_shape:
            raise ValueError(
                f"Input tensor shape mismatch in VGG backbone: expected {expected_shape}, got {input_tensor.shape}"
            )

    # Convert input to TTNN format if it's a torch tensor
    if isinstance(input_tensor, torch.Tensor):
        x_torch = input_tensor.permute(0, 2, 3, 1)
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

        # Verify dimensions are 512x512
        if input_height != 512 or input_width != 512:
            raise ValueError(f"Image dimensions must be 512x512, got {input_height}x{input_width}")
    else:
        shape = x.shape
        batch_size = 1
        input_height = shape[1]
        input_width = shape[2]
        current_channels = shape[3]

        # Verify dimensions are 512x512
        if input_height != 512 or input_width != 512:
            raise ValueError(f"TTNN tensor dimensions must be 512x512, got {input_height}x{input_width}")

    current_h = input_height
    current_w = input_width
    current_c = current_channels

    sources = []
    if return_sources is not None:
        return_sources = set(return_sources)

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
            layer_memory_config = ttnn.L1_MEMORY_CONFIG if use_l1_for_this_layer else memory_config

            if kernel_size == (1, 1) and stride == (1, 1) and padding == (0, 0):
                if isinstance(weight, torch.Tensor):
                    weight_torch = weight
                else:
                    weight_torch = ttnn.to_torch(weight)

                weight_2d_torch = weight_torch.reshape(out_channels, in_channels).permute(1, 0)

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
                compute_config = None
                if device is not None:
                    compute_config = ttnn.init_device_compute_kernel_config(
                        device.arch(),
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        packer_l1_acc=False,
                        math_approx_mode=False,
                    )

                estimated_memory_bytes = input_height * input_width * in_channels  # bytes per element, bfloat8_b
                if use_l1_for_this_layer and estimated_memory_bytes < 1024 * 1024:
                    if in_channels > 256 or out_channels > 512:
                        use_l1_for_this_layer = False
                        act_block_h = 256
                        layer_memory_config = memory_config
                        conv_config = ttnn.Conv2dConfig(
                            weights_dtype=dtype,
                            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            deallocate_activation=True,
                            enable_act_double_buffer=False,
                            enable_weights_double_buffer=False,
                            reshard_if_not_optimal=False,
                            act_block_h_override=act_block_h,
                            act_block_w_div=1,
                        )
                    elif input_height <= 32:
                        act_block_h = 32
                        enable_double_buffer = True
                        conv_config = ttnn.Conv2dConfig(
                            weights_dtype=dtype,
                            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            deallocate_activation=True,
                            enable_act_double_buffer=enable_double_buffer,
                            enable_weights_double_buffer=False,
                            reshard_if_not_optimal=True,
                            act_block_h_override=act_block_h,
                            act_block_w_div=1,
                        )
                    else:
                        if in_channels <= 128 and out_channels <= 256:
                            act_block_h = 128
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
                            use_l1_for_this_layer = False
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
                    # For DRAM or too large for L1, use default config
                    use_l1_for_this_layer = False
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
                    "bias_tensor": bias,
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
                    "return_weights_and_bias": False,
                    "dtype": dtype,
                    "compute_config": compute_config,
                    "conv_config": conv_config,
                }

                if not use_l1_for_this_layer:
                    layer_memory_config = memory_config
                if layer_memory_config != ttnn.DRAM_MEMORY_CONFIG:
                    conv2d_kwargs["memory_config"] = layer_memory_config

                output_tensor, [output_height, output_width] = ttnn.conv2d(**conv2d_kwargs)

            # reshape output to proper dimensions
            x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])

            # update tracked dimensions for next layer
            current_h = output_height
            current_w = output_width
            current_c = out_channels

        elif layer["type"] == "pool":
            config = layer["config"]

            input_height = current_h
            input_width = current_w
            channels = current_c

            tensor_size_estimate = batch_size * input_height * input_width * channels
            use_l1_for_this_layer = (
                input_height <= 128 and input_width <= 128 and tensor_size_estimate <= 2 * 1024 * 1024
            )
            layer_memory_config = ttnn.L1_MEMORY_CONFIG if use_l1_for_this_layer else memory_config

            if hasattr(x, "memory_config") and x.memory_config() is not None:
                if x.memory_config().is_sharded():
                    x = ttnn.sharded_to_interleaved(x, memory_config=layer_memory_config)

            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            kernel_size = list(config["kernel_size"])
            stride = list(config["stride"])
            padding = list(config["padding"])
            dilation = list(config["dilation"])
            ceil_mode = config["ceil_mode"]

            x = ttnn.max_pool2d(
                x,
                batch_size=batch_size,
                input_h=input_height,
                input_w=input_width,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                memory_config=layer_memory_config,
                dtype=ttnn.bfloat8_b,
                output_layout=ttnn.TILE_LAYOUT,
            )

            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

            kernel_h, kernel_w = kernel_size
            stride_h, stride_w = stride
            padding_h, padding_w = padding
            dilation_h, dilation_w = dilation

            if ceil_mode:
                output_height = int((input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
                if (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) % stride_h != 0:
                    output_height += 1
                output_width = int((input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
                if (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) % stride_w != 0:
                    output_width += 1
            else:
                output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
                output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

            current_h = output_height
            current_w = output_width
            current_c = channels

        elif layer["type"] == "relu":
            x = ttnn.relu(x)

        if return_sources is not None and layer_idx in return_sources:
            sources.append(x)

    if return_sources is not None:
        return x, sources
    return x


# configuration dictionaries matching torch_reference_ssd.py
base = {
    "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
}


def build_vgg_backbone(size=512, input_channels=3, device=None):
    """
    Build VGG backbone for specified input size.
    """
    if size not in [512]:
        raise ValueError(f"Size must be 512, got {size}")

    cfg = base[str(size)]
    layers_config = vgg_backbone(cfg, input_channels=input_channels, device=device)

    return layers_config
