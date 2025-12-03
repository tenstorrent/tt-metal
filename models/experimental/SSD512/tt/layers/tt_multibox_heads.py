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


def multibox_heads(
    vgg_source_indices, extra_source_indices, mbox_cfg, num_classes, vgg_channels=None, extra_channels=None
):
    """
    Build multibox location and confidence heads using TTNN operations.
    """
    loc_layers_config = []
    conf_layers_config = []

    total_sources = len(vgg_source_indices) + len(extra_source_indices)

    layer_idx = 0

    # process VGG source layers
    for vgg_idx in vgg_source_indices:
        if vgg_channels is not None:
            vgg_channel_idx = vgg_source_indices.index(vgg_idx)
            in_channels = vgg_channels[vgg_channel_idx]
        else:
            in_channels = 512 if len(loc_layers_config) == 0 else 1024

        # bounds check for mbox_cfg
        if layer_idx >= len(mbox_cfg):
            num_boxes = mbox_cfg[-1]
        else:
            num_boxes = mbox_cfg[layer_idx]

        # Location head: predicts 4 coordinates per box
        loc_out_channels = num_boxes * 4
        loc_layers_config.append(
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
                "out_channels": loc_out_channels,
                "source": "vgg",
                "source_idx": vgg_idx,
            }
        )

        # Confidence head: predicts num_classes per box
        conf_out_channels = num_boxes * num_classes
        conf_layers_config.append(
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
                "out_channels": conf_out_channels,
                "source": "vgg",
                "source_idx": vgg_idx,
            }
        )

        layer_idx += 1

    for extra_idx_pos, extra_idx in enumerate(extra_source_indices):
        if extra_channels is not None:
            if extra_idx_pos < len(extra_channels):
                in_channels = extra_channels[extra_idx_pos]
            else:
                if extra_idx_pos == 0:
                    in_channels = 512
                else:
                    in_channels = 256
        else:
            if extra_idx_pos == 0:
                in_channels = 512
            else:
                in_channels = 256

        # Bounds check for mbox_cfg
        if layer_idx >= len(mbox_cfg):
            num_boxes = mbox_cfg[-1]
        else:
            num_boxes = mbox_cfg[layer_idx]

        # Location head
        loc_out_channels = num_boxes * 4
        loc_layers_config.append(
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
                "out_channels": loc_out_channels,
                "source": "extra",
                "source_idx": extra_idx,
            }
        )

        # Confidence head
        conf_out_channels = num_boxes * num_classes
        conf_layers_config.append(
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
                "out_channels": conf_out_channels,
                "source": "extra",
                "source_idx": extra_idx,
            }
        )

        layer_idx += 1

    return loc_layers_config, conf_layers_config


def create_multibox_layers_with_weights(loc_layers_config, conf_layers_config, device=None, dtype=ttnn.bfloat16):
    """Create multibox layers with initialized weights and biases."""
    from models.experimental.SSD512.common import create_conv2d_weights_and_bias

    loc_layers_with_weights = []
    conf_layers_with_weights = []

    # process location layers
    for layer in loc_layers_config:
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
            loc_layers_with_weights.append(layer_with_weights)

    # Process confidence layers (same logic)
    for layer in conf_layers_config:
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
            conf_layers_with_weights.append(layer_with_weights)

    return loc_layers_with_weights, conf_layers_with_weights


def apply_multibox_heads(
    sources, loc_layers_with_weights, conf_layers_with_weights, device=None, dtype=ttnn.bfloat8_b, memory_config=None
):
    """
    Apply multibox heads to source feature maps using TTNN operations.
    """
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    loc_preds = []
    conf_preds = []

    for source_idx, source in enumerate(sources):
        loc_layer = loc_layers_with_weights[source_idx]
        conf_layer = conf_layers_with_weights[source_idx]

        loc_pred = apply_multibox_head(
            source,
            loc_layer,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        loc_preds.append(loc_pred)

        conf_pred = apply_multibox_head(
            source,
            conf_layer,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        conf_preds.append(conf_pred)

    return loc_preds, conf_preds


def apply_multibox_head(input_tensor, layer_with_weights, device=None, dtype=ttnn.bfloat16, memory_config=None):
    """Apply a single multibox head (location or confidence) to input tensor."""
    if isinstance(input_tensor, torch.Tensor):
        batch_size = 1
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]
        in_channels = input_tensor.shape[1]
    else:
        shape = input_tensor.shape
        batch_size = 1
        input_height = shape[1]
        input_width = shape[2]
        in_channels = shape[3]

    # Use L1 for smaller feature maps (<=128x128), DRAM for larger ones
    tensor_size_estimate = batch_size * input_height * input_width * in_channels
    use_l1_for_this_layer = input_height <= 128 and input_width <= 128 and tensor_size_estimate <= 2 * 1024 * 1024

    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    if isinstance(input_tensor, torch.Tensor):
        x_torch = input_tensor.permute(0, 2, 3, 1)
        x = ttnn.from_torch(
            x_torch,
            device=None,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        x = input_tensor
        x_torch = ttnn.to_torch(x)
        x = ttnn.from_torch(
            x_torch,
            device=None,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )

    weight = layer_with_weights["weight"]
    bias = layer_with_weights.get("bias", None)
    config = layer_with_weights["config"]
    out_channels = layer_with_weights["out_channels"]

    expected_in_channels = layer_with_weights.get("in_channels", config.get("in_channels", None))

    kernel_size = config["kernel_size"]
    stride = config["stride"]
    padding = config["padding"]
    dilation = config["dilation"]
    groups = config["groups"]

    if isinstance(weight, torch.Tensor):
        actual_weight_in_channels = weight.shape[1]
    else:
        weight_torch = ttnn.to_torch(weight)
        actual_weight_in_channels = weight_torch.shape[1]

    if actual_weight_in_channels != in_channels:
        raise ValueError(
            f"Input tensor channels ({in_channels}) don't match weight's expected input channels ({actual_weight_in_channels})!"
        )

    is_1x1_conv = kernel_size == (1, 1) or (kernel_size[0] == 1 and kernel_size[1] == 1)
    is_very_small = input_height <= 2 or input_width <= 2

    is_l1_memory = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
    force_l1_slice = is_very_small or is_1x1_conv

    # Use DRAM slicing only if using DRAM memory config and tensor is large
    use_dram_slicing = (
        not force_l1_slice
        and not is_l1_memory
        and ((tensor_size_estimate > 1024 * 1024 or input_height > 64 or input_width > 64))
    )

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
        weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight_torch, bias_torch)

    if use_dram_slicing and x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if device is not None:
        x = ttnn.to_device(x, device, memory_config=memory_config)

    estimated_memory_bytes = input_height * input_width * in_channels * 2

    if use_dram_slicing:
        slice_count = max(1, (batch_size * input_height * input_width) // (1024))
        slice_count = min(slice_count, 64)
        slice_strategy = WidthSliceStrategyConfiguration(num_slices=slice_count)
        sharding_strategy = AutoShardedStrategyConfiguration()
        enable_act_double_buffer = False
        deallocate_activation = False
    elif use_l1_for_this_layer and not use_dram_slicing and estimated_memory_bytes < 1024 * 1024:
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

    if output_tensor.is_sharded():
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config)

    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size

    output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])

    return x


# configuration dictionaries matching torch_reference_ssd.py
mbox = {
    "512": [4, 6, 6, 6, 4, 4, 4],
}


def build_multibox_heads(
    size=512,
    num_classes=21,
    vgg_channels=None,
    extra_channels=None,
    vgg_source_indices=None,
    extra_source_indices=None,
    device=None,
):
    """
    Build multibox heads for specified input size.
    """
    if size not in [300, 512]:
        raise ValueError(f"Size must be 512, got {size}")

    cfg = mbox[str(size)]

    if vgg_source_indices is None:
        vgg_source_indices = [21, -2]

    # Extras source indices: every other layer starting from index 1
    # For SSD512: indices 1, 3, 5, 7, 9, 11 (6 layers)
    if extra_source_indices is None:
        if size == 300:
            extra_source_indices = [1, 3, 5, 7]
        else:  # size == 512
            extra_source_indices = [1, 3, 5, 7, 9, 11]

    # Default channel counts if not provided
    if vgg_channels is None:
        vgg_channels = [512, 1024]  # Conv4_3, Conv7

    if extra_channels is None:
        if size == 300:
            extra_channels = [512, 256, 256, 256]
        else:  # size == 512
            extra_channels = [512, 256, 256, 256, 256, 256]

    loc_layers_config, conf_layers_config = multibox_heads(
        vgg_source_indices=vgg_source_indices,
        extra_source_indices=extra_source_indices,
        mbox_cfg=cfg,
        num_classes=num_classes,
        vgg_channels=vgg_channels,
        extra_channels=extra_channels,
    )

    return loc_layers_config, conf_layers_config
