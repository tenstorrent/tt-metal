# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Multibox Heads for SSD Architecture - TTNN Implementation

This module implements the multibox location and confidence heads using TTNN operations.
Multibox heads are detection layers applied to multi-scale feature maps from VGG and extras.
"""

import ttnn
import torch


def multibox_heads(
    vgg_source_indices, extra_source_indices, mbox_cfg, num_classes, vgg_channels=None, extra_channels=None
):
    """
    Build multibox location and confidence heads using TTNN operations.

    This function creates detection heads similar to PyTorch's multibox implementation,
    but using TTNN operations.

    Args:
        vgg_source_indices: List of indices into VGG layers to use as sources (e.g., [21, -2])
        extra_source_indices: List of indices into extras layers to use as sources (e.g., [1, 3, 5, ...])
        mbox_cfg: List of number of boxes per feature map location (e.g., [4, 6, 6, 6, 4, 4])
        num_classes: Number of classes for classification
        vgg_channels: List of output channels for VGG source layers (e.g., [512, 1024])
        extra_channels: List of output channels for extras source layers (e.g., [512, 256, 256, 256])

    Returns:
        Tuple of (loc_layers_config, conf_layers_config):
            - loc_layers_config: List of location head configurations
            - conf_layers_config: List of confidence head configurations
    """
    loc_layers_config = []
    conf_layers_config = []

    # Calculate total number of sources (VGG + extras)
    total_sources = len(vgg_source_indices) + len(extra_source_indices)

    # Validate that mbox_cfg has enough elements for all sources
    if len(mbox_cfg) < total_sources:
        print(
            f"Warning: mbox_cfg has {len(mbox_cfg)} elements, but we have {total_sources} sources (VGG: {len(vgg_source_indices)}, Extras: {len(extra_source_indices)})."
        )
        print(f"This might indicate a mismatch between source indices and mbox configuration.")

    layer_idx = 0

    # Process VGG source layers
    for vgg_idx in vgg_source_indices:
        if vgg_channels is not None:
            # Index into vgg_channels array (should match vgg_idx position)
            vgg_channel_idx = vgg_source_indices.index(vgg_idx)
            in_channels = vgg_channels[vgg_channel_idx]
        else:
            # Default: Conv4_3 (512) and Conv7 (1024)
            in_channels = 512 if len(loc_layers_config) == 0 else 1024

        # Bounds check for mbox_cfg
        if layer_idx >= len(mbox_cfg):
            print(f"Warning: layer_idx {layer_idx} exceeds mbox_cfg length {len(mbox_cfg)}. Using last element.")
            num_boxes = mbox_cfg[-1]  # Use last element as fallback
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

    # Process extras source layers (every other layer starting from index 1)
    # Use enumerate to track position in extra_source_indices
    for extra_idx_pos, extra_idx in enumerate(extra_source_indices):
        if extra_channels is not None:
            # Index directly into extra_channels using position in extra_source_indices
            # Add bounds checking to avoid IndexError
            if extra_idx_pos < len(extra_channels):
                in_channels = extra_channels[extra_idx_pos]
            else:
                # Fallback: use default progression if not enough channels provided
                # This can happen if extra_source_indices has more elements than extra_channels
                if extra_idx_pos == 0:
                    in_channels = 512
                else:
                    in_channels = 256
                print(
                    f"Warning: extra_channels has {len(extra_channels)} elements, but need index {extra_idx_pos}. Using default channels."
                )
        else:
            # Default progression: 512, 256, 256, 256, ...
            if extra_idx_pos == 0:
                in_channels = 512
            else:
                in_channels = 256

        # Bounds check for mbox_cfg
        if layer_idx >= len(mbox_cfg):
            print(f"Warning: layer_idx {layer_idx} exceeds mbox_cfg length {len(mbox_cfg)}. Using last element.")
            num_boxes = mbox_cfg[-1]  # Use last element as fallback
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
    """
    Create multibox layers with initialized weights and biases.

    This function creates actual TTNN-compatible tensors for weights and biases.
    In a real implementation, these would be loaded from a pretrained model.

    Args:
        loc_layers_config: List of location head layer configurations
        conf_layers_config: List of confidence head layer configurations
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)

    Returns:
        Tuple of (loc_layers_with_weights, conf_layers_with_weights):
            - loc_layers_with_weights: List of location head layers with weights
            - conf_layers_with_weights: List of confidence head layers with weights
    """
    loc_layers_with_weights = []
    conf_layers_with_weights = []

    # Process location layers
    for layer in loc_layers_config:
        if layer["type"] == "conv":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]

            # Initialize weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
            kernel_size = layer["config"]["kernel_size"]
            weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])

            # Initialize with Kaiming normal (He initialization)
            weight = torch.empty(weight_shape)
            torch.nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

            # Initialize bias: (out_channels,)
            bias = torch.zeros(out_channels)

            # Convert to TTNN format
            if device is not None:
                weight_ttnn = ttnn.from_torch(
                    weight,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                bias_reshaped = bias.reshape((1, 1, 1, -1))
                bias_ttnn = ttnn.from_torch(
                    bias_reshaped,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                weight_ttnn = weight
                bias_ttnn = bias

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
            weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])

            weight = torch.empty(weight_shape)
            torch.nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

            bias = torch.zeros(out_channels)

            if device is not None:
                weight_ttnn = ttnn.from_torch(
                    weight,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                bias_reshaped = bias.reshape((1, 1, 1, -1))
                bias_ttnn = ttnn.from_torch(
                    bias_reshaped,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                weight_ttnn = weight
                bias_ttnn = bias

            layer_with_weights = layer.copy()
            layer_with_weights["weight"] = weight_ttnn
            layer_with_weights["bias"] = bias_ttnn
            conf_layers_with_weights.append(layer_with_weights)

    return loc_layers_with_weights, conf_layers_with_weights


def apply_multibox_heads(
    sources, loc_layers_with_weights, conf_layers_with_weights, device=None, dtype=ttnn.bfloat16, memory_config=None
):
    """
    Apply multibox heads to source feature maps using TTNN operations.

    This is the forward pass function that applies location and confidence heads
    to each source feature map.

    Args:
        sources: List of source feature maps (ttnn.Tensor or torch.Tensor)
                 Each tensor should be in NCHW format (torch) or NHWC format (ttnn)
        loc_layers_with_weights: List of location head layers with weights
        conf_layers_with_weights: List of confidence head layers with weights
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)
        memory_config: TTNN memory config (default: None uses DRAM_MEMORY_CONFIG)

    Returns:
        Tuple of (loc_preds, conf_preds):
            - loc_preds: List of location predictions (one per source)
            - conf_preds: List of confidence predictions (one per source)
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    loc_preds = []
    conf_preds = []

    # Apply heads to each source feature map
    for source_idx, source in enumerate(sources):
        # Get corresponding head layers
        loc_layer = loc_layers_with_weights[source_idx]
        conf_layer = conf_layers_with_weights[source_idx]

        # Apply location head
        loc_pred = apply_multibox_head(
            source,
            loc_layer,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        loc_preds.append(loc_pred)

        # Apply confidence head
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
    """
    Apply a single multibox head (location or confidence) to input tensor.

    Args:
        input_tensor: Input tensor - either torch.Tensor (N, C, H, W) or ttnn.Tensor
        layer_with_weights: Layer dictionary with weight and bias
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)
        memory_config: TTNN memory config (default: None uses DRAM_MEMORY_CONFIG)

    Returns:
        Output tensor as ttnn.Tensor (in NHWC format)
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Convert input to TTNN format if it's a torch tensor
    if isinstance(input_tensor, torch.Tensor):
        # Input format: (N, C, H, W) -> convert to TTNN format (N, H, W, C)
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

    # Get dimensions
    if isinstance(input_tensor, torch.Tensor):
        batch_size = 1
        input_height = input_tensor.shape[2]  # H from NCHW
        input_width = input_tensor.shape[3]  # W from NCHW
        in_channels = input_tensor.shape[1]  # C from NCHW
    else:
        shape = x.shape
        batch_size = 1
        input_height = shape[1]
        input_width = shape[2]
        in_channels = shape[3]

    # Get layer parameters
    weight = layer_with_weights["weight"]
    bias = layer_with_weights.get("bias", None)
    config = layer_with_weights["config"]
    out_channels = layer_with_weights["out_channels"]

    kernel_size = config["kernel_size"]
    stride = config["stride"]
    padding = config["padding"]
    dilation = config["dilation"]
    groups = config["groups"]

    # Multibox heads are 3x3 convs with padding=1, typically smaller tensors
    # Can use L1 full slice config for most cases
    # Only use DRAM slicing for very large feature maps
    tensor_size_estimate = batch_size * input_height * input_width * in_channels

    # Check if we need DRAM slicing (unlikely for multibox heads, but handle it)
    # Skip DRAM slicing for 1x1 convs (but multibox doesn't have 1x1, all are 3x3)
    is_1x1_conv = kernel_size == (1, 1) or (kernel_size[0] == 1 and kernel_size[1] == 1)
    is_very_small = input_height <= 2 or input_width <= 2  # Very small feature maps like 1x1 or 2x2
    use_dram_slicing = (
        (tensor_size_estimate > 1024 * 1024 or input_height > 64 or input_width > 64) or is_very_small
    ) and not is_1x1_conv

    # TTNN limitation: bias is not supported with batched inputs (DRAM slicing)
    if use_dram_slicing:
        conv_bias = None  # Python None, not a tensor
        slice_count = max(1, (batch_size * input_height * input_width) // (1024))
        slice_count = min(slice_count, 64)

        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
            weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)

        slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth,
            num_slices=slice_count,
        )
    else:
        conv_bias = bias
        slice_config = ttnn.Conv2dL1FullSliceConfig

    # Call ttnn.conv2d
    output_tensor, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=conv_bias,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        device=device,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=dtype,
        memory_config=memory_config,
        slice_config=slice_config,
    )

    # Convert back to TILE_LAYOUT if we switched to ROW_MAJOR for slicing
    if slice_config != ttnn.Conv2dL1FullSliceConfig and output_tensor.layout != ttnn.TILE_LAYOUT:
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

    # Add bias separately if needed (when using DRAM slicing)
    if bias is not None and use_dram_slicing:
        output_tensor = output_tensor.reshape([batch_size, output_height, output_width, out_channels])

        if bias.layout != output_tensor.layout:
            bias = ttnn.to_layout(bias, output_tensor.layout)

        # Convert bias to torch, expand, convert back
        bias_torch = ttnn.to_torch(bias)
        bias_expanded = bias_torch.expand(batch_size, output_height, output_width, out_channels)

        bias_full = ttnn.from_torch(
            bias_expanded,
            device=device,
            dtype=dtype,
            layout=output_tensor.layout,
            memory_config=memory_config,
        )

        output_tensor = ttnn.add(output_tensor, bias_full, memory_config=memory_config)

    # Update layer weights
    layer_with_weights["weight"] = prepared_weight
    if prepared_bias is not None and not use_dram_slicing:
        layer_with_weights["bias"] = prepared_bias

    # Reshape output to proper dimensions
    if bias is None or not use_dram_slicing:
        x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])
    else:
        x = output_tensor

    return x


# Configuration dictionaries matching torch_reference_ssd.py
mbox = {
    "300": [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    "512": [4, 6, 6, 6, 4, 4, 4],
}


def build_multibox_heads(
    size=300,
    num_classes=21,
    vgg_channels=None,
    extra_channels=None,
    vgg_source_indices=None,
    extra_source_indices=None,
    device=None,
):
    """
    Build multibox heads for specified input size.

    Args:
        size: Input size (300 or 512)
        num_classes: Number of classes
        vgg_channels: List of VGG source layer channels [Conv4_3, Conv7]
        extra_channels: List of extras source layer channels
        vgg_source_indices: VGG source indices (default: [21, -2])
        extra_source_indices: Extras source indices (default: based on size)
        device: TTNN device object (not used, kept for compatibility)

    Returns:
        Tuple of (loc_layers_config, conf_layers_config)
    """
    if size not in [300, 512]:
        raise ValueError(f"Size must be 300 or 512, got {size}")

    cfg = mbox[str(size)]

    # VGG source indices: [21, -2] for Conv4_3 and Conv7
    if vgg_source_indices is None:
        vgg_source_indices = [21, -2]

    # Extras source indices: every other layer starting from index 1
    # For SSD300: indices 1, 3, 5, 7 (4 layers)
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
            extra_channels = [512, 256, 256, 256]  # From extras[1::2]
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
