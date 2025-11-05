# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Extras Backbone for SSD Architecture - TTNN Implementation

This module implements the extras layers using TTNN operations.
Extras layers are additional feature layers added after VGG backbone.
"""

import ttnn
import torch


def extras_backbone(cfg, input_channels=1024, batch_norm=False, device=None):
    """
    Build extras layers using TTNN operations.

    This function creates extra feature layers similar to PyTorch's
    add_extras implementation, but using TTNN operations.

    Args:
        cfg: List of configuration values specifying layer structure
            - Integer values: number of output channels for conv layers
            - "S": Marker for stride=2 downsampling layer
        input_channels: Number of input channels (default: 1024 from VGG)
        batch_norm: Whether to use batch normalization (not currently used)
        device: TTNN device object (required for TTNN operations)

    Returns:
        List of layer dictionaries containing:
            - 'type': Layer type ('conv', 'relu')
            - 'config': TTNN operation configuration
            - 'in_channels': Input channels
            - 'out_channels': Output channels
    """
    layers = []
    in_channels = input_channels
    flag = False  # Toggles between kernel_size 1 and 3

    for k, v in enumerate(cfg):
        if in_channels != "S":  # Skip if previous value was "S"
            if v == "S":
                # Stride=2 downsampling layer
                # Output channels come from next element in cfg
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
                # Add ReLU after conv (matching PyTorch pattern)
                layers.append(
                    {
                        "type": "relu",
                    }
                )
                flag = not flag  # Toggle for next layer
            else:
                # Regular layer (stride=1, no padding)
                out_channels = v
                kernel_size = (1, 3)[flag]  # Alternating: 1 or 3

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
                # Add ReLU after conv (matching PyTorch pattern)
                layers.append(
                    {
                        "type": "relu",
                    }
                )
                flag = not flag  # Toggle for next layer

        # Update in_channels to current v (this happens regardless of the if condition)
        # Note: If v == "S", in_channels becomes "S", which causes next iteration to skip
        in_channels = v

    # Special case for SSD512: add final 4x4 conv layer
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
        # Add ReLU after conv
        layers.append(
            {
                "type": "relu",
            }
        )

    return layers


def create_extras_layers_with_weights(layers_config, device=None, dtype=ttnn.bfloat16):
    """
    Create extras layers with initialized weights and biases.

    This function creates actual TTNN-compatible tensors for weights and biases.
    In a real implementation, these would be loaded from a pretrained model.

    Args:
        layers_config: List of layer dictionaries from extras_backbone()
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)

    Returns:
        List of layer dictionaries with 'weight' and 'bias' tensors added
    """
    layers_with_weights = []

    for layer in layers_config:
        if layer["type"] == "conv":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]

            # Initialize weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
            kernel_size = layer["config"]["kernel_size"]
            weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])

            # Initialize with Kaiming normal (He initialization)
            # This matches PyTorch's default conv2d initialization
            weight = torch.empty(weight_shape)
            torch.nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

            # Initialize bias: (out_channels,)
            bias = torch.zeros(out_channels)

            # Convert to TTNN format
            if device is not None:
                # Convert weights to TTNN tensor format
                weight_ttnn = ttnn.from_torch(
                    weight,
                    device=device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                # Convert bias to TTNN format: reshape to (1, 1, 1, out_channels)
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
            layers_with_weights.append(layer_with_weights)
        else:
            # For relu layers, just copy the config
            layers_with_weights.append(layer.copy())

    return layers_with_weights


def apply_extras_backbone(input_tensor, layers_with_weights, device=None, dtype=ttnn.bfloat16, memory_config=None):
    """
    Apply extras layers to input tensor using TTNN operations.

    This is the forward pass function that applies all extras layers sequentially.

    Args:
        input_tensor: Input tensor - either torch.Tensor (N, C, H, W) or ttnn.Tensor
        layers_with_weights: List of layer dictionaries with weights/biases
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)
        memory_config: TTNN memory config (default: None uses DRAM_MEMORY_CONFIG)

    Returns:
        Output tensor as ttnn.Tensor
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

    # Track dimensions explicitly throughout the forward pass
    if isinstance(input_tensor, torch.Tensor):
        # Get initial dimensions from input tensor
        batch_size = 1
        input_height = input_tensor.shape[2]  # H from NCHW
        input_width = input_tensor.shape[3]  # W from NCHW
        current_channels = input_tensor.shape[1]  # C from NCHW
    else:
        # If already TTNN tensor, extract from shape (NHWC format)
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
            # Apply TTNN conv2d
            weight = layer["weight"]
            bias = layer.get("bias", None)
            config = layer["config"]

            # if isinstance(weight, torch.Tensor):
            #     weight = ttnn.from_torch(
            #         weight, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            #     )
            # if bias is not None and isinstance(bias, torch.Tensor):
            #     bias_reshaped = bias.reshape((1, 1, 1, -1))
            #     bias = ttnn.from_torch(
            #         bias_reshaped, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            #     )

            # Use tracked dimensions instead of reading from tensor shape
            input_height = current_h
            input_width = current_w
            in_channels = current_c
            out_channels = layer["out_channels"]

            # Extract parameters from config
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            padding = config["padding"]
            dilation = config["dilation"]
            groups = config["groups"]

            # Determine slice config based on tensor size (same logic as VGG)
            tensor_size_estimate = batch_size * input_height * input_width * in_channels

            # TTNN limitation:
            # 1. Bias is not supported with batched inputs (DRAM slicing)
            # 2. DRAM slicing might have issues with 1x1 convolutions (matmul shape mismatch)
            # For now, only use DRAM slicing for larger kernels (3x3, 4x4) or avoid it for 1x1
            is_1x1_conv = kernel_size == (1, 1) or (kernel_size[0] == 1 and kernel_size[1] == 1)

            # Determine if we need DRAM slicing (batched operation)
            # Skip DRAM slicing for 1x1 convs to avoid matmul shape issues
            use_dram_slicing = (
                tensor_size_estimate > 1024 * 1024 or input_height > 64 or input_width > 64
            ) and not is_1x1_conv

            # TTNN limitation: bias is not supported with batched inputs (DRAM slicing)
            # We need to handle bias separately when using DRAM slicing
            # IMPORTANT: conv_bias must be Python None (not a TTNN tensor) when using DRAM slicing
            if use_dram_slicing:
                conv_bias = None  # Python None, not a tensor
                slice_count = max(1, (batch_size * input_height * input_width) // (1024))
                slice_count = min(slice_count, 64)

                # Convert to ROW_MAJOR_LAYOUT for DRAM slicing (required)
                if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

                # Ensure weight tensor is also in ROW_MAJOR_LAYOUT for DRAM slicing
                if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
                    weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)

                slice_config = ttnn.Conv2dSliceConfig(
                    slice_type=ttnn.Conv2dDRAMSliceWidth,
                    num_slices=slice_count,
                )
            else:
                # For 1x1 convs or smaller tensors, use L1 full slice config
                # This allows bias to be used normally
                conv_bias = bias  # Use bias normally for L1 slicing
                slice_config = ttnn.Conv2dL1FullSliceConfig

            # Create compute_config with HiFi4 and fp32 accumulator for higher precision
            compute_config = None
            if device is not None:
                compute_config = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,  # High fidelity for better precision
                    fp32_dest_acc_en=True,  # Use fp32 accumulator for higher precision
                    packer_l1_acc=False,
                    math_approx_mode=False,  # Disable math approximation for maximum precision
                )

            # Call ttnn.conv2d with slice_config
            # Note: When using DRAM slicing, bias_tensor must be Python None
            output_tensor, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=conv_bias,  # Python None when using DRAM slicing
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
                compute_config=compute_config,  # HiFi4 with fp32 accumulator for higher precision
            )

            # Convert back to TILE_LAYOUT if we switched to ROW_MAJOR for slicing
            if slice_config != ttnn.Conv2dL1FullSliceConfig and output_tensor.layout != ttnn.TILE_LAYOUT:
                output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

            # If we had bias but couldn't use it in conv2d (due to DRAM slicing), add it separately
            if bias is not None and use_dram_slicing:
                # Reshape output first to ensure proper shape
                output_tensor = output_tensor.reshape([batch_size, output_height, output_width, out_channels])

                # Convert bias to torch, expand to full output shape, then convert back
                # This ensures proper shape matching for element-wise addition
                bias_torch = ttnn.to_torch(bias)
                # bias_torch is (1, 1, 1, out_channels), expand to (batch_size, output_height, output_width, out_channels)
                bias_expanded = bias_torch.expand(batch_size, output_height, output_width, out_channels)

                # Convert back to TTNN with matching layout
                bias_full = ttnn.from_torch(
                    bias_expanded,
                    device=device,
                    dtype=dtype,
                    layout=output_tensor.layout,
                    memory_config=memory_config,
                )

                # Now add bias with matching shapes
                output_tensor = ttnn.add(output_tensor, bias_full, memory_config=memory_config)

            # Update layer weights to prepared weights (for reuse in subsequent passes)
            layer["weight"] = prepared_weight
            if prepared_bias is not None and not use_dram_slicing:
                layer["bias"] = prepared_bias

            # Reshape output to proper dimensions (if not already reshaped)
            if bias is None or not use_dram_slicing:
                x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])
            else:
                x = output_tensor

            # Update tracked dimensions for next layer
            current_h = output_height
            current_w = output_width
            current_c = out_channels

        elif layer["type"] == "relu":
            # Apply TTNN relu
            x = ttnn.relu(x, memory_config=memory_config)

    return x


# Configuration dictionaries matching torch_reference_ssd.py
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
}


def build_extras_backbone(size=512, input_channels=1024, device=None):
    """
    Build extras backbone for specified input size.

    Args:
        size: Input size (300 or 512)
        input_channels: Number of input channels (default: 1024 from VGG)
        device: TTNN device object

    Returns:
        Layer configuration list ready for weight loading and forward pass
    """
    if size not in [300, 512]:
        raise ValueError(f"Size must be 300 or 512, got {size}")

    cfg = extras[str(size)]
    layers_config = extras_backbone(cfg, input_channels=input_channels, device=device)

    return layers_config
