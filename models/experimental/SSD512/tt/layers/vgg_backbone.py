"""
VGG Backbone for SSD Architecture - TTNN Implementation

This module implements the VGG backbone using TTNN operations.
The VGG backbone is derived from torchvision VGG make_layers().
"""

import ttnn
import torch


def vgg_backbone(cfg, input_channels=3, batch_norm=False, device=None):
    """
    Build VGG backbone layers using TTNN operations.

    This function creates the VGG base network layers similar to PyTorch's
    torchvision VGG implementation, but using TTNN operations.

    Args:
        cfg: List of configuration values specifying layer structure
            - Integer values: number of output channels for conv layers
            - "M": MaxPool2d with kernel_size=2, stride=2, ceil_mode=False
            - "C": MaxPool2d with kernel_size=2, stride=2, ceil_mode=True
        input_channels: Number of input channels (default: 3 for RGB)
        batch_norm: Whether to use batch normalization (not currently used)
        device: TTNN device object (required for TTNN operations)

    Returns:
        List of layer dictionaries containing:
            - 'type': Layer type ('conv', 'pool', 'relu')
            - 'params': Layer parameters (weights, bias, config, etc.)
            - 'config': TTNN operation configuration
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
            # This is critical for handling odd-sized inputs in pool3
            layers.append(
                {
                    "type": "pool",
                    "config": {
                        "kernel_size": (2, 2),
                        "stride": (2, 2),
                        "padding": (0, 0),
                        "dilation": (1, 1),
                        "ceil_mode": True,  # CRITICAL for pool3 layer
                    },
                }
            )
        else:
            # Standard 3x3 Conv2d: kernel_size=3, stride=1, padding=1, dilation=1
            # This will be followed by ReLU
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
            # Add ReLU after conv (not storing params, just config)
            layers.append(
                {
                    "type": "relu",
                }
            )
            in_channels = v

    # Add pool5: kernel_size=3, stride=1, padding=1, ceil_mode=False
    # This maintains spatial dimensions rather than downsampling
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
    # Dilated convolution to maintain receptive field without downsampling
    layers.append(
        {
            "type": "conv",
            "config": {
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (6, 6),  # padding = 6
                "dilation": (6, 6),  # dilation = 6 - CRITICAL
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

    This function creates actual TTNN-compatible tensors for weights and biases.
    In a real implementation, these would be loaded from a pretrained model.

    Args:
        layers_config: List of layer dictionaries from vgg_backbone()
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
            # Note: For conv2d, weights will be prepared automatically by ttnn.conv2d
            # but we can also prepare them here if needed
            if device is not None:
                # Convert weights to TTNN tensor format
                # Weights format: (out_channels, in_channels, kernel_h, kernel_w)
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
            # For pool and relu layers, just copy the config
            layers_with_weights.append(layer.copy())

    return layers_with_weights


def apply_vgg_backbone(input_tensor, layers_with_weights, device=None, dtype=ttnn.bfloat16, memory_config=None):
    """
    Apply VGG backbone layers to input tensor using TTNN operations.

    This is the forward pass function that applies all VGG layers sequentially.

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
        # TTNN typically uses NHWC layout for conv operations
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
    # This avoids issues with reading shape from TTNN tensors
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

            # Determine slice config based on tensor size
            # For VGG backbone with large inputs (300x300 or 512x512), we need DRAM slicing
            # This avoids L1_SMALL buffer allocation errors
            # Calculate approximate tensor size to decide slicing strategy
            tensor_size_estimate = batch_size * input_height * input_width * in_channels

            # Use DRAM slicing for large tensors or when input dimensions are large
            # VGG backbone typically processes large feature maps, so default to DRAM slicing
            if tensor_size_estimate > 1024 * 1024 or input_height > 64 or input_width > 64:
                # For large tensors, use DRAM slicing with a reasonable slice count
                # Calculate slice count based on batch size and dimensions
                # Similar to stable_diffusion implementation
                slice_count = max(1, (batch_size * input_height * input_width) // (1024))
                slice_count = min(slice_count, 64)  # Cap at reasonable maximum

                # Convert to ROW_MAJOR_LAYOUT for DRAM slicing (required)
                if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

                slice_config = ttnn.Conv2dSliceConfig(
                    slice_type=ttnn.Conv2dDRAMSliceWidth,
                    num_slices=slice_count,
                )
            else:
                # For smaller tensors, try L1 full slice config
                slice_config = ttnn.Conv2dL1FullSliceConfig

            # Call ttnn.conv2d with slice_config to avoid L1 memory errors
            output_tensor, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=bias,
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

            # Update layer weights to prepared weights (for reuse in subsequent passes)
            layer["weight"] = prepared_weight
            if prepared_bias is not None:
                layer["bias"] = prepared_bias

            # Reshape output to proper dimensions
            x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])

            # Update tracked dimensions for next layer
            current_h = output_height
            current_w = output_width
            current_c = out_channels

        elif layer["type"] == "pool":
            # Apply TTNN max_pool2d
            config = layer["config"]

            # Use tracked dimensions instead of reading from tensor shape
            input_height = current_h
            input_width = current_w
            channels = current_c

            # Convert to ROW_MAJOR_LAYOUT to ensure correct dimension interpretation
            # max_pool2d may need ROW_MAJOR_LAYOUT to correctly read tensor dimensions
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            # Extract parameters from config and convert tuples to lists
            # TTNN requires lists, not tuples, for these parameters
            kernel_size = list(config["kernel_size"])
            stride = list(config["stride"])
            padding = list(config["padding"])
            dilation = list(config["dilation"])
            ceil_mode = config["ceil_mode"]

            # Call ttnn.max_pool2d with explicit dimensions
            # Required parameters: batch_size, input_h, input_w, channels
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
                memory_config=memory_config,
                dtype=dtype,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            # Convert back to TILE_LAYOUT for subsequent operations
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

            # Calculate output dimensions for max_pool2d
            # Formula: output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
            # Or ceil((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1) if ceil_mode
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

            # Update tracked dimensions for next layer
            current_h = output_height
            current_w = output_width
            current_c = channels  # Channels don't change with pooling

        elif layer["type"] == "relu":
            # Apply TTNN relu
            x = ttnn.relu(x, memory_config=memory_config)

    return x


# Configuration dictionaries matching torch_reference_ssd.py
base = {
    "300": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
    "512": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
}


def build_vgg_backbone(size=300, input_channels=3, device=None):
    """
    Build VGG backbone for specified input size.

    Args:
        size: Input size (300 or 512)
        input_channels: Number of input channels (default: 3)
        device: TTNN device object

    Returns:
        Layer configuration list ready for weight loading and forward pass
    """
    if size not in [300, 512]:
        raise ValueError(f"Size must be 300 or 512, got {size}")

    cfg = base[str(size)]
    layers_config = vgg_backbone(cfg, input_channels=input_channels, device=device)

    return layers_config
