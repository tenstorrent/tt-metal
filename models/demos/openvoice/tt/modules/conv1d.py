# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Conv1d wrapper that maps to TTNN conv2d operations.

TTNN provides conv2d but not conv1d, so we reshape tensors to use conv2d
with height=1, effectively implementing conv1d.
"""

from typing import Optional, Any, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def ttnn_conv1d(
    x: Any,
    weight: Any,
    bias: Optional[Any] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    device: Optional[Any] = None,
    conv_config: Optional[Any] = None,
    compute_config: Optional[Any] = None,
    activation: Optional[str] = None,
) -> Any:
    """
    Perform 1D convolution using TTNN's conv2d operation.

    Reshapes input from [B, C, L] to [B, 1, L, C] (NHWC for TTNN),
    runs conv2d with kernel (1, K), then reshapes back.

    Args:
        x: Input tensor [B, C_in, L] or [B, 1, L, C_in] (pre-reshaped)
        weight: Weight tensor [C_out, C_in, 1, K] (pre-reshaped from Conv1d)
        bias: Optional bias tensor [C_out]
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation
        groups: Number of groups for grouped convolution
        device: TTNN device
        conv_config: TTNN Conv2dConfig
        compute_config: TTNN compute kernel config
        activation: Fused activation ("relu", "leaky_relu", etc.)

    Returns:
        Output tensor [B, C_out, L_out]
    """
    # Check if inputs are PyTorch tensors - use PyTorch path
    # Check BOTH x and weight to avoid mixing torch/ttnn operations
    is_torch_x = isinstance(x, torch.Tensor)
    is_torch_weight = isinstance(weight, torch.Tensor)
    is_torch = is_torch_x or is_torch_weight

    # Force PyTorch for large stride operations (L1 memory constraints)
    # HiFi-GAN uses kernel=16, stride=8 which exceeds L1 buffer limits
    # Use stride >= 2 as the trigger - large upsampling operations cause L1 issues
    large_stride_op = stride >= 2

    if not TTNN_AVAILABLE or is_torch or large_stride_op:
        # Convert TTNN tensors to PyTorch if needed
        if not is_torch_x and TTNN_AVAILABLE:
            x = ttnn.to_torch(x)
        if not is_torch_weight and TTNN_AVAILABLE:
            weight = ttnn.to_torch(weight)
        if bias is not None and not isinstance(bias, torch.Tensor) and TTNN_AVAILABLE:
            bias = ttnn.to_torch(bias)

        # Ensure float32 for PyTorch operations
        x = x.float() if x.dtype != torch.float32 else x
        weight = weight.float() if weight.dtype != torch.float32 else weight
        if bias is not None:
            bias = bias.float() if bias.dtype != torch.float32 else bias

        # Fallback to PyTorch for development or when given torch tensors
        if x.dim() == 4:
            x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, L, C] -> [B, C, L]
        weight_3d = weight.squeeze(2) if weight.dim() == 4 else weight  # [C_out, C_in, 1, K] -> [C_out, C_in, K]
        out = F.conv1d(x, weight_3d, bias, stride, padding, dilation, groups)
        if activation == "relu":
            out = F.relu(out)
        elif activation == "leaky_relu":
            out = F.leaky_relu(out, 0.1)
        # Convert back to TTNN if input was originally TTNN
        if (not is_torch_x or not is_torch_weight) and TTNN_AVAILABLE and device is not None:
            out = ttnn.from_torch(out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return out

    # Get dimensions (TTNN tensors have shape.rank)
    is_3d_input = x.shape.rank == 3

    if is_3d_input:
        # Reshape [B, C, L] -> [B, 1, L, C] for TTNN (NHWC format)
        batch, channels, length = x.shape
        x = ttnn.permute(x, (0, 2, 1))  # [B, C, L] -> [B, L, C]
        x = ttnn.reshape(x, (batch, 1, length, channels))  # [B, L, C] -> [B, 1, L, C]

    # Get weight dimensions - handle both 3D [C_out, C_in, K] and 4D [C_out, C_in, 1, K]
    weight_rank = weight.shape.rank if hasattr(weight.shape, 'rank') else len(weight.shape)
    if weight_rank == 3:
        out_channels, in_channels, kernel_size = weight.shape
        # Reshape to 4D for conv2d: [C_out, C_in, K] -> [C_out, C_in, 1, K]
        weight = ttnn.reshape(weight, (out_channels, in_channels, 1, kernel_size))
    else:
        out_channels, in_channels, _, kernel_size = weight.shape
    # x.shape is [B, H, W, C] after reshape where H=1, W=L (sequence length), C=channels
    batch_size, input_height, input_width, input_channels = x.shape

    # Note: In TTNN's NHWC convention with our reshape:
    # - input_height = 1 (our added dimension)
    # - input_width = L (sequence length)
    # - input_channels = C_in (last dimension in NHWC)

    # Default conv config if not provided
    # Use WIDTH_SHARDED for convolutions - channels are naturally parallel
    if conv_config is None:
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            config_tensors_in_dram=True,  # Required - L1_SMALL may not be available
            deallocate_activation=False,
            reallocate_halo_output=False,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,  # Sharding optimization
        )

    # Add activation to config if specified
    if activation == "relu":
        conv_config.activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    elif activation == "leaky_relu":
        conv_config.activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1)

    # Default compute config - use HiFi4 for optimal accuracy
    if compute_config is None and device is not None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    # Prepare conv kwargs (shared between prepare_conv_weights and conv2d)
    conv_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (1, kernel_size),
        "stride": (1, stride),
        "padding": (0, padding),
        "dilation": (1, dilation),
        "groups": groups,
    }

    # Track bias for manual addition after conv2d
    bias_to_add = None

    # Prepare conv weights if not already on device
    # This is REQUIRED - conv weights must go through prepare_conv_weights() before conv2d
    # See: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md
    if device is not None and not ttnn.is_tensor_storage_on_device(weight):
        weight = ttnn.prepare_conv_weights(
            weight_tensor=weight,
            weights_format="OIHW",
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=x.get_layout(),
            input_dtype=x.dtype,
            has_bias=False,  # We'll add bias manually
            device=device,
            conv_config=conv_config,
            **conv_kwargs,
        )
        # prepare_conv_weights may already put weight on device, check before to_device
        if not ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.to_device(weight, device)

        # Store bias for manual addition after conv2d
        # (Passing bias to conv2d has issues with shape validation)
        if bias is not None and not ttnn.is_tensor_storage_on_device(bias):
            # Reshape 1D bias to broadcastable shape [1, C_out, 1] for [B, C, L] output
            if bias.shape.rank == 1:
                bias_to_add = ttnn.reshape(bias, (1, out_channels, 1))
            else:
                bias_to_add = bias

    # Run conv2d without bias
    out = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=None,  # Handle bias manually
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        **conv_kwargs,
    )

    if is_3d_input:
        # Reshape back [B, 1, L_out, C_out] -> [B, C_out, L_out]
        # TTNN conv2d output is [B*H*W, 1, 1, C] in some cases or [B, H, W, C] depending on settings
        # We need to check the actual rank and shape
        if out.shape.rank == 4:
            batch = out.shape[0]
            out_length = out.shape[2]
            out_channels = out.shape[3]
            out = ttnn.reshape(out, (batch, out_length, out_channels))  # [B, L, C]
            out = ttnn.permute(out, (0, 2, 1))  # [B, C, L]
        elif out.shape.rank == 2:
            # Flattened output [B*L, C] - need to reshape
            batch = x.shape[0] if is_3d_input else x.shape[0]
            total = out.shape[0]
            out_channels = out.shape[1]
            out_length = total // batch
            out = ttnn.reshape(out, (batch, out_length, out_channels))  # [B, L, C]
            out = ttnn.permute(out, (0, 2, 1))  # [B, C, L]

    # Add bias after reshape (if we stored it earlier)
    if bias_to_add is not None and device is not None:
        # Convert bias to TILE layout for broadcast
        bias_to_add_tile = ttnn.to_layout(bias_to_add, ttnn.TILE_LAYOUT)
        bias_to_add_tile = ttnn.to_device(bias_to_add_tile, device)
        out = ttnn.add(out, bias_to_add_tile)

    return out


def ttnn_conv_transpose1d(
    x: Any,
    weight: Any,
    bias: Optional[Any] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    device: Optional[Any] = None,
    conv_config: Optional[Any] = None,
    compute_config: Optional[Any] = None,
) -> Any:
    """
    Perform 1D transposed convolution (upsampling) using TTNN's conv_transpose2d.

    Used in the HiFi-GAN vocoder for upsampling latent representations to audio.

    Args:
        x: Input tensor [B, C_in, L]
        weight: Weight tensor [C_in, C_out, 1, K] (transposed conv format)
        bias: Optional bias tensor [C_out]
        stride: Upsampling stride
        padding: Convolution padding
        output_padding: Additional padding for output size
        groups: Number of groups
        device: TTNN device
        conv_config: TTNN Conv2dConfig
        compute_config: TTNN compute kernel config

    Returns:
        Output tensor [B, C_out, L_out] where L_out = (L-1)*stride - 2*padding + K + output_padding
    """
    # Check if inputs are PyTorch tensors - use PyTorch path
    # Check BOTH x and weight to avoid mixing torch/ttnn operations
    is_torch_x = isinstance(x, torch.Tensor)
    is_torch_weight = isinstance(weight, torch.Tensor)
    is_torch = is_torch_x or is_torch_weight

    # Force PyTorch for large stride operations (L1 memory constraints)
    # HiFi-GAN uses kernel=16, stride=8 which exceeds L1 buffer limits
    # Use stride >= 2 as the trigger - large upsampling operations cause L1 issues
    large_stride_op = stride >= 2

    if not TTNN_AVAILABLE or is_torch or large_stride_op:
        # Convert TTNN tensors to PyTorch if needed
        if not is_torch_x and TTNN_AVAILABLE:
            x = ttnn.to_torch(x)
        if not is_torch_weight and TTNN_AVAILABLE:
            weight = ttnn.to_torch(weight)
        if bias is not None and not isinstance(bias, torch.Tensor) and TTNN_AVAILABLE:
            bias = ttnn.to_torch(bias)

        # Ensure float32 for PyTorch operations
        x = x.float() if x.dtype != torch.float32 else x
        weight = weight.float() if weight.dtype != torch.float32 else weight
        if bias is not None:
            bias = bias.float() if bias.dtype != torch.float32 else bias

        # Fallback to PyTorch
        if x.dim() == 4:
            x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, L, C] -> [B, C, L]
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, L*C] case
        weight_3d = weight.squeeze(2) if weight.dim() == 4 else weight  # [C_in, C_out, 1, K] -> [C_in, C_out, K]
        out = F.conv_transpose1d(x, weight_3d, bias, stride, padding, output_padding, groups)
        # Convert back to TTNN if input was TTNN
        if large_stride_op and (not is_torch_x or not is_torch_weight) and TTNN_AVAILABLE and device is not None:
            # Use TILE_LAYOUT for compatibility with TTNN operations like leaky_relu
            out = ttnn.from_torch(out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return out

    # Get dimensions
    is_3d_input = x.shape.rank == 3

    if is_3d_input:
        # Reshape [B, C, L] -> [B, 1, L, C] for TTNN (NHWC format)
        batch, channels, length = x.shape
        x = ttnn.permute(x, (0, 2, 1))  # [B, C, L] -> [B, L, C]
        x = ttnn.reshape(x, (batch, 1, length, channels))  # [B, L, C] -> [B, 1, L, C]

    # Get weight dimensions - handle both 3D [C_in, C_out, K] and 4D [C_in, C_out, 1, K]
    weight_rank = weight.shape.rank if hasattr(weight.shape, 'rank') else len(weight.shape)
    if weight_rank == 3:
        in_channels, out_channels, kernel_size = weight.shape
        # Reshape to 4D for conv_transpose2d: [C_in, C_out, K] -> [C_in, C_out, 1, K]
        weight = ttnn.reshape(weight, (in_channels, out_channels, 1, kernel_size))
    else:
        in_channels, out_channels, _, kernel_size = weight.shape

    # x.shape is [B, H, W, C] after reshape where H=1, W=L (sequence length), C=channels
    batch_size, input_height, input_width, input_channels = x.shape


    # Default conv config if not provided
    if conv_config is None:
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            config_tensors_in_dram=True,
            deallocate_activation=False,
            reallocate_halo_output=False,
        )

    # Default compute config - use HiFi4 for optimal accuracy
    if compute_config is None and device is not None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    # Conv kwargs shared between prepare_conv_transpose2d_weights and conv_transpose2d
    conv_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (1, kernel_size),
        "stride": (1, stride),
        "padding": (0, padding),
        "dilation": (1, 1),
        "groups": groups,
    }

    # Track bias for manual addition after conv_transpose2d
    bias_to_add = None

    # Prepare conv transpose weights if not already on device
    if device is not None and not ttnn.is_tensor_storage_on_device(weight):
        weight = ttnn.prepare_conv_transpose2d_weights(
            weight_tensor=weight,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=x.get_layout(),
            weights_format="IOHW",
            has_bias=False,  # We'll add bias manually
            input_dtype=x.dtype,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            **conv_kwargs,
        )
        # prepare_conv_transpose2d_weights may already put weight on device
        if not ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.to_device(weight, device)

        # Store bias for manual addition after conv_transpose2d
        if bias is not None and not ttnn.is_tensor_storage_on_device(bias):
            # Reshape 1D bias to broadcastable shape [1, C_out, 1] for [B, C, L] output
            if bias.shape.rank == 1:
                bias_to_add = ttnn.reshape(bias, (1, out_channels, 1))
            else:
                bias_to_add = bias

    # Run conv_transpose2d without bias
    out = ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=weight,
        device=device,
        bias_tensor=None,  # Handle bias manually
        conv_config=conv_config,
        compute_config=compute_config,
        **conv_kwargs,
    )

    if is_3d_input:
        # Reshape back [B, 1, L_out, C_out] -> [B, C_out, L_out]
        if out.shape.rank == 4:
            batch = out.shape[0]
            out_length = out.shape[2]
            out_ch = out.shape[3]
            out = ttnn.reshape(out, (batch, out_length, out_ch))  # [B, L, C]
            out = ttnn.permute(out, (0, 2, 1))  # [B, C, L]
        elif out.shape.rank == 2:
            # Flattened output [B*L, C] - need to reshape
            total = out.shape[0]
            out_ch = out.shape[1]
            out_length = total // batch_size
            out = ttnn.reshape(out, (batch_size, out_length, out_ch))  # [B, L, C]
            out = ttnn.permute(out, (0, 2, 1))  # [B, C, L]

    # Add bias after reshape (if we stored it earlier)
    if bias_to_add is not None and device is not None:
        # Convert bias to TILE layout for broadcast
        bias_to_add_tile = ttnn.to_layout(bias_to_add, ttnn.TILE_LAYOUT)
        bias_to_add_tile = ttnn.to_device(bias_to_add_tile, device)
        out = ttnn.add(out, bias_to_add_tile)

    return out


class Conv1dLayer:
    """
    Wrapper class for a single Conv1d layer with pre-loaded weights.

    Provides a callable interface matching PyTorch nn.Conv1d.
    """

    def __init__(
        self,
        weight: Any,
        bias: Optional[Any] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        device: Optional[Any] = None,
        activation: Optional[str] = None,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.device = device
        self.activation = activation

        # Pre-compute conv config for efficiency
        self._conv_config = None
        self._compute_config = None

    def __call__(self, x: Any) -> Any:
        return ttnn_conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            device=self.device,
            conv_config=self._conv_config,
            compute_config=self._compute_config,
            activation=self.activation,
        )
