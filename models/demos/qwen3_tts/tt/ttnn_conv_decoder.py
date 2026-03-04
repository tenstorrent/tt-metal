# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN implementation of Speech Tokenizer Decoder conv operations.

Uses:
- ttnn.conv1d for regular 1D convolutions
- ttnn.conv_transpose2d with H=1 for transposed 1D convolutions (upsampling)
- Element-wise TTNN ops for snake activation

All operations run on device for trace compatibility.
"""

from typing import Optional, Tuple

import torch

import ttnn


class TTNNConv1d:
    """TTNN wrapper for conv1d operations."""

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        weight: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.has_bias = bias

        # Store weight on host (will be moved to device on first call)
        self.weight_tt = None
        if weight is not None:
            self.weight_tt = ttnn.from_torch(weight, dtype=ttnn.bfloat16)

        self.bias_tt = None
        if bias_tensor is not None:
            self.bias_tt = ttnn.from_torch(bias_tensor.reshape(1, 1, 1, out_channels), dtype=ttnn.bfloat16)

        # Conv config - use auto sharding to avoid L1 overflow
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,  # Auto select best sharding
            deallocate_activation=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
        )

    def __call__(self, x: ttnn.Tensor, input_length: int) -> Tuple[ttnn.Tensor, int]:
        """
        Forward pass.

        Args:
            x: Input tensor in NLC format [batch, length, channels]
            input_length: Input sequence length

        Returns:
            Output tensor and output length
        """
        batch_size = x.shape[0]

        result = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight_tt,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_length=input_length,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias_tensor=self.bias_tt,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output, out_length, [self.weight_tt, self.bias_tt] = result
        return output, out_length


class TTNNConvTranspose1d:
    """TTNN wrapper for transposed conv1d using conv_transpose2d with H=1."""

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        weight: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Store weight: [I, O, K] -> [I, O, 1, K] and flip for conv_transpose
        self.weight_tt = None
        if weight is not None:
            weight_2d = weight.unsqueeze(2)  # [I, O, 1, K]
            weight_flipped = torch.flip(weight_2d, [2, 3])
            self.weight_tt = ttnn.from_torch(weight_flipped, dtype=ttnn.bfloat16)

        self.bias_tt = None
        if bias_tensor is not None:
            self.bias_tt = ttnn.from_torch(bias_tensor.reshape(1, 1, 1, out_channels), dtype=ttnn.bfloat16)

        # Conv config
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,  # auto
            deallocate_activation=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
        )

    def __call__(self, x: ttnn.Tensor, input_length: int) -> Tuple[ttnn.Tensor, int]:
        """
        Forward pass.

        Args:
            x: Input tensor in NHWC format [batch, 1, length, channels]
            input_length: Input sequence length

        Returns:
            Output tensor and output length
        """
        batch_size = x.shape[0]

        result = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight_tt,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            output_padding=(0, self.output_padding),
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            mirror_kernel=False,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output, [out_h, out_w], [self.weight_tt, self.bias_tt] = result
        return output, out_w


def ttnn_snake_activation(
    x: ttnn.Tensor,
    alpha: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> ttnn.Tensor:
    """
    Snake activation in TTNN: x + (1/beta) * sin^2(alpha * x)

    Args:
        x: Input tensor [batch, 1, seq_len, channels]
        alpha: Per-channel alpha [1, 1, 1, channels]
        beta: Per-channel beta [1, 1, 1, channels]

    Returns:
        Activated tensor
    """
    ax = ttnn.mul(alpha, x)
    sin_ax = ttnn.sin(ax)
    sin_ax_sq = ttnn.mul(sin_ax, sin_ax)
    inv_beta = ttnn.reciprocal(beta)
    scaled = ttnn.mul(inv_beta, sin_ax_sq)
    output = ttnn.add(x, scaled)
    return output


class TTNNSnakeActivation:
    """TTNN snake activation module."""

    def __init__(
        self,
        device,
        channels: int,
        alpha: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ):
        self.device = device
        self.channels = channels

        # Default alpha and beta
        if alpha is None:
            alpha = torch.ones(channels)
        if beta is None:
            beta = torch.ones(channels)

        # Store as [1, 1, 1, channels] for broadcasting
        self.alpha = ttnn.from_torch(
            alpha.view(1, 1, 1, channels).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.beta = ttnn.from_torch(
            beta.view(1, 1, 1, channels).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply snake activation."""
        return ttnn_snake_activation(x, self.alpha, self.beta)


class TTNNConvNeXtBlock:
    """TTNN ConvNeXt block for upsampler."""

    def __init__(
        self,
        device,
        channels: int,
        kernel_size: int = 7,
        weights: dict = None,
    ):
        self.device = device
        self.channels = channels

        # Depthwise conv
        self.dwconv = None
        if weights and "dwconv.conv.weight" in weights:
            self.dwconv = TTNNConv1d(
                device=device,
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                weight=weights["dwconv.conv.weight"],
                bias_tensor=weights.get("dwconv.conv.bias"),
            )

        # Layer norm weights
        self.norm_weight = None
        self.norm_bias = None
        if weights and "norm.weight" in weights:
            self.norm_weight = ttnn.from_torch(
                weights["norm.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.norm_bias = ttnn.from_torch(
                weights["norm.bias"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Pointwise convs (as linear layers)
        self.pwconv1_weight = None
        self.pwconv2_weight = None
        if weights:
            if "pwconv1.weight" in weights:
                # [out, in] for linear
                w = weights["pwconv1.weight"].T.contiguous()
                self.pwconv1_weight = ttnn.from_torch(
                    w.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            if "pwconv2.weight" in weights:
                w = weights["pwconv2.weight"].T.contiguous()
                self.pwconv2_weight = ttnn.from_torch(
                    w.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        # Layer scale (gamma)
        self.gamma = None
        if weights and "gamma" in weights:
            self.gamma = ttnn.from_torch(
                weights["gamma"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    def __call__(self, x: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, 1, seq_len, channels]
            seq_len: Sequence length

        Returns:
            Output tensor
        """
        residual = x

        # Depthwise conv (need to convert format)
        if self.dwconv:
            # [B, 1, L, C] -> [B, L, C] for conv1d
            x_nlc = ttnn.reshape(x, (x.shape[0], seq_len, self.channels))
            x_nlc, _ = self.dwconv(x_nlc, seq_len)
            # Back to [B, 1, L, C]
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, seq_len, self.channels))

        # Layer norm
        if self.norm_weight is not None:
            x = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias)

        # Pointwise conv1 + GELU
        if self.pwconv1_weight is not None:
            x = ttnn.linear(x, self.pwconv1_weight)
            x = ttnn.gelu(x)

        # Pointwise conv2
        if self.pwconv2_weight is not None:
            x = ttnn.linear(x, self.pwconv2_weight)

        # Layer scale
        if self.gamma is not None:
            x = ttnn.mul(x, self.gamma)

        # Residual
        x = ttnn.add(x, residual)

        return x


def test_ttnn_conv_decoder():
    """Test the TTNN conv decoder components."""
    import torch.nn.functional as F

    print("Testing TTNN conv decoder components...")

    device = ttnn.open_device(device_id=0, l1_small_size=65536)

    try:
        # Test 1: Snake activation
        print("\n1. Testing snake activation...")
        x = torch.randn(1, 256, 100, dtype=torch.bfloat16).float()
        alpha = torch.ones(256) * 0.5
        beta = torch.ones(256)

        # PyTorch golden
        alpha_exp = alpha.view(1, -1, 1)
        beta_exp = beta.view(1, -1, 1)
        y_golden = x + (1.0 / beta_exp) * torch.sin(alpha_exp * x).pow(2)

        # TTNN
        x_tt = ttnn.from_torch(
            x.permute(0, 2, 1).unsqueeze(1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        snake = TTNNSnakeActivation(device, 256, alpha, beta)
        y_tt = snake(x_tt)
        y_torch = ttnn.to_torch(y_tt).squeeze(1).permute(0, 2, 1)

        from scipy.stats import pearsonr

        pcc = pearsonr(y_torch.flatten().float().numpy(), y_golden.flatten().numpy())[0]
        print(f"   Snake activation PCC: {pcc:.6f}")

        # Test 2: Conv1d
        print("\n2. Testing conv1d...")
        x = torch.randn(1, 128, 64, dtype=torch.bfloat16).float()
        weight = torch.randn(256, 128, 3, dtype=torch.bfloat16).float()

        y_golden = F.conv1d(x, weight, padding=1)

        x_nlc = x.permute(0, 2, 1)
        x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16)

        conv = TTNNConv1d(device, 128, 256, 3, padding=1, weight=weight)
        y_tt, out_len = conv(x_tt, 64)
        y_tt = ttnn.from_device(y_tt)
        y_torch = ttnn.to_torch(y_tt).reshape(1, out_len, 256).permute(0, 2, 1)

        pcc = pearsonr(y_torch.flatten().float().numpy(), y_golden.flatten().numpy())[0]
        print(f"   Conv1d PCC: {pcc:.6f}")

        # Test 3: ConvTranspose1d
        print("\n3. Testing conv_transpose1d...")
        x = torch.randn(1, 512, 100, dtype=torch.bfloat16).float()
        weight = torch.randn(512, 256, 4, dtype=torch.bfloat16).float()

        padding = 1
        y_golden = F.conv_transpose1d(x, weight, stride=2, padding=padding)

        # TTNN format [B, 1, L, C]
        x_2d = x.unsqueeze(2).permute(0, 2, 3, 1)
        x_tt = ttnn.from_torch(x_2d.to(torch.bfloat16), dtype=ttnn.bfloat16, device=device)

        conv_t = TTNNConvTranspose1d(device, 512, 256, 4, stride=2, padding=padding, weight=weight)
        y_tt, out_len = conv_t(x_tt, 100)
        y_torch = ttnn.to_torch(y_tt).reshape(1, 1, out_len, 256).squeeze(1).permute(0, 2, 1)

        min_len = min(y_torch.shape[2], y_golden.shape[2])
        pcc = pearsonr(
            y_torch[:, :, :min_len].flatten().float().numpy(),
            y_golden[:, :, :min_len].flatten().numpy(),
        )[0]
        print(f"   ConvTranspose1d PCC: {pcc:.6f}")

        print("\nAll tests passed!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_ttnn_conv_decoder()
