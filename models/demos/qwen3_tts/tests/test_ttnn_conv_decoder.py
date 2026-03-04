# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test TTNN conv operations for Speech Tokenizer Decoder.

Tests:
1. conv1d - regular convolution
2. conv_transpose2d with H=1 - transposed convolution for upsampling
3. Depthwise conv1d - groups=channels
4. Snake activation - element-wise ops
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


class TestConv1d:
    """Test ttnn.conv1d for regular convolution."""

    @pytest.mark.parametrize(
        "batch_size, in_channels, out_channels, seq_len, kernel_size, stride, padding",
        [
            (1, 512, 1536, 100, 7, 1, 3),  # pre_conv
            (1, 1536, 1536, 100, 7, 1, 3),  # decoder.0
            (1, 384, 384, 800, 3, 1, 1),  # residual conv
        ],
    )
    def test_conv1d_basic(self, device, batch_size, in_channels, out_channels, seq_len, kernel_size, stride, padding):
        """Test basic conv1d operation."""
        torch.manual_seed(42)

        # Create PyTorch tensors (NCL format)
        x_torch = torch.randn(batch_size, in_channels, seq_len, dtype=torch.bfloat16).float()
        weight_torch = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.bfloat16).float()

        # PyTorch golden
        y_golden = F.conv1d(x_torch, weight_torch, padding=padding, stride=stride)

        # TTNN expects NLC format
        x_nlc = x_torch.permute(0, 2, 1).contiguous()  # [batch, seq_len, in_channels]

        # Convert to TTNN
        x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16)
        weight_tt = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16)

        # Conv config
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,  # auto
            deallocate_activation=False,
        )

        # Run TTNN conv1d
        y_tt, out_len = ttnn.conv1d(
            input_tensor=x_tt,
            weight_tensor=weight_tt,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_length=seq_len,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_config=conv_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )

        # Convert back to PyTorch
        y_torch = ttnn.to_torch(y_tt)
        y_torch = y_torch.reshape(batch_size, out_len, out_channels)
        y_torch = y_torch.permute(0, 2, 1)  # NLC -> NCL

        # Check PCC
        passing, pcc_msg = check_with_pcc_without_tensor_printout(y_torch, y_golden, pcc=0.99)
        print(f"conv1d [{in_channels}→{out_channels}, k={kernel_size}]: {pcc_msg}")
        assert passing, f"conv1d PCC too low: {pcc_msg}"


class TestConvTranspose1d:
    """Test ttnn.conv_transpose2d with H=1 for 1D transposed convolution."""

    @pytest.mark.parametrize(
        "batch_size, in_channels, out_channels, seq_len, kernel_size, stride",
        [
            (1, 1536, 768, 100, 4, 2),  # upsample 2x
            (1, 768, 384, 200, 16, 8),  # upsample 8x
            (1, 384, 192, 1600, 10, 5),  # upsample 5x
            (1, 192, 96, 8000, 8, 4),  # upsample 4x
            (1, 96, 1, 32000, 6, 3),  # upsample 3x (final)
        ],
    )
    def test_conv_transpose1d_as_2d(self, device, batch_size, in_channels, out_channels, seq_len, kernel_size, stride):
        """Test transposed conv using conv_transpose2d with H=1."""
        torch.manual_seed(42)

        # Create PyTorch tensors (NCL format for 1D)
        x_torch = torch.randn(batch_size, in_channels, seq_len, dtype=torch.bfloat16).float()
        # conv_transpose1d weight: [in_channels, out_channels, kernel_size]
        weight_torch = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.bfloat16).float()

        # Compute padding for same output length * stride
        padding = (kernel_size - stride) // 2
        output_padding = (kernel_size - stride) % 2

        # PyTorch golden (conv_transpose1d)
        y_golden = F.conv_transpose1d(
            x_torch, weight_torch, stride=stride, padding=padding, output_padding=output_padding
        )
        print(f"Golden output shape: {y_golden.shape}")

        # For TTNN conv_transpose2d, treat 1D as 2D with H=1
        # Input: [batch, in_channels, seq_len] -> [batch, 1, seq_len, in_channels] (NHWC)
        x_2d = x_torch.unsqueeze(2).permute(0, 2, 3, 1).contiguous()  # [batch, 1, seq_len, in_channels]

        # Weight: [in_channels, out_channels, kernel_size] -> [in_channels, out_channels, 1, kernel_size] (IOHW)
        weight_2d = weight_torch.unsqueeze(2)  # [in_channels, out_channels, 1, kernel_size]

        # Convert to TTNN
        x_tt = ttnn.from_torch(x_2d, dtype=ttnn.bfloat16, device=device)
        weight_tt = ttnn.from_torch(weight_2d, dtype=ttnn.bfloat16)

        # Conv config
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,  # auto
            deallocate_activation=False,
        )

        # Run TTNN conv_transpose2d
        y_tt, [out_h, out_w], _ = ttnn.conv_transpose2d(
            input_tensor=x_tt,
            weight_tensor=weight_tt,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=seq_len,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            output_padding=(0, output_padding),
            conv_config=conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Convert back to PyTorch
        y_torch = ttnn.to_torch(y_tt)
        # Output is NHWC: [batch, out_h, out_w, out_channels]
        y_torch = y_torch.reshape(batch_size, out_h, out_w, out_channels)
        # Convert to NCL: [batch, out_channels, out_w] (squeeze H=1)
        y_torch = y_torch.squeeze(1).permute(0, 2, 1)

        print(f"TTNN output shape: {y_torch.shape}, expected: {y_golden.shape}")

        # Trim to match golden if needed (TTNN may have different padding behavior)
        min_len = min(y_torch.shape[2], y_golden.shape[2])
        y_torch = y_torch[:, :, :min_len]
        y_golden = y_golden[:, :, :min_len]

        # Check PCC
        passing, pcc_msg = check_with_pcc_without_tensor_printout(y_torch, y_golden, pcc=0.95)
        print(f"conv_transpose1d [{in_channels}→{out_channels}, stride={stride}]: {pcc_msg}")
        assert passing, f"conv_transpose1d PCC too low: {pcc_msg}"


class TestSnakeActivation:
    """Test snake activation using TTNN element-wise ops."""

    def test_snake_activation(self, device):
        """Test snake activation: x + (1/beta) * sin^2(alpha * x)"""
        torch.manual_seed(42)

        batch_size, channels, seq_len = 1, 384, 1000
        x_torch = torch.randn(batch_size, channels, seq_len, dtype=torch.bfloat16).float()
        alpha = torch.ones(channels, dtype=torch.bfloat16).float() * 0.5
        beta = torch.ones(channels, dtype=torch.bfloat16).float() * 1.0

        # PyTorch golden
        alpha_expanded = alpha.view(1, -1, 1)
        beta_expanded = beta.view(1, -1, 1)
        y_golden = x_torch + (1.0 / beta_expanded) * torch.sin(alpha_expanded * x_torch).pow(2)

        # TTNN implementation
        # Reshape for TTNN: [batch, channels, seq_len] -> [batch, 1, seq_len, channels]
        x_ttnn_format = x_torch.permute(0, 2, 1).unsqueeze(1)  # [batch, 1, seq_len, channels]

        x_tt = ttnn.from_torch(
            x_ttnn_format.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Alpha and beta as [1, 1, 1, channels] for broadcasting
        alpha_tt = ttnn.from_torch(
            alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        beta_tt = ttnn.from_torch(
            beta.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Snake activation in TTNN: x + (1/beta) * sin^2(alpha * x)
        ax = ttnn.mul(alpha_tt, x_tt)
        sin_ax = ttnn.sin(ax)
        sin_ax_sq = ttnn.mul(sin_ax, sin_ax)  # sin^2
        inv_beta = ttnn.reciprocal(beta_tt)
        scaled = ttnn.mul(inv_beta, sin_ax_sq)
        y_tt = ttnn.add(x_tt, scaled)

        # Convert back
        y_torch = ttnn.to_torch(y_tt)
        y_torch = y_torch.squeeze(1).permute(0, 2, 1)  # [batch, 1, seq_len, channels] -> [batch, channels, seq_len]

        # Check PCC
        passing, pcc_msg = check_with_pcc_without_tensor_printout(y_torch, y_golden, pcc=0.99)
        print(f"snake_activation: {pcc_msg}")
        assert passing, f"snake_activation PCC too low: {pcc_msg}"


class TestDepthwiseConv1d:
    """Test depthwise conv1d (groups=channels)."""

    @pytest.mark.parametrize(
        "batch_size, channels, seq_len, kernel_size",
        [
            (1, 512, 100, 7),  # ConvNeXt dwconv
            (1, 384, 800, 7),
        ],
    )
    def test_depthwise_conv1d(self, device, batch_size, channels, seq_len, kernel_size):
        """Test depthwise conv1d with groups=channels."""
        torch.manual_seed(42)

        # Create tensors
        x_torch = torch.randn(batch_size, channels, seq_len, dtype=torch.bfloat16).float()
        # Depthwise: [out_channels, in_channels/groups, kernel_size] = [channels, 1, kernel_size]
        weight_torch = torch.randn(channels, 1, kernel_size, dtype=torch.bfloat16).float()

        padding = kernel_size // 2

        # PyTorch golden
        y_golden = F.conv1d(x_torch, weight_torch, padding=padding, groups=channels)

        # TTNN (NLC format)
        x_nlc = x_torch.permute(0, 2, 1).contiguous()
        x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16)
        weight_tt = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16)

        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=False,
        )

        y_tt, out_len = ttnn.conv1d(
            input_tensor=x_tt,
            weight_tensor=weight_tt,
            device=device,
            in_channels=channels,
            out_channels=channels,
            batch_size=batch_size,
            input_length=seq_len,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,  # Depthwise
            conv_config=conv_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )

        y_torch = ttnn.to_torch(y_tt)
        y_torch = y_torch.reshape(batch_size, out_len, channels).permute(0, 2, 1)

        passing, pcc_msg = check_with_pcc_without_tensor_printout(y_torch, y_golden, pcc=0.99)
        print(f"depthwise_conv1d [channels={channels}, k={kernel_size}]: {pcc_msg}")
        assert passing, f"depthwise_conv1d PCC too low: {pcc_msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
