# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Attention DenseUNet - PyTorch Reference Implementation

This module implements Attention DenseUNet, which combines:
1. DenseNet encoder with densely connected blocks
2. Attention gates for skip connections
3. U-Net style decoder path

Architecture based on:
- DenseNet: https://arxiv.org/abs/1608.06993
- Attention U-Net: https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """
    Single dense layer following DenseNet pattern: BN -> ReLU -> Conv

    Args:
        in_channels: Number of input channels
        growth_rate: Number of output channels (k in paper)
        bn_size: Bottleneck size multiplier (typically 4)
    """

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """
    Dense block containing multiple dense layers.
    Each layer receives feature maps from all preceding layers.

    Args:
        in_channels: Number of input channels
        num_layers: Number of dense layers in this block
        growth_rate: Number of new channels added per layer
        bn_size: Bottleneck size multiplier
    """

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int, bn_size: int = 4):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(DenseLayer(layer_in_channels, growth_rate, bn_size))

        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionDown(nn.Module):
    """
    Transition layer for downsampling between dense blocks.
    Compresses channels and reduces spatial dimensions by 2x.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (typically in_channels * compression)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionDown, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.relu(self.bn(x), inplace=True))
        return self.pool(out)


class TransitionUp(nn.Module):
    """
    Transition layer for upsampling in decoder path.
    Uses transposed convolution for 2x upsampling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionUp, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upconv(x)


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.

    Computes spatial attention to emphasize relevant features
    from the skip connection based on the gating signal.

    Args:
        in_channels: Number of channels in skip connection (x)
        gating_channels: Number of channels in gating signal (g)
        inter_channels: Number of intermediate channels (typically in_channels // 2)
    """

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int = None):
        super(AttentionGate, self).__init__()

        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Skip connection features (higher resolution)
            g: Gating signal from decoder (lower resolution)

        Returns:
            Attention-weighted skip connection features
        """
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        attention = torch.sigmoid(self.psi(f))
        y = attention * x
        return self.W(y)


class DecoderBlock(nn.Module):
    """
    Decoder block with two convolutions.

    Args:
        in_channels: Number of input channels (after concatenation with skip)
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class AttentionDenseUNet(nn.Module):
    """
    Attention DenseUNet for image segmentation.

    Combines DenseNet encoder with attention-gated skip connections
    in a U-Net style architecture.

    Architecture (for input 256x256):
        Encoder:
            Input (3, 256, 256) -> Conv0 -> (32, 256, 256)
            DenseBlock1 -> (96, 256, 256) -> TransitionDown -> (48, 128, 128)
            DenseBlock2 -> (112, 128, 128) -> TransitionDown -> (56, 64, 64)
            DenseBlock3 -> (120, 64, 64) -> TransitionDown -> (60, 32, 32)
            DenseBlock4 -> (124, 32, 32) -> TransitionDown -> (62, 16, 16)

        Bottleneck:
            (62, 16, 16) -> Conv -> (62, 16, 16)

        Decoder:
            UpConv + Attention + Decoder4: (62, 16, 16) -> (124, 32, 32)
            UpConv + Attention + Decoder3: (124, 32, 32) -> (120, 64, 64)
            UpConv + Attention + Decoder2: (120, 64, 64) -> (112, 128, 128)
            UpConv + Attention + Decoder1: (112, 128, 128) -> (96, 256, 256)

        Output:
            1x1 Conv -> (1, 256, 256)

    Args:
        in_channels: Number of input image channels (default: 3 for RGB)
        out_channels: Number of output classes (default: 1 for binary segmentation)
        init_features: Initial number of features after first convolution
        growth_rate: DenseNet growth rate (k)
        num_layers_per_block: Number of dense layers per block
        compression: Channel compression factor for transitions (0-1)
        bn_size: Bottleneck multiplier for dense layers
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        growth_rate: int = 16,
        num_layers_per_block: tuple = (4, 4, 4, 4),
        compression: float = 0.5,
        bn_size: int = 4,
    ):
        super(AttentionDenseUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.growth_rate = growth_rate
        self.num_encoder_blocks = len(num_layers_per_block)
        self.conv0 = nn.Conv2d(in_channels, init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(init_features)
        current_channels = init_features
        self.encoder_blocks = nn.ModuleList()
        self.transitions_down = nn.ModuleList()
        self.skip_channels = []

        for i, num_layers in enumerate(num_layers_per_block):
            dense_block = DenseBlock(current_channels, num_layers, growth_rate, bn_size)
            self.encoder_blocks.append(dense_block)
            current_channels = dense_block.out_channels
            self.skip_channels.append(current_channels)
            trans_out_channels = int(current_channels * compression)
            trans_down = TransitionDown(current_channels, trans_out_channels)
            self.transitions_down.append(trans_down)
            current_channels = trans_out_channels

        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_channels = current_channels

        self.transitions_up = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        reversed_skip_channels = list(reversed(self.skip_channels))

        for i, skip_ch in enumerate(reversed_skip_channels):
            if i == 0:
                trans_in = self.bottleneck_channels
            else:
                trans_in = decoder_out_channels
            trans_out = skip_ch
            self.transitions_up.append(TransitionUp(trans_in, trans_out))
            self.attention_gates.append(AttentionGate(skip_ch, trans_out))
            decoder_in = trans_out + skip_ch  # upsampled + attended skip
            decoder_out_channels = skip_ch
            self.decoder_blocks.append(DecoderBlock(decoder_in, decoder_out_channels))
        self.conv_out = nn.Conv2d(decoder_out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn0(self.conv0(x)), inplace=True)
        skips = []
        for i in range(self.num_encoder_blocks):
            x = self.encoder_blocks[i](x)
            skips.append(x)
            x = self.transitions_down[i](x)
        x = self.bottleneck(x)
        reversed_skips = list(reversed(skips))
        for i, (trans_up, att_gate, decoder) in enumerate(
            zip(self.transitions_up, self.attention_gates, self.decoder_blocks)
        ):
            x = trans_up(x)
            skip = reversed_skips[i]
            attended_skip = att_gate(skip, x)
            x = torch.cat([x, attended_skip], dim=1)
            x = decoder(x)

        return self.conv_out(x)


def create_attention_denseunet(
    in_channels: int = 3,
    out_channels: int = 1,
    init_features: int = 32,
    growth_rate: int = 16,
    pretrained: bool = False,
) -> AttentionDenseUNet:
    """
    Factory function to create AttentionDenseUNet model.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        init_features: Initial feature channels
        growth_rate: DenseNet growth rate
        pretrained: Whether to load pretrained weights (not implemented yet)

    Returns:
        AttentionDenseUNet model
    """
    model = AttentionDenseUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        growth_rate=growth_rate,
        num_layers_per_block=(4, 4, 4, 4),
        compression=0.5,
        bn_size=4,
    )

    if pretrained:
        raise NotImplementedError("Pretrained weights not yet available")

    return model


if __name__ == "__main__":
    model = create_attention_denseunet()
    model.eval()

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert x.shape[2:] == y.shape[2:], f"Spatial dimensions should match! Got {x.shape} -> {y.shape}"
    print("✓ Output shape matches input spatial dimensions")
