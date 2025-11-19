# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of DVAE for PCC validation.

Simplified implementation focusing on encoder/decoder architecture.
"""

import torch
import torch.nn as nn


class PyTorchDVAE(nn.Module):
    """
    Simplified PyTorch reference implementation of DVAE.
    """

    def __init__(
        self,
        num_encoder_layers: int = 12,  # Production: 12 layers
        num_decoder_layers: int = 12,  # Production: 12 layers
        hidden_dim: int = 256,
        num_mel_bins: int = 100,
        bn_dim: int = 128,  # Production: 128
        enable_gfsq: bool = True,  # Enable/disable GFSQ quantization
    ):
        """
        PyTorch reference DVAE with production configuration.

        Production Configuration (from MiniCPM-o-2_6):
        - Encoder: 12 ConvNeXt blocks, hidden_dim=256, bn_dim=128
        - Decoder: 12 ConvNeXt blocks, hidden_dim=256, bn_dim=128

        Args:
            num_encoder_layers: Number of encoder ConvNeXt blocks (default 12 for production)
            num_decoder_layers: Number of decoder ConvNeXt blocks (default 12 for production)
            hidden_dim: Hidden dimension (default 256)
            num_mel_bins: Number of mel bins (default 100)
            bn_dim: Bottleneck dimension (default 128)
            enable_gfsq: Enable GFSQ quantization (default True)
        """
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim
        self.num_mel_bins = num_mel_bins
        self.enable_gfsq = enable_gfsq
        self.bn_dim = bn_dim

        # Coefficient
        self.coef = nn.Parameter(torch.randn(1, num_mel_bins, 1))

        # Encoder downsampling (2D conv)
        self.encoder_downsample = nn.Sequential(
            nn.Conv2d(num_mel_bins, 512, (1, 3), padding=(0, 1)),  # 2D conv: H=1, W=kernel
            nn.GELU(),  # Production: GELU instead of ReLU
            nn.Conv2d(512, 512, (1, 4), stride=(1, 2), padding=(0, 1)),  # 2D conv with stride
            nn.GELU(),  # Production: GELU instead of ReLU
        )

        # Encoder input (Production: bn_dim=128)
        self.encoder_input = nn.Sequential(
            nn.Conv2d(512, bn_dim, (1, 3), padding=(0, 1)),
            nn.GELU(),  # Production: GELU instead of ReLU
            nn.Conv2d(bn_dim, hidden_dim, (1, 3), padding=(0, 1)),
            nn.GELU(),  # Production: GELU instead of ReLU
        )

        # Encoder blocks (2D ConvNeXt)
        self.encoder_blocks = nn.ModuleList([ConvNeXtBlock2D(hidden_dim) for _ in range(num_encoder_layers)])

        # Encoder output
        self.encoder_output = nn.Conv2d(hidden_dim, 1024, (1, 1))  # 1x1 conv

        # Decoder input (Production: decoder processes 1024 channels from encoder)
        self.decoder_input = nn.Sequential(
            nn.Conv2d(1024, bn_dim, (1, 3), padding=(0, 1)),  # Production: 1024 input channels from encoder
            nn.GELU(),  # Production: GELU instead of ReLU
            nn.Conv2d(bn_dim, hidden_dim, (1, 3), padding=(0, 1)),
            nn.GELU(),  # Production: GELU instead of ReLU
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([ConvNeXtBlock2D(hidden_dim) for _ in range(num_decoder_layers)])

        # Decoder projection: hidden_dim -> 512 channels (NEW layer)
        self.decoder_proj = nn.Conv2d(hidden_dim, 512, (1, 1))  # 1x1 conv

        # Decoder output (Production: 512 -> num_mel_bins)
        self.decoder_output = nn.Conv2d(512, num_mel_bins, (1, 3), padding=(0, 1))

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mel_spectrogram: [batch_size, num_mel_bins, time_steps]

        Returns:
            torch.Tensor: Reconstructed mel spectrogram
        """
        # Reshape for 2D conv: [batch, mel_bins, time_steps] -> [batch, 1, time_steps, mel_bins] (NHWC)
        batch_size, num_mel_bins, time_steps = mel_spectrogram.shape
        x = mel_spectrogram.permute(0, 2, 1).unsqueeze(1)  # [batch, 1, time_steps, mel_bins]

        # Encoder
        encoded = self._encode(x)

        # Apply GFSQ quantization (or bypass if disabled)
        if self.enable_gfsq:
            # Apply GFSQ quantization (simplified - pass through for now)
            quantized = encoded
        else:
            # Bypass quantization - pass through unchanged
            quantized = encoded

        # Decoder
        reconstructed = self._decode(quantized)

        # Reshape back: [batch, 1, time_steps, mel_bins] -> [batch, mel_bins, time_steps] (NCHW)
        reconstructed = reconstructed.squeeze(1).permute(0, 2, 1)

        return reconstructed

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward pass.
        Input x: [batch, 1, time_steps, mel_bins] (NHWC)
        """
        # Skip coefficient for testing basic conv operations
        # coef_expanded = self.coef.unsqueeze(-1).unsqueeze(1)  # [1, 1, mel_bins, 1]
        # x = x * coef_expanded

        # Convert NHWC to NCHW for downsampling: [B, 1, T, C] -> [B, C, 1, T]
        x = x.permute(0, 3, 1, 2)  # [B, C, 1, T]

        # Downsampling (NCHW format)
        x = self.encoder_downsample(x)

        # Keep in NCHW for encoder_input: [B, C, 1, T]

        # Input processing (NCHW)
        x = self.encoder_input(x)

        # Convert to NHWC for ConvNeXt blocks: [B, C, 1, T] -> [B, 1, T, C]
        x = x.permute(0, 2, 3, 1)  # [B, 1, T, C]

        # ConvNeXt blocks (PRODUCTION: 12 blocks enabled, NHWC)
        for block in self.encoder_blocks:
            x = block(x)

        # Convert to NCHW for encoder_output: [B, 1, T, C] -> [B, C, 1, T]
        x = x.permute(0, 3, 1, 2)  # [B, C, 1, T]

        # Output (NCHW)
        x = self.encoder_output(x)

        # Convert back to NHWC for decoder: [B, C, 1, T] -> [B, 1, T, C]
        x = x.permute(0, 2, 3, 1)  # [B, 1, T, C]

        return x

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decoder forward pass.
        Production: processes 512-channel VQ features, applies 12 ConvNeXt blocks
        Input x: [batch, 1, time_steps, 1024] (encoder output, NHWC)
        """
        # Convert to NCHW for decoder_input: [B, 1, T, C] -> [B, C, 1, T]
        x = x.permute(0, 3, 1, 2)  # [B, C, 1, T]

        # Input processing (1024 -> 512 -> hidden_dim, NCHW)
        x = self.decoder_input(x)

        # Convert to NHWC for ConvNeXt blocks: [B, C, 1, T] -> [B, 1, T, C]
        x = x.permute(0, 2, 3, 1)  # [B, 1, T, C]

        # ConvNeXt blocks (PRODUCTION: 12 blocks enabled, NHWC)
        for block in self.decoder_blocks:
            x = block(x)

        # Convert to NCHW for decoder output: [B, 1, T, C] -> [B, C, 1, T]
        x = x.permute(0, 3, 1, 2)  # [B, C, 1, T]

        # Decoder projection: hidden_dim -> 512 channels
        x = self.decoder_proj(x)

        # Output (512 -> num_mel_bins, NCHW)
        x = self.decoder_output(x)

        # Convert back to NHWC for return: [B, C, 1, T] -> [B, 1, T, C]
        x = x.permute(0, 2, 3, 1)  # [B, 1, T, C]

        return x


class ConvNeXtBlock2D(nn.Module):
    """
    Simplified ConvNeXt block for 2D convolutions.
    Adapted for mel spectrogram processing in NHWC format: [batch, 1, time_steps, channels]
    """

    def __init__(self, dim: int):
        super().__init__()
        # Depthwise 2D conv: groups=dim for depthwise
        self.dwconv = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, 1, time_steps, dim] (NHWC format)

        Returns:
            torch.Tensor: [batch_size, 1, time_steps, dim] (NHWC format)
        """
        residual = x

        # Depthwise conv (NHWC -> NHWC, but Conv2d expects NCHW)
        # Convert to NCHW for Conv2d: [B, 1, T, C] -> [B, C, 1, T]
        x_nchw = x.permute(0, 3, 1, 2)  # [B, C, 1, T]
        x_nchw = self.dwconv(x_nchw)  # [B, C, 1, T]

        # Convert back to NHWC for LayerNorm: [B, C, 1, T] -> [B, 1, T, C]
        x = x_nchw.permute(0, 2, 3, 1)  # [B, 1, T, C]

        # Permute for LayerNorm: [B, 1, T, C] -> [B, T, C]
        x_for_norm = x.squeeze(1)  # [B, T, C]
        x_for_norm = self.norm(x_for_norm)
        x_for_norm = self.pwconv1(x_for_norm)
        x_for_norm = self.act(x_for_norm)
        x_for_norm = self.pwconv2(x_for_norm)

        # Reshape back: [B, T, C] -> [B, 1, T, C]
        x = x_for_norm.unsqueeze(1)  # [B, 1, T, C]

        # Residual
        x = x + residual

        return x
