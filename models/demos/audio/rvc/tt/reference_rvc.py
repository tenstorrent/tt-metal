# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference model for RVC (Retrieval-based Voice Conversion).

This module provides a PyTorch implementation of RVC components used for:
1. Generating reference outputs for validation
2. Weight initialization for TTNN bring-up
3. Parameter preprocessing

Based on: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, c=None, mask=None):
        B, C, T = x.shape
        q = self.conv_q(x).reshape(B, self.n_heads, self.head_dim, T)
        k = self.conv_k(x if c is None else c).reshape(B, self.n_heads, self.head_dim, T)
        v = self.conv_v(x if c is None else c).reshape(B, self.n_heads, self.head_dim, T)

        attn = torch.einsum("bhct,bhcs->bhts", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.einsum("bhts,bhcs->bhct", attn, v)
        out = out.reshape(B, C, T)
        out = self.conv_o(out)
        return out


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, channels, filter_channels, n_heads, kernel_size):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = MultiHeadAttention(channels, channels, n_heads)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels, channels, filter_channels, kernel_size)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = self.drop(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop(x)
        x = residual + x
        return x


class PosteriorEncoder(nn.Module):
    """VITS posterior encoder with WaveNet-like convolutions."""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=5, n_layers=6):
        super().__init__()
        self.pre_conv = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc_layers = nn.ModuleList([
            EncoderBlock(hidden_channels, hidden_channels * 4, n_heads=2, kernel_size=3)
            for _ in range(n_layers)
        ])
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_logs = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, g=None):
        x = self.pre_conv(x)
        for layer in self.enc_layers:
            x = layer(x)
        m = self.proj_m(x)
        logs = self.proj_logs(x)
        z = m + torch.randn_like(m) * torch.exp(logs) * 0.1
        return z, m, logs


class ResBlock(nn.Module):
    """Multi-receptive field fusion residual block for HiFi-GAN."""

    def __init__(self, channels, kernel_sizes=[3, 7, 11], dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.convs = nn.ModuleList()
        for k, dilations in zip(kernel_sizes, dilation_sizes):
            blocks = nn.ModuleList()
            for d in dilations:
                blocks.append(nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, k, dilation=d, padding=(k * d - d) // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, k, dilation=d, padding=(k * d - d) // 2),
                ))
            self.convs.append(blocks)

    def forward(self, x):
        for blocks in self.convs:
            residual = x
            for block in blocks:
                x = block(x)
            x = x + residual
        return x


class HiFiGANVocoder(nn.Module):
    """HiFi-GAN vocoder for mel → waveform generation."""

    def __init__(
        self,
        in_channels=80,
        upsample_initial_channel=512,
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // (2 ** i)
            self.ups.append(nn.ConvTranspose1d(
                max(ch, upsample_initial_channel // (2 ** (len(upsample_rates) - 1))),
                max(ch // 2, upsample_initial_channel // (2 ** len(upsample_rates))),
                k, stride=u, padding=(k - u) // 2,
            ))

        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(ResBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes))

        self.conv_post = nn.Conv1d(
            max(upsample_initial_channel // (2 ** len(upsample_rates)), 32),
            1, 7, padding=3,
        )

    def forward(self, mel):
        x = self.conv_pre(mel)
        for up, resblock in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = resblock(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        audio = torch.tanh(x)
        return audio


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""

    def __init__(self, channels, hidden_channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels // 2, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.conv3 = nn.Conv1d(hidden_channels, channels, 1)
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x, reverse=False):
        C = x.shape[1]
        x1, x2 = x[:, :C // 2], x[:, C // 2:]

        h = F.relu(self.conv1(x1))
        h = F.relu(self.conv2(h))
        params = self.conv3(h)
        log_s = params[:, :C // 2]
        t = params[:, C // 2:]

        if not reverse:
            x2 = x2 * torch.exp(log_s) + t
        else:
            x2 = (x2 - t) * torch.exp(-log_s)

        return torch.cat([x1, x2], dim=1)


class FlowDecoder(nn.Module):
    """Flow-based decoder with affine coupling layers."""

    def __init__(self, channels, hidden_channels, n_flows=4, kernel_size=5):
        super().__init__()
        self.flows = nn.ModuleList([
            AffineCouplingLayer(channels, hidden_channels, kernel_size)
            for _ in range(n_flows)
        ])

    def forward(self, z, reverse=False):
        for flow in (reversed(self.flows) if reverse else self.flows):
            z = flow(z, reverse=reverse)
            z = z.flip(1)
        return z


class RMVPE(nn.Module):
    """Simplified RMVPE pitch extraction network."""

    def __init__(self, in_channels=128, hidden_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.f0_head = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, mel):
        x = F.relu(self.conv1(mel))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        f0 = torch.sigmoid(self.f0_head(x)) * 1000.0
        return f0.squeeze(1)


class RVCModel(nn.Module):
    """
    Full RVC (Retrieval-based Voice Conversion) model.

    Combines posterior encoder, pitch extraction, feature retrieval,
    flow-based decoder, and HiFi-GAN vocoder for voice conversion.
    """

    def __init__(
        self,
        n_mels=80,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        n_flows=4,
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.encoder = PosteriorEncoder(
            n_mels, inter_channels, hidden_channels, kernel_size=5, n_layers=n_layers
        )
        self.flow = FlowDecoder(inter_channels, hidden_channels, n_flows=n_flows)
        self.vocoder = HiFiGANVocoder(
            in_channels=inter_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )
        self.rmvpe = RMVPE(in_channels=n_mels, hidden_channels=256)

    def forward(
        self,
        source_mel,
        target_features_index=None,
        f0_up_key=0,
        index_rate=0.5,
    ):
        """
        Run voice conversion.

        Args:
            source_mel: Source mel spectrogram [B, n_mels, T]
            target_features_index: Target speaker feature index [N, C]
            f0_up_key: Pitch transposition semitones
            index_rate: Feature retrieval blending ratio

        Returns:
            Converted audio waveform [B, 1, T_audio]
        """
        # 1. Posterior encode
        z, m, logs = self.encoder(source_mel)

        # 2. Feature retrieval (if index provided)
        if index_rate > 0 and target_features_index is not None:
            B, C, T = z.shape
            z_norm = F.normalize(z.reshape(B, C, -1).permute(0, 2, 1), dim=-1)
            idx_norm = F.normalize(target_features_index, dim=-1)
            sim = torch.matmul(z_norm, idx_norm.T)
            _, top_idx = sim.max(dim=-1)
            retrieved = target_features_index[top_idx].permute(0, 2, 1)
            z = (1 - index_rate) * z + index_rate * retrieved

        # 3. Flow decode
        z = self.flow(z, reverse=True)

        # 4. Vocode
        audio = self.vocoder(z)

        return audio
