"""MiniMaxMusic3Vocoder (DAC-style Flow-VAE decoder) as a plain torch module. Keys (weight_g/weight_v) match
vocoder/*.safetensors, so the classic `torch.nn.utils.weight_norm` parametrization is used."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from models.autoports.minimaxai_minimax_music3.config import VocoderConfig

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.nn.utils import weight_norm


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x):
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            )
        )
        self.res_unit1 = ResidualUnit(output_dim, dilation=1)
        self.res_unit2 = ResidualUnit(output_dim, dilation=3)
        self.res_unit3 = ResidualUnit(output_dim, dilation=9)

    def forward(self, x):
        x = self.conv_t1(self.snake1(x))
        return self.res_unit3(self.res_unit2(self.res_unit1(x)))


class Music3Vocoder(nn.Module):
    def __init__(self, cfg: VocoderConfig = VocoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.dec_in_proj = nn.Conv1d(cfg.latent_channels // 2, cfg.decoder_input_dim, kernel_size=1)
        self.conv_in = weight_norm(nn.Conv1d(cfg.decoder_input_dim, cfg.decoder_hidden_dim, kernel_size=7, padding=3))
        blocks, output_dim = [], cfg.decoder_hidden_dim
        for i, stride in enumerate(cfg.upsampling_ratios):
            input_dim = cfg.decoder_hidden_dim // (2**i)
            output_dim = cfg.decoder_hidden_dim // (2 ** (i + 1))
            blocks.append(DecoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = Snake1d(output_dim)
        self.conv_out = weight_norm(nn.Conv1d(output_dim, 1, kernel_size=7, padding=3))

    @property
    def hop_length(self):
        return math.prod(self.cfg.upsampling_ratios)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, 128, L] -> [B, 2, samples] in [-1, 1] (two folded 64-channel streams = stereo)."""
        b, _, length = latents.shape
        x = latents.reshape(b * 2, self.cfg.latent_channels // 2, length)
        x = self.conv_in(self.dec_in_proj(x))
        for block in self.blocks:
            x = block(x)
        wav = torch.tanh(self.conv_out(self.snake_out(x)))
        return wav.reshape(b, 2, -1)

    @staticmethod
    def load(snapshot, dtype=torch.float32, device="cpu", cfg: VocoderConfig | None = None) -> "Music3Vocoder":
        from safetensors.torch import load_file

        m = Music3Vocoder(cfg or VocoderConfig())
        sd = load_file(str(Path(snapshot, "vocoder", "diffusion_pytorch_model.safetensors")), device="cpu")
        m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
        return m.to(device).eval()
