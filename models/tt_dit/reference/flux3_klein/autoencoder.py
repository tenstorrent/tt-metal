# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/autoencoder.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the PyTorch reference for the tt-metal port.

"""Image autoencoder: 8x VAE + 2x2 patchify = 16x; latents are 128-channel tokens with (t, h, w, l) ids."""


import torch
import torch.nn.functional as F
from torch import Tensor, nn

EMBED_DIM = 32
CH = 128
CH_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
DOWNSAMPLING_RATIO = 16
LATENT_CHANNELS = 128


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        b, c, height, width = h.shape
        flat = lambda t: t.reshape(b, c, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        h = F.scaled_dot_product_attention(flat(self.q(h)), flat(self.k(h)), flat(self.v(h)))
        return x + self.proj_out(h.squeeze(1).permute(0, 2, 1).reshape(b, c, height, width))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * EMBED_DIM, 2 * EMBED_DIM, 1)
        self.conv_in = nn.Conv2d(3, CH, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = CH
        for level, mult in enumerate(CH_MULT):
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS):
                down.block.append(ResnetBlock(block_in, CH * mult))
                block_in = CH * mult
            if level != len(CH_MULT) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * EMBED_DIM, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if level != len(CH_MULT) - 1:
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(EMBED_DIM, EMBED_DIM, 1)
        block_in = CH * CH_MULT[-1]
        self.conv_in = nn.Conv2d(EMBED_DIM, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.up = nn.ModuleList()
        for level in reversed(range(len(CH_MULT))):
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS + 1):
                up.block.append(ResnetBlock(block_in, CH * CH_MULT[level]))
                block_in = CH * CH_MULT[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(len(CH_MULT))):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bn_eps = 1e-4
        self.bn = nn.BatchNorm2d(LATENT_CHANNELS, eps=self.bn_eps, affine=False, track_running_stats=True)

    @torch.inference_mode()
    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """image (3, H, W) in [-1, 1] -> tokens (h*w, 128) and ids (h*w, 4) = (0, row, col, 0)."""
        mean, _ = self.encoder(image.unsqueeze(0)).chunk(2, dim=1)
        z = mean.reshape(1, EMBED_DIM, mean.shape[2] // 2, 2, mean.shape[3] // 2, 2).permute(0, 1, 3, 5, 2, 4)
        z = self.bn(z.reshape(1, LATENT_CHANNELS, mean.shape[2] // 2, mean.shape[3] // 2))
        h, w = z.shape[2:]
        return z[0].permute(1, 2, 0).reshape(h * w, LATENT_CHANNELS), latent_ids(h, w, z.device)

    @torch.inference_mode()
    def decode(self, tokens: Tensor, ids: Tensor) -> Tensor:
        """tokens (h*w, 128) in id order -> image (3, H, W) in [-1, 1]."""
        h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
        z = tokens.float().reshape(h, w, LATENT_CHANNELS).permute(2, 0, 1).unsqueeze(0)
        z = z * torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps) + self.bn.running_mean.view(1, -1, 1, 1)
        z = z.reshape(1, EMBED_DIM, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, EMBED_DIM, 2 * h, 2 * w)
        return self.decoder(z)[0]


def latent_ids(h: int, w: int, device: torch.device | str = "cpu") -> Tensor:
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)


def latent_shape(height: int, width: int) -> tuple[int, int]:
    if height % DOWNSAMPLING_RATIO or width % DOWNSAMPLING_RATIO:
        raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
