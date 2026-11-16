# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compact torch reference for the Qwen-Image-2.1 VAE *encoder*, single frame.

`AutoencoderKLQwenImage21` is a 3D (video) VAE. `_encode` splits the clip into chunks of 4 frames
and runs `QwenImage21Encoder3d` once per chunk with a `feat_cache` of one slot per
`QwenImage21CausalConv3d`; a still image is one frame, so there is exactly one chunk and every
temporal mechanism degenerates:

  * `QwenImage21CausalConv3d` squeezes the time axis away and is a plain `nn.Conv2d` with the
    symmetric spatial padding it stashed in `_padding`. With `feat_cache[idx] is None` on the first
    chunk it is called as `conv(x, None)`, i.e. with no temporal context to prepend.
  * `QwenImage21Resample` in `downsample3d` mode takes its "first chunk" branch, which only *stores*
    `x` in the cache slot; `time_conv` is not applied at all. So `downsample3d` is `downsample2d`
    here: `ZeroPad2d((0, 1, 0, 1))` then a 3x3 stride-2 convolution. (The `time_conv` parameters are
    still registered so `load_state_dict` stays strict.)
  * `QwenImage21AvgDown3D`, the `avg_shortcut` of every down block, collapses to a 2x2 average pool
    plus a fixed channel mapping (see `avg_down_fast` and the literal `avg_down_reference` it is
    checked against). For `factor_t == 2` the time axis is zero-padded *in front* to length 2 and
    the channel count doubles, with the pool landing in the odd output channels and the zero frame
    in the even ones.

So the whole encoder is 2D and this module works on `[B, C, H, W]` tensors:

    image [1, 4, 1024, 1024] -> conv_in -> 5 down blocks -> mid -> head -> quant_conv
                             -> [1, 128, 64, 64] = mean | logvar, mode = mean = [1, 64, 64, 64]

Parameter names and shapes match the checkpoint exactly (`encoder.*`, `quant_conv.*`), so
`load_state_dict` takes the safetensors state dict unmodified. The residual block, attention block
and mid block are the decoder's (`reference/torch_vae.py`); they are structurally identical and use
the same key names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.config import VAE, VAEConfig
from .torch_vae import MidBlock, ResidualBlock, RMSNormCF, pcc  # noqa: F401  (pcc re-exported)


@dataclass(frozen=True)
class EncoderConfig:
    """The encoder-side half of the `vae/config.json` that `common.config.VAEConfig` omits.

    `temperal_downsample` is `VAEConfig.temporal_upsample` reversed, which is how the checkpoint
    config relates the two (`temperal_upsample = temperal_downsample[::-1]`).
    """

    base_dim: int = 96
    in_channels: int = 4
    z_dim: int = 64
    dim_mult: Tuple[int, ...] = VAE.dim_mult
    num_res_blocks: int = 2
    temperal_downsample: Tuple[bool, ...] = tuple(reversed(VAE.temporal_upsample))

    @property
    def dims(self):
        """`[96, 96, 192, 384, 768, 768]` -- the down blocks are `zip(dims[:-1], dims[1:])`."""
        return [self.base_dim * u for u in [1] + list(self.dim_mult)]

    @property
    def latent_channels(self) -> int:
        """What `conv_out` / `quant_conv` carry: `mean` concatenated with `logvar`."""
        return 2 * self.z_dim


ENC = EncoderConfig()

# (in_dim, out_dim, factor_t, factor_s) of the five down-block `avg_shortcut`s.
AVG_DOWN_CASES = [(96, 96, 1, 2), (96, 192, 2, 2), (192, 384, 2, 2), (384, 768, 2, 2), (768, 768, 1, 1)]


# -------------------------------------------------------------------------------------- AvgDown3D


def avg_down_reference(
    x5: torch.Tensor, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
) -> torch.Tensor:
    """Literal transcription of `QwenImage21AvgDown3D.forward` (5D `[B, C, T, H, W]` in and out)."""
    factor = factor_t * factor_s * factor_s
    assert in_channels * factor % out_channels == 0
    group_size = in_channels * factor // out_channels

    pad_t = (factor_t - x5.shape[2] % factor_t) % factor_t
    x = F.pad(x5, (0, 0, 0, 0, pad_t, 0))
    b, c, t, h, w = x.shape
    x = x.view(b, c, t // factor_t, factor_t, h // factor_s, factor_s, w // factor_s, factor_s)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(b, c * factor, t // factor_t, h // factor_s, w // factor_s)
    x = x.view(b, out_channels, group_size, t // factor_t, h // factor_s, w // factor_s)
    return x.mean(dim=2)


def avg_down_fast(
    x: torch.Tensor, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
) -> torch.Tensor:
    """`avg_down_reference` for one frame, as a 4D `[B, C, H, W]` pool.

    After the `permute`/`view` the channel axis is indexed by `k = c * factor + a * factor_s**2 +
    i * factor_s + j` (`a` the temporal offset, `i`/`j` the row/column offset inside the 2x2 spatial
    block), and output channel `co = k // group_size` averages the `group_size` consecutive `k`.
    The three combinations the encoder uses:

      * `factor_t == 1`, `factor_s == 1`, `in == out`     -> `group_size = 1`, identity.
      * `factor_t == 1`, `factor_s == 2`, `in == out`     -> `group_size = 4` = the 2x2 block, so
                                                            `co = c` and this is a 2x2 average pool.
      * `factor_t == 2`, `factor_s == 2`, `out == 2 * in` -> `group_size = 4` again, and `a` has
                                                            stride 4 in `k`, so `co = 2 * c + a` and
                                                            every group is *one* `a`. The time axis
                                                            was zero-padded in front, so `a = 0` is
                                                            the zero frame (even output channels are
                                                            exactly 0) and `a = 1` is the real frame
                                                            (odd output channels are the 2x2 pool).
    """
    if factor_s == 1:
        assert factor_t == 1 and in_channels == out_channels, (in_channels, out_channels, factor_t)
        return x

    assert factor_s == 2, factor_s
    pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
    if factor_t == 1:
        assert in_channels == out_channels, (in_channels, out_channels)
        return pooled

    assert factor_t == 2 and out_channels == 2 * in_channels, (in_channels, out_channels, factor_t)
    b, c, h, w = pooled.shape
    out = torch.zeros(b, 2 * c, h, w, dtype=pooled.dtype, device=pooled.device)
    out[:, 1::2] = pooled  # even channels average the zero frame, odd ones the real frame
    return out


# ----------------------------------------------------------------------------------------- layers


class Downsample(nn.Module):
    """`QwenImage21Resample` in downsample mode, single frame: zero-pad bottom/right then 3x3 s2.

    `time_conv` exists in the checkpoint for `downsample3d` but the first (only) chunk merely fills
    the feature cache without applying it, so it is registered purely so `load_state_dict` stays
    strict.
    """

    def __init__(self, dim: int, temporal: bool) -> None:
        super().__init__()
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2))
        if temporal:
            self.time_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class ResidualDownBlock(nn.Module):
    """`QwenImage21ResidualDownBlock`: `num_res_blocks` residual blocks, an optional 3x3 stride-2
    downsampler, and the `AvgDown3D` shortcut of the block *input* added to the result."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temporal: bool, down_flag: bool) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.factor_t = 2 if temporal else 1
        self.factor_s = 2 if down_flag else 1
        blocks, cur = [], in_dim
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(cur, out_dim))
            cur = out_dim
        self.resnets = nn.ModuleList(blocks)
        self.downsampler = Downsample(out_dim, temporal) if down_flag else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + avg_down_fast(skip, self.in_dim, self.out_dim, self.factor_t, self.factor_s)


class Encoder3d(nn.Module):
    def __init__(self, cfg: EncoderConfig = ENC) -> None:
        super().__init__()
        dims = cfg.dims
        self.dims = dims
        self.conv_in = nn.Conv2d(cfg.in_channels, dims[0], 3, padding=1)
        blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            down_flag = i != len(cfg.dim_mult) - 1
            blocks.append(
                ResidualDownBlock(
                    in_dim,
                    out_dim,
                    cfg.num_res_blocks,
                    temporal=cfg.temperal_downsample[i] if down_flag else False,
                    down_flag=down_flag,
                )
            )
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = MidBlock(dims[-1])
        self.norm_out = RMSNormCF(dims[-1], images=False)
        self.conv_out = nn.Conv2d(dims[-1], cfg.latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class QwenImageVAEEncoderTorch(nn.Module):
    """encoder + `quant_conv`, i.e. `AutoencoderKLQwenImage21.encode` for a single frame."""

    LOGVAR_CLAMP = (-30.0, 20.0)  # DiagonalGaussianDistribution clamps before exp

    def __init__(self, cfg: EncoderConfig = ENC) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder3d(cfg)
        self.quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)

    @staticmethod
    def _as_4d(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            assert image.shape[2] == 1, "single-frame encoder"
            image = image[:, :, 0]
        assert image.dim() == 4, image.shape
        return image

    def parameters_of(self, image: torch.Tensor) -> torch.Tensor:
        """`[B, 4, H, W]` (or `[B, 4, 1, H, W]`) -> `[B, 128, H/16, W/16]` = mean | logvar."""
        return self.quant_conv(self.encoder(self._as_4d(image)))

    def encode(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(mean, logvar)`, the two halves of the diagonal-Gaussian posterior."""
        mean, logvar = torch.chunk(self.parameters_of(image), 2, dim=1)
        return mean, torch.clamp(logvar, *self.LOGVAR_CLAMP)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """The posterior *mode*, which is what the pipeline uses as the condition latent."""
        return self.encode(image)[0]


# ------------------------------------------------------------------------------------- weight load


def state_dict_from_ckpt(ckpt, dtype: torch.dtype = torch.float32) -> dict:
    """Pull the `encoder.*` / `quant_conv.*` tensors out of a `LazyCheckpoint`."""
    wanted = ("encoder.", "quant_conv.")
    return {k: ckpt.get(k, dtype) for k in ckpt.keys() if k.startswith(wanted)}


def load_encoder(ckpt=None, dtype: torch.dtype = torch.float32, cfg: EncoderConfig = ENC) -> QwenImageVAEEncoderTorch:
    if ckpt is None:
        from ..common.weights import vae_ckpt

        ckpt = vae_ckpt()
    model = QwenImageVAEEncoderTorch(cfg).to(dtype)
    missing, unexpected = model.load_state_dict(state_dict_from_ckpt(ckpt, dtype), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------------------- goldens


def normalize_latent(mode: torch.Tensor, cfg: VAEConfig = VAE) -> torch.Tensor:
    """`(mode - latents_mean) / latents_std`, the form the DiT consumes. Needs the shipped stats."""
    import json
    import os

    from ..common.config import snapshot_dir

    cfg_json = json.load(open(os.path.join(snapshot_dir(), "vae", "config.json")))
    shape = (1, -1, 1, 1) if mode.dim() == 4 else (1, -1, 1, 1, 1)
    mean = torch.tensor(cfg_json["latents_mean"], dtype=torch.float32).reshape(shape)
    std = torch.tensor(cfg_json["latents_std"], dtype=torch.float32).reshape(shape)
    return (mode.float() - mean) / std


def load_golden(path: Optional[str] = None):
    """Load `goldens/edit/vae_encode.pt` -> dict with `vae_in` `[1, 4, 1, 1024, 1024]`,
    `latent_mode` / `latent_logvar` / `latent_normalized` `[1, 64, 1, 64, 64]`, or None."""
    import os

    from ..common.config import GOLDENS_DIR

    path = path or os.path.join(GOLDENS_DIR, "edit", "vae_encode.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")
