# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 image patch projection modules:
#   UNetDown  (patch_embed)  -- VAE latent  -> transformer token sequence
#   UNetUp    (final_layer)  -- transformer tokens -> VAE latent / noise pred
#
# Extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     conv_nd / avg_pool_nd / zero_module / normalization  (lines 180-229)
#     Upsample        (lines 596-626)
#     Downsample      (lines 629-657)
#     ResBlock        (lines 660-756)
#     UNetDown        (lines 759-813)
#     UNetUp          (lines 816-883)
#
# Used as the golden reference for TT-Metal numeric validation.
#
# Model instantiation (HunyuanImage3ForCausalMM.__init__, lines 1736-1758, with
# the released config.json):
#   patch_size            = config.patch_size           = 1
#   emb_channels          = config.hidden_size          = 4096
#   in_channels  (down)   = config.vae.latent_channels  = 32
#   hidden_channels       = config.patch_embed_hidden_dim = 1024
#   out_channels (down)   = config.hidden_size          = 4096
#   final_layer is UNetUp with in=4096, hidden=1024, out=32, out_norm=True
#
# With patch_size == 1 there is NO up/down sampling: every ResBlock takes the
# simple (updown=False) path, so the spatial resolution of the latent is the
# transformer token grid (token_h, token_w == latent H, W). The Upsample /
# Downsample classes are included for completeness / faithfulness to upstream
# but are not exercised at patch_size==1.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers (verbatim from upstream)
# ---------------------------------------------------------------------------
def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, **kwargs):
    """Make a standard normalization layer (GroupNorm with 32 groups)."""
    return nn.GroupNorm(32, channels, **kwargs)


# ---------------------------------------------------------------------------
# Upsample / Downsample (verbatim) -- not used at patch_size==1
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    """An upsampling layer with an optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1, **factory_kwargs)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1, **factory_kwargs)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ---------------------------------------------------------------------------
# ResBlock (verbatim)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    Timestep conditioning is applied as adaptive group-norm: the embedding MLP
    produces (scale, shift) which modulate the group-normalised features:
        h = out_norm(h) * (1 + scale) + shift
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels, **factory_kwargs),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs)
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
# UNetDown (patch_embed) -- verbatim
# ---------------------------------------------------------------------------
class UNetDown(nn.Module):
    """
    patch_size: one of [1, 2, 4, 8]
    in_channels: vae latent dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: transformer model dim
    """

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [
                conv_nd(
                    2, in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1, **factory_kwargs
                )
            ]
        )

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels if (i + 1) * 2 != self.patch_size else out_channels,
                        dropout=dropout,
                        down=True,
                        **factory_kwargs,
                    )
                )

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, token_h, token_w


# ---------------------------------------------------------------------------
# UNetUp (final_layer) -- verbatim
# ---------------------------------------------------------------------------
class UNetUp(nn.Module):
    """
    patch_size: one of [1, 2, 4, 8]
    in_channels: transformer model dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: vae latent dim
    """

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
        out_norm=False,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if i == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        up=True,
                        **factory_kwargs,
                    )
                )

        if out_norm:
            self.model.append(
                nn.Sequential(
                    normalization(hidden_channels, **factory_kwargs),
                    nn.SiLU(),
                    conv_nd(
                        2,
                        in_channels=hidden_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        **factory_kwargs,
                    ),
                )
            )
        else:
            self.model.append(
                conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            )

    # batch_size, seq_len, model_dim
    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.experimental.hunyuan_image_3_0.ref.model_config import (
        HIDDEN_SIZE,
        PATCH_EMBED_HIDDEN_CHANNELS,
        VAE_LATENT_CHANNELS,
    )

    torch.manual_seed(0)
    PATCH, LATENT, HID, HSZ = 1, VAE_LATENT_CHANNELS, PATCH_EMBED_HIDDEN_CHANNELS, HIDDEN_SIZE
    H = W = 8
    down = UNetDown(PATCH, LATENT, HSZ, HID, HSZ).eval()
    up = UNetUp(PATCH, HSZ, HSZ, HID, LATENT, out_norm=True).eval()

    x = torch.randn(1, LATENT, H, W)
    t = torch.randn(1, HSZ)  # time embedding (already TimestepEmbedder output)
    with torch.no_grad():
        seq, th, tw = down(x, t)
        rec = up(seq, t, th, tw)
    print(f"down: {tuple(x.shape)} -> seq {tuple(seq.shape)} (token_h={th}, token_w={tw})")
    print(f"up:   seq -> {tuple(rec.shape)}  (should match input latent)")
