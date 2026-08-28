# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `encoder` (the FLUX.2 VAE `Encoder`).

Reference module: `encoder` of `AutoencoderKLFlux2` — a diffusers `Encoder`
compressing `[1, 3, 224, 224]` into `[1, 64, 28, 28]` (32 latent channels plus
32 for the posterior log-variance):

    sample = conv_in(sample)                     # Conv2d(3 -> 128, 3x3)
    for down_block in down_blocks: sample = down_block(sample)
    sample = mid_block(sample)                   # resnet -> attention -> resnet, 512ch
    sample = conv_out(silu(conv_norm_out(sample)))

`down_blocks` widths are `128, 256, 512, 512` with a stride-2 downsampler on the
first three, so the spatial size walks 224 -> 112 -> 56 -> 28.

The blocks live in `_vae_blocks.py`, shared with the other VAE components of
this pipeline; that module's docstring carries the layout convention, the
tensor-parallel derivation and the GroupNorm reasoning. In short:

  * every conv is COLUMN-parallel over its output channels
    (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias, so a bias
    stays with its own columns) followed by an `all_gather` on the channel dim.
    Concatenating disjoint output channels is the identity, so the gathered
    activation is exactly the single-device one;
  * GroupNorm gamma/beta stay REPLICATED and see the full channel dim, because
    the activation between convs is always full-width;
  * the mid-block attention reuses the `attention` component's own
    tensor-parallel stub, which splits its qkv channel axis and all_reduces
    after the row-parallel out projection.
"""
from __future__ import annotations

import ttnn
from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks
from models.demos.flux_2_klein_9b.vae._stubs.attention import TtVaeAttention


class TtEncoder:
    """Tensor-parallel native-ttnn `Encoder` for the FLUX.2 VAE."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("encoder stub needs the torch reference module to stage its weights")

        self.device = device
        self.tp = _vae_blocks.mesh_width(device)

        self.conv_in = _vae_blocks.Conv2d(device, torch_module.conv_in, self.tp)
        self.down_blocks = [
            _vae_blocks.DownEncoderBlock2D(device, block, self.tp) for block in torch_module.down_blocks
        ]
        self.mid_block = _vae_blocks.UNetMidBlock2D(
            device,
            torch_module.mid_block,
            self.tp,
            attention_factory=TtVaeAttention,
        )
        self.conv_norm_out = _vae_blocks.GroupNorm(device, torch_module.conv_norm_out, self.tp)
        self.conv_out = _vae_blocks.Conv2d(device, torch_module.conv_out, self.tp)

    def __call__(self, sample, *args, **kwargs):
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(sample)

        x, height, width = self.conv_in(x, batch, height, width)
        for down_block in self.down_blocks:
            x, height, width = down_block(x, batch, height, width)
        x, height, width = self.mid_block(x, batch, height, width)

        x = self.conv_norm_out(x, batch, height * width)
        x = ttnn.silu(x)
        x, height, width = self.conv_out(x, batch, height, width)

        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.conv_out.out_channels_full, height, width)


def build(device, torch_module=None):
    return TtEncoder(device, torch_module)


def encoder(device, torch_module=None):
    return TtEncoder(device, torch_module)
