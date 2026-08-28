# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `decoder` (the FLUX.2 VAE `Decoder`).

Reference module: `decoder` of `AutoencoderKLFlux2` — a diffusers `Decoder`
that lifts a `[1, 32, 28, 28]` latent to a `[1, 3, 224, 224]` image:

    sample = conv_in(sample)                     # Conv2d(32 -> 512, 3x3)
    sample = mid_block(sample)                   # resnet -> attention -> resnet, 512ch
    for up_block in up_blocks: sample = up_block(sample)
    sample = conv_out(silu(conv_norm_out(sample)))

with `up_blocks` widths `512, 512, 256, 128` and a nearest-2x upsampler on the
first three, so the spatial size walks 28 -> 56 -> 112 -> 224.

The blocks live in `_vae_blocks.py`, shared with the other VAE components of
this pipeline; that module's docstring carries the layout convention, the
tensor-parallel derivation and the GroupNorm reasoning. In short:

  * every conv is COLUMN-parallel over its output channels
    (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias, so a bias
    stays with its own columns) followed by an `all_gather` on the channel dim.
    Concatenating disjoint output channels is the identity, so the gathered
    activation is exactly the single-device one;
  * `conv_out` has 3 output channels and cannot be split 8 ways, so it — alone
    — stays replicated. It is also the cheapest conv in the decoder;
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


class TtDecoder:
    """Tensor-parallel native-ttnn `Decoder` for the FLUX.2 VAE."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("decoder stub needs the torch reference module to stage its weights")

        self.device = device
        self.tp = _vae_blocks.mesh_width(device)

        self.conv_in = _vae_blocks.Conv2d(device, torch_module.conv_in, self.tp)
        self.mid_block = _vae_blocks.UNetMidBlock2D(
            device,
            torch_module.mid_block,
            self.tp,
            attention_factory=TtVaeAttention,
        )
        self.up_blocks = [_vae_blocks.UpDecoderBlock2D(device, block, self.tp) for block in torch_module.up_blocks]
        self.conv_norm_out = _vae_blocks.GroupNorm(device, torch_module.conv_norm_out, self.tp)
        self.conv_out = _vae_blocks.Conv2d(device, torch_module.conv_out, self.tp)

    def __call__(self, sample, *args, **kwargs):
        if kwargs.get("latent_embeds") is not None or (args and args[0] is not None):
            raise RuntimeError("this VAE decoder has no `latent_embeds` conditioning path")

        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(sample)

        x, height, width = self.conv_in(x, batch, height, width)
        x, height, width = self.mid_block(x, batch, height, width)
        for up_block in self.up_blocks:
            x, height, width = up_block(x, batch, height, width)

        x = self.conv_norm_out(x, batch, height * width)
        x = ttnn.silu(x)
        x, height, width = self.conv_out(x, batch, height, width)

        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.conv_out.out_channels_full, height, width)


def build(device, torch_module=None):
    return TtDecoder(device, torch_module)


def decoder(device, torch_module=None):
    return TtDecoder(device, torch_module)
