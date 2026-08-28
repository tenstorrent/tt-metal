# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `down_encoder_block2_d`.

Reference module: `encoder.down_blocks.0` of `AutoencoderKLFlux2` — a diffusers
`DownEncoderBlock2D`, two 128-channel resnets at 224x224 followed by a stride-2
downsampler, so `[1, 128, 224, 224] -> [1, 128, 112, 112]`:

    for resnet in resnets: hidden_states = resnet(hidden_states)
    for downsampler in downsamplers: hidden_states = downsampler(hidden_states)

The downsampler is a `Downsample2D` with `padding=0`, which in diffusers means
the activation is padded by hand with `F.pad(..., (0, 1, 0, 1))` — bottom and
right only — before a padding-free stride-2 3x3 conv. `ttnn.conv2d` takes
`(pad_top, pad_bottom, pad_left, pad_right)` directly, so that asymmetric window
is expressed in the conv itself rather than as a separate pad op.

Blocks come from `_vae_blocks.py`, shared with the other VAE components of this
pipeline; see that module's docstring for the layout convention and the
tensor-parallel derivation. In short: every conv is COLUMN-parallel over its
output channels (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias
so each bias stays with its own columns) followed by an `all_gather` on the
channel dim — concatenating disjoint output channels is the identity, so the
gathered activation is exactly the single-device one. GroupNorm gamma/beta stay
REPLICATED, since the activation between convs is always full-width.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks


class TtDownEncoderBlock2D:
    """Tensor-parallel native-ttnn `DownEncoderBlock2D`."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("down_encoder_block2_d stub needs the torch reference module to stage its weights")
        self.device = device
        self.tp = _vae_blocks.mesh_width(device)
        self.block = _vae_blocks.DownEncoderBlock2D(device, torch_module, self.tp)

    def __call__(self, hidden_states, *args, **kwargs):
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(hidden_states)
        x, height, width = self.block(x, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.block.out_channels, height, width)


def build(device, torch_module=None):
    return TtDownEncoderBlock2D(device, torch_module)


def down_encoder_block2_d(device, torch_module=None):
    return TtDownEncoderBlock2D(device, torch_module)
