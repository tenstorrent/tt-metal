# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `downsample2_d`.

Reference module: `encoder.down_blocks.0.downsamplers.0` of
`AutoencoderKLFlux2` — a diffusers `Downsample2D` halving the feature map,
`[1, 128, 224, 224] -> [1, 128, 112, 112]`.

Its `padding` is 0, which in diffusers means the activation is padded by hand:

    hidden_states = F.pad(hidden_states, (0, 1, 0, 1))   # bottom and right only
    hidden_states = self.conv(hidden_states)             # 3x3, stride 2, no padding

`F.pad`'s `(left, right, top, bottom)` order over the trailing two NCHW dims is
`W += (0, 1)`, `H += (0, 1)`; `ttnn.conv2d` accepts
`(pad_top, pad_bottom, pad_left, pad_right)`, so the same asymmetric window is
`(0, 1, 0, 1)` folded straight into the conv — no separate pad op, and
`(224 + 1 - 3) / 2 + 1 = 112` as the golden requires.

Blocks come from `_vae_blocks.py`, shared with the other VAE components of this
pipeline; see that module's docstring for the layout convention and the
tensor-parallel derivation. The conv here is COLUMN-parallel over its output
channels (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias so the
bias stays with its own columns) followed by an `all_gather` on the channel dim.
Concatenating disjoint output channels is the identity, so the gathered output is
exactly the single-device one.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks


class TtDownsample2D:
    """Tensor-parallel native-ttnn `Downsample2D`."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("downsample2_d stub needs the torch reference module to stage its weights")
        self.device = device
        self.tp = _vae_blocks.mesh_width(device)
        self.downsample = _vae_blocks.Downsample2D(device, torch_module, self.tp)

    def __call__(self, hidden_states, *args, **kwargs):
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(hidden_states)
        x, height, width = self.downsample(x, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.downsample.out_channels, height, width)


def build(device, torch_module=None):
    return TtDownsample2D(device, torch_module)


def downsample2_d(device, torch_module=None):
    return TtDownsample2D(device, torch_module)
