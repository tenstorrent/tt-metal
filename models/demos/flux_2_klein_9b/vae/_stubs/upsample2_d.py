# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `upsample2_d`.

Reference module: `decoder.up_blocks.0.upsamplers.0` of `AutoencoderKLFlux2` — a
diffusers `Upsample2D` doubling the feature map, `[1, 512, 28, 28] -> [1, 512, 56, 56]`.

It is configured `use_conv_transpose=False, interpolate=True, use_conv=True`, with
no norm, so the forward is:

    hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
    hidden_states = self.conv(hidden_states)             # 3x3, stride 1, padding 1

`ttnn.upsample` is the nearest-neighbour equivalent, but unlike the convs it wants
an unpadded `[N, H, W, C]` activation in `ROW_MAJOR_LAYOUT`, so
`_vae_blocks._upsample2x` steps out of this pipeline's flat-NHWC convention for
that one op and returns to it immediately. `output_size` is never passed by this
model (the up blocks always double), and a non-None one is rejected rather than
silently upsampled by 2 anyway.

Blocks come from `_vae_blocks.py`, shared with the other VAE components of this
pipeline; see that module's docstring for the layout convention.

Tensor parallelism (TP=8)
-------------------------
The only weights here are the conv's, and it is COLUMN-parallel over its OUTPUT
channels (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias so each
bias stays with its own columns) followed by an `all_gather` on the channel dim:
at TP=8 each device owns 64 of the 512 output channels, and concatenating disjoint
output channels is the identity, so the gathered output is exactly the
single-device one.

The interpolation itself needs no collective and no split: nearest-neighbour
upsampling is a per-channel spatial copy, so it is the same operation on the
replicated full-width activation on every device — and running it BEFORE the conv
(the order the reference uses) means the conv still receives all of its input
channels, which is what column-parallel requires.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks


class TtUpsample2D:
    """Tensor-parallel native-ttnn `Upsample2D`."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("upsample2_d stub needs the torch reference module to stage its weights")
        self.device = device
        self.tp = _vae_blocks.mesh_width(device)
        self.upsample = _vae_blocks.Upsample2D(device, torch_module, self.tp)

    def __call__(self, hidden_states, output_size=None, *args, **kwargs):
        if output_size is not None:
            raise RuntimeError(
                f"upsample2_d implements the scale-factor-2 nearest upsample this VAE "
                f"uses; an explicit output_size={output_size} was requested"
            )
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(hidden_states)
        x, height, width = self.upsample(x, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.upsample.out_channels, height, width)


def build(device, torch_module=None):
    return TtUpsample2D(device, torch_module)


def upsample2_d(device, torch_module=None):
    return TtUpsample2D(device, torch_module)
