# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `up_decoder_block2_d`.

Reference module: `decoder.up_blocks.0` of `AutoencoderKLFlux2` — a diffusers
`UpDecoderBlock2D`, 512-channel resnets at 28x28 followed by a 2x upsampler, so
`[1, 512, 28, 28] -> [1, 512, 56, 56]`:

    for resnet in resnets: hidden_states = resnet(hidden_states)
    for upsampler in upsamplers: hidden_states = upsampler(hidden_states)

The upsampler is an `Upsample2D` with `interpolate=True, use_conv=True`: nearest
2x interpolation and then a padding-1 3x3 conv. `ttnn.upsample` wants unpadded
`[N, H, W, C]` in `ROW_MAJOR_LAYOUT`, so `_vae_blocks._upsample2x` leaves the
flat-NHWC convention for that one op and comes straight back to it. There is no
time embedding (`temb=None`) and no `conv_shortcut` — `in_channels == out_channels`
here, so each resnet's residual is its input.

Tensor parallelism (TP=8)
-------------------------
The parallel axis for a VAE is the CHANNEL axis, and every conv in this block —
the resnets' `conv1`/`conv2` and the upsampler's conv — is COLUMN-parallel over
its OUTPUT channels:

    W [C_out, C_in, kh, kw]  --ShardTensorToMesh(dim=0)-->  [C_out/TP, C_in, kh, kw]
    b [1, 1, 1, C_out]       --ShardTensorToMesh(dim=3)-->  [1, 1, 1, C_out/TP]

so at TP=8 each device owns 64 of the 512 output channels — bias included, since
the bias travels with its own columns — and one `all_gather` on the channel dim
concatenates the disjoint shards back. Concatenating disjoint output channels is
the identity, so the gathered activation is exactly the single-device one.

Column-parallel rather than a Megatron column-then-row pairing because a conv
needs ALL of its input channels; row-parallelising `conv2` would hand it 64 input
channels per device and, worse, leave both GroupNorms and the residual add looking
at a channel sliver. Keeping the activation full-width between convs is what lets
each GroupNorm see all 32 groups, so no group straddles a chip and gamma/beta stay
REPLICATED, as the TP principles ask for elementwise parameters. It also means the
nearest-neighbour upsample — which is a pure per-channel spatial copy — needs no
collective of its own: it runs on the already-gathered full-width activation.

See `_vae_blocks.py` for the shared derivation; the graduated `decoder` builds its
own up blocks from exactly these pieces.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks


class TtUpDecoderBlock2D:
    """Tensor-parallel native-ttnn `UpDecoderBlock2D`."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("up_decoder_block2_d stub needs the torch reference module to stage its weights")
        self.device = device
        self.tp = _vae_blocks.mesh_width(device)
        self.block = _vae_blocks.UpDecoderBlock2D(device, torch_module, self.tp)

    def __call__(self, hidden_states, temb=None, *args, **kwargs):
        if temb is not None:
            raise RuntimeError("the VAE up block has no time embedding; got a non-None `temb`")
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(hidden_states)
        x, height, width = self.block(x, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.block.out_channels, height, width)


def build(device, torch_module=None):
    return TtUpDecoderBlock2D(device, torch_module)


def up_decoder_block2_d(device, torch_module=None):
    return TtUpDecoderBlock2D(device, torch_module)
