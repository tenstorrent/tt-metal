# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `resnet_block2_d`.

Reference module: `encoder.down_blocks.0.resnets.0` of `AutoencoderKLFlux2` — a
diffusers `ResnetBlock2D` in its VAE configuration, 128 channels in and out at
224x224, so `[1, 128, 224, 224] -> [1, 128, 224, 224]`:

    h = conv1(silu(norm1(x)))
    h = conv2(silu(norm2(h)))
    out = (x + h) / output_scale_factor

The VAE flavour of the block has no time embedding (`time_emb_proj is None`, and
the encoder always calls it with `temb=None`), does not resample inside the block,
and here `in_channels == out_channels` so there is no `conv_shortcut` — the
residual is the input itself. `_vae_blocks.ResnetBlock2D` asserts all of that at
construction rather than silently dropping a term.

Tensor parallelism
------------------
The parallel axis for a VAE is the CHANNEL axis, and both convs are
COLUMN-parallel over their OUTPUT channels:

    W [C_out, C_in, kh, kw]  --ShardTensorToMesh(dim=0)-->  [C_out/TP, C_in, kh, kw]
    b [1, 1, 1, C_out]       --ShardTensorToMesh(dim=3)-->  [1, 1, 1, C_out/TP]

so at TP=8 each device owns 16 of the 128 output channels — bias included, since
the bias travels with its own columns — and one `all_gather` on the channel dim
concatenates the disjoint shards back into the full activation. Concatenating
disjoint output channels is the identity, so the math is unchanged and the
gathered output is exactly what a single device produces.

Column-parallel rather than a Megatron column-then-row pairing because a conv
needs ALL of its input channels: row-parallelising `conv2` would hand it 16 input
channels per device and, more importantly, would leave the residual add and both
GroupNorms looking at a channel sliver. Keeping the activation full-width between
convs is what lets `norm1`/`norm2` see all 32 groups, so their gamma/beta stay
REPLICATED — as the TP principles ask for elementwise parameters — and the
residual `x + h` is a plain elementwise add on identical full-width tensors.

The blocks live in `_vae_blocks.py`, shared with the other VAE components of this
pipeline (this same `ResnetBlock2D` is what `down_encoder_block2_d`,
`up_decoder_block2_d` and `u_net_mid_block2_d` compose); see that module's
docstring for the layout convention and the GroupNorm derivation.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks


class TtResnetBlock2D:
    """Tensor-parallel native-ttnn `ResnetBlock2D` (VAE configuration)."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("resnet_block2_d stub needs the torch reference module to stage its weights")
        self.device = device
        self.tp = _vae_blocks.mesh_width(device)
        self.block = _vae_blocks.ResnetBlock2D(device, torch_module, self.tp)

    def __call__(self, hidden_states, temb=None, *args, **kwargs):
        if temb is not None:
            raise RuntimeError("VAE ResnetBlock2D has no time embedding; got a non-None `temb`")
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(hidden_states)
        x, height, width = self.block(x, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.block.out_channels, height, width)


def build(device, torch_module=None):
    return TtResnetBlock2D(device, torch_module)


def resnet_block2_d(device, torch_module=None):
    return TtResnetBlock2D(device, torch_module)
