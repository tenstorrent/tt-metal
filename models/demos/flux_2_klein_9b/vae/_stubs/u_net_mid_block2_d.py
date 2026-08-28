# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `u_net_mid_block2_d`.

Reference module: `encoder.mid_block` of `AutoencoderKLFlux2` — a diffusers
`UNetMidBlock2D` at the VAE's bottleneck, 512 channels at 28x28, so
`[1, 512, 28, 28] -> [1, 512, 28, 28]`:

    hidden_states = resnets[0](hidden_states)
    for attn, resnet in zip(attentions, resnets[1:]):
        if attn is not None: hidden_states = attn(hidden_states)
        hidden_states = resnet(hidden_states)

Here that is resnet -> attention -> resnet (`mid_block_add_attention: true` gives
one attention between the block's two resnets). Both resnets are the VAE flavour
— no time embedding, no resampling, `in_channels == out_channels == 512`, so the
residual is the input itself — and the attention is the single-head spatial
self-attention the `attention` component brought up.

Tensor parallelism (TP=8)
-------------------------
The block owns no weights of its own; it composes two schemes, each derived where
its weights live, and each leaving the activation FULL-WIDTH at the seam between
stages — which is what makes them composable at all:

  * The resnets' convs are COLUMN-parallel over their output channels
    (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias so each bias
    stays with its own columns) followed by an `all_gather` on the channel dim.
    Concatenating disjoint output channels is the identity, so the math is
    unchanged. Column- rather than row-parallel because a conv needs ALL of its
    input channels.
  * The attention splits its qkv on the CHANNEL axis rather than the head axis —
    it has a single head, so there is no head axis to split — with an
    `all_gather` on q/k before the score matmul, `v` left sharded, and a
    ROW-parallel `to_out` closed by an `all_reduce` with its bias replicated and
    applied after the reduction.
  * Every GroupNorm (two per resnet, one in the attention) therefore sees all 32
    groups on every device, so no group straddles a chip and gamma/beta stay
    REPLICATED, as the TP principles ask for elementwise parameters.

See `_vae_blocks.py` for the conv/GroupNorm derivation and `_stubs/attention.py`
for the attention derivation. Both are shared with the already-graduated
`encoder` and `decoder`, which build their own mid block from exactly these
pieces — this component is that same composition, PCC-gated on its own.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks
from models.demos.flux_2_klein_9b.vae._stubs.attention import TtVaeAttention


class TtUNetMidBlock2D:
    """Tensor-parallel native-ttnn `UNetMidBlock2D`."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("u_net_mid_block2_d stub needs the torch reference module to stage its weights")
        self.device = device
        self.tp = _vae_blocks.mesh_width(device)
        self.block = _vae_blocks.UNetMidBlock2D(
            device,
            torch_module,
            self.tp,
            attention_factory=TtVaeAttention,
        )

    def __call__(self, hidden_states, temb=None, *args, **kwargs):
        if temb is not None:
            raise RuntimeError("the VAE mid block has no time embedding; got a non-None `temb`")
        x, batch, _, height, width = _vae_blocks.nchw_to_flat_nhwc(hidden_states)
        x, height, width = self.block(x, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(x, batch, self.block.out_channels, height, width)


def build(device, torch_module=None):
    return TtUNetMidBlock2D(device, torch_module)


def u_net_mid_block2_d(device, torch_module=None):
    return TtUNetMidBlock2D(device, torch_module)
