# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2SingleTransformerBlock`
(`single_transformer_blocks.0`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    shift, scale, gate = Flux2Modulation.split(temb_mod, 1)[0]
    h                  = norm(hidden_states)              # LayerNorm, no affine
    h                  = (1 + scale) * h + shift
    hidden_states      = hidden_states + gate * attn(h)   # parallel attn + MLP

One of the 24 single-stream blocks: the image and text tokens are already
concatenated, and the attention and feed-forward run in PARALLEL off the same
normalised input (ViT-22B style), fused into `attn`'s single input and single
output projection.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
All of the block's parameters live in `attn`, so the scheme is that layer's --
`to_qkv_mlp_proj` COLUMN-parallel over its five fused groups (q, k, v and
SwiGLU's two halves), `to_out` ROW-parallel over its two (attention output, MLP
activation) with one `all_reduce`. See `_stubs/flux2_parallel_self_attention.py`
for the derivation.

What the BLOCK adds is the residual stream, and it needs no collective of its
own: `attn` already all_reduces to a full-width result on every chip, so the
`hidden_states + gate * attn_out` residual is a replicated full-width tensor
between blocks. The LayerNorm has `elementwise_affine=False` (nothing to shard),
and the shift/scale/gate vectors are elementwise, so they stay REPLICATED --
sharding them would only force a gather before the very next multiply.

The implementation lives in `_flux2_ttnn.py`. The forward is pure ttnn: no torch
math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtFlux2SingleTransformerBlock


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("flux2_single_transformer_block needs the torch reference module for weights")
    return TtFlux2SingleTransformerBlock(device, torch_module)


def flux2_single_transformer_block(device, torch_module=None):
    return build(device, torch_module)
