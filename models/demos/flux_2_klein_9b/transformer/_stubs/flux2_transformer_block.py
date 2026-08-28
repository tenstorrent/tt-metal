# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2TransformerBlock`
(`transformer_blocks.0`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split(temb_mod_img, 6)
    c_shift_msa, ...                                               = split(temb_mod_txt, 6)

    attn_img, attn_ctx = attn((1+scale_msa)*norm1(x) + shift_msa,
                              (1+c_scale_msa)*norm1_context(ctx) + c_shift_msa)
    x   = x   + gate_msa   * attn_img
    x   = x   + gate_mlp   * ff((1+scale_mlp)*norm2(x) + shift_mlp)
    ctx = ctx + c_gate_msa * attn_ctx
    ctx = ctx + c_gate_mlp * ff_context((1+c_scale_mlp)*norm2_context(ctx) + c_shift_mlp)
    return ctx, x

One of the DUAL-STREAM blocks: the image tokens (`hidden_states`) and the text
tokens (`encoder_hidden_states`) keep separate norms, feed-forwards and
modulation vectors, but share ONE joint attention -- the two streams are
concatenated along the sequence so a single softmax attends over their union,
then split back. Each stream carries its own residual, and the block returns
them in the reference's order, `(encoder_hidden_states, hidden_states)`.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
Every parameter of this block lives in a sub-layer that is already tensor-
parallel on its own, so the block inherits their schemes and adds no weight
split of its own:

* `attn` (`Flux2Attention`): `to_q/k/v` and `add_q/k/v_proj` COLUMN-parallel --
  the output features are head-major, so a contiguous 1/TP slice is exactly
  32/8 = 4 whole heads. Both streams split the SAME way, so the sequence-concat
  before the joint softmax joins MATCHING heads and each chip runs a complete
  softmax over its own heads with no collective. `to_out[0]` and `to_add_out`
  are ROW-parallel, one `all_reduce` (SUM) per stream.
* `ff` and `ff_context` (`Flux2FeedForward`): `linear_in` COLUMN-parallel over
  the SwiGLU-regrouped columns (see `_stubs/flux2_feed_forward.py` -- the fused
  24576 columns are reordered at load time so each chip holds matching halves of
  the gate), `linear_out` ROW-parallel with one `all_reduce`.

What the BLOCK contributes is the residual stream and the modulation, and
neither needs a collective:

* `norm1/norm1_context/norm2/norm2_context` are REPLICATED. They have no affine
  parameters to split, and they reduce over the full model dim -- normalising a
  1/TP slice would be a different function, so the tensor must be full-width
  here, which it is: every sub-layer ends in an `all_reduce`, so the residual
  stream between them is full-width on every chip.
* The twelve shift/scale/gate vectors are elementwise over that same full-width
  residual, so they stay REPLICATED. Sharding them would only force a gather
  before the very next multiply.

Math is unchanged -- only the placement of the same products and sums; the
gathered output equals the single-device golden. The implementation lives in
`_flux2_ttnn.py`, shared with the sub-layers this block is built from. The
forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtFlux2TransformerBlock


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("flux2_transformer_block needs the torch reference module for weights")
    return TtFlux2TransformerBlock(device, torch_module)


def flux2_transformer_block(device, torch_module=None):
    return build(device, torch_module)
