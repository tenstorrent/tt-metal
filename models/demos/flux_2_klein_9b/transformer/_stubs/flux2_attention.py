# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2Attention`
(`transformer_blocks.0.attn`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

The dual-stream block's JOINT attention: the image stream goes through
`to_q/to_k/to_v` and the text stream through `add_{q,k,v}_proj`, both are
per-head RMS-normalised, then concatenated along the sequence (text first) so a
single softmax attends over the union; the result is split back and each stream
gets its own output projection (`to_out[0]` / `to_add_out`).

The planner mapped this component to `models/tt_transformers/tt/attention.py`.
That reference's Megatron column-then-row scheme carries over directly -- it is
the same shape of layer -- so this port keeps the scheme and replaces the body,
which differs in three ways the reference does not cover: two sets of
projections feeding one joint softmax, QK-RMSNorm over head_dim, and Flux's
adjacent-pair RoPE convention.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
* `to_q/to_k/to_v` and `add_q/add_k/add_v_proj`: COLUMN-parallel. Output
  features are head-major, so a contiguous 1/TP slice is exactly `32/8 = 4`
  whole heads (512 features, 16 tiles). No collective: QK-norm is over head_dim
  (never split), RoPE is per-position, and softmax is per-head, so each chip
  finishes its own heads independently. Both streams are split the SAME way, so
  the sequence-concat before the softmax joins matching heads.
* `norm_q/norm_k/norm_added_q/norm_added_k`: REPLICATED -- they normalise over
  head_dim, which is not split, and they are elementwise.
* RoPE cos/sin tables: REPLICATED lookup tables.
* `to_out[0]` and `to_add_out`: ROW-parallel. Each reduces its stream's per-head
  output back to the model dim, so its INPUT features are split and each chip
  produces a partial sum; one `all_reduce` (SUM) per stream makes every chip hold
  the single-device answer.

Math is unchanged -- only the placement of the same products and sums. The
implementation lives in `_flux2_ttnn.py`, shared with the blocks that contain
this layer. The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtFlux2Attention


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("flux2_attention needs the torch reference module for weights")
    return TtFlux2Attention(device, torch_module)


def flux2_attention(device, torch_module=None):
    return build(device, torch_module)
