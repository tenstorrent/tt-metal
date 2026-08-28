# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2ParallelSelfAttention`
(`single_transformer_blocks.0.attn`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

The single-stream block's ViT-22B-style *parallel* attention: the attention and
the feed-forward run side by side off the same normalised input, and both are
fused into ONE input projection and ONE output projection.

    fused = to_qkv_mlp_proj(x)                        # 4096 -> 3*4096 + 2*12288
    qkv, mlp = split(fused, [3*4096, 2*12288])
    q, k, v  = qkv.chunk(3)                           # then QK-RMSNorm + RoPE
    attn     = sdpa(q, k, v)
    mlp      = Flux2SwiGLU(mlp)                       # silu(first half)*second
    out      = to_out(cat([attn, mlp]))               # 4096+12288 -> 4096

The planner mapped this to `models/tt_transformers/tt/attention.py`; that
reference's column-then-row scheme is the right one, but its q/k/v are three
separate linears and its output projection consumes attention alone, so the
fusion needs the extra handling below.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
* `to_qkv_mlp_proj`: COLUMN-parallel over FIVE declared groups -- q (4096),
  k (4096), v (4096), and SwiGLU's two 12288-wide halves. A contiguous 8-way
  split of the 36864 columns would be wrong twice over: it would cut across the
  q/k/v boundaries (chip 0 would own all of q and part of k, with no v at all)
  and it would separate SwiGLU's paired halves. Declaring the groups makes
  `_regroup` reorder the columns to `[q_c, k_c, v_c, mlp1_c, mlp2_c]` per chip
  c, so one contiguous `ShardTensorToMesh(dim=-1)` gives every chip 4 whole
  heads of q, the SAME 4 heads of k and v, and matching 1536-wide slices of both
  SwiGLU halves. Nothing downstream (QK-norm over head_dim, RoPE, softmax, the
  gate) crosses those boundaries, so no collective is needed here.
* `norm_q` / `norm_k`: REPLICATED -- over head_dim, which is never split.
* RoPE cos/sin: REPLICATED lookup tables.
* `to_out`: ROW-parallel over TWO groups -- the attention output (4096) and the
  MLP activation (12288). Its input is their concatenation, and each chip holds
  512 features of the first and 1536 of the second, so its ROWS are regrouped the
  same way before `ShardTensorToMesh(dim=2)`; a contiguous 2048-row split would
  have taken the wrong rows. One `all_reduce` (SUM) then turns the per-chip
  partial sums into the single-device answer.

Math is unchanged -- only the placement. The implementation lives in
`_flux2_ttnn.py`, shared with the single-stream block that contains this layer.
The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtFlux2ParallelSelfAttention


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("flux2_parallel_self_attention needs the torch reference module for weights")
    return TtFlux2ParallelSelfAttention(device, torch_module)


def flux2_parallel_self_attention(device, torch_module=None):
    return build(device, torch_module)
