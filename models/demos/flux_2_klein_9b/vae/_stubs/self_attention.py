# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `self_attention`.

Reference module: `decoder.mid_block.attentions.0` of `AutoencoderKLFlux2` — a
diffusers `Attention` driven by `AttnProcessor2_0`. `self_attention` arrived as a
transformer-template role wired to the canonical
`models/tt_transformers/tt/attention.py::Attention`, but that class cannot stand
in for this module: it is a GQA text-attention with RoPE, a KV cache and a
`ModelArgs` Llama-style config, while this is *single-head spatial
self-attention over a feature map*, pre-normed by a GroupNorm and closed with a
residual — no rotary, no cache, no head grouping. There is no config of the
canonical class that expresses it, so the port is native ttnn instead.

What it is, and why it is shared
--------------------------------
This model has exactly two `Attention` modules, `encoder.mid_block.attentions.0`
and `decoder.mid_block.attentions.0`, and they are structurally identical
(query_dim=512, heads=1, GroupNorm(32 groups, eps=1e-6), residual_connection=True,
rescale_output_factor=1). The `attention` component already brought that module up
natively and tensor-parallel, so `self_attention` reuses that implementation —
`TtVaeAttention` — rather than carrying a second copy of the same TP scheme. This
is the same cross-stub reuse the graduated `encoder` and `decoder` stubs already
do (both build their mid block with `attention_factory=TtVaeAttention`); keeping
one implementation means the TP derivation is stated, and fixed, in one place.

Tensor parallelism (TP=8) — see `_stubs/attention.py` for the full derivation
-----------------------------------------------------------------------------
`heads == 1`, so the textbook Megatron head-parallel split has no axis to work
with; the split lands one level down, on the qkv CHANNEL axis (512 = 8 x 64):

  * `to_q`/`to_k`/`to_v` are COLUMN-parallel (`ShardTensorToMesh` on the output
    features, each bias split with its own columns). Because the score matmul
    `q @ k^T` contracts over exactly that split axis, the column-parallel qkv is
    followed by its natural collective, an `all_gather` on the feature dim, so
    every device forms the full `[N, N]` scores and probabilities.
  * `v` is deliberately left sharded: `probs @ v_shard` keeps the context
    channel-sharded, which is the layout the out projection wants.
  * `to_out` is ROW-parallel (`W` split on its INPUT features) so each device
    yields a partial sum over the full model dim, closed by an `all_reduce`. Its
    bias is REPLICATED and added AFTER the reduction — sharding it would add it
    eight times.
  * GroupNorm gamma/beta, the group-membership matrices, the out bias and the
    residual stay REPLICATED: elementwise/lookup tensors never shard, and the
    residual is full model dim on every device regardless.

Placement changes, math does not: the gathered `q`/`k` are the concatenation of
what a single device computes, and the `all_reduce`d `to_out` partials sum to the
single-device product — so the gathered output matches the golden.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs.attention import TtVaeAttention


def build(device, torch_module=None):
    return TtVaeAttention(device, torch_module)


def self_attention(device, torch_module=None):
    return TtVaeAttention(device, torch_module)
