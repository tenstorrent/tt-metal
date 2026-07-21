# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of a single XTTS-v2 GPT decoder block.

Mirrors ``reference/xtts_gpt_block.py`` (a HuggingFace ``GPT2Block``):

    h = x + attn(ln_1(x))          # causal multi-head self-attention
    y = h + mlp(ln_2(h))           # c_fc -> gelu -> c_proj

Weight-layout notes:
  * GPT-2 uses ``Conv1D``, whose weight is stored ``[in, out]`` — already the
    layout ``ttnn.linear`` expects (y = x @ W + b), so NO transpose is needed.
  * Attention is causal with scale ``1/sqrt(head_dim)`` — matches the defaults
    of ``ttnn.transformer.scaled_dot_product_attention``.
"""

import math

import torch
import ttnn

from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    LAYER_NORM_EPS,
    NUM_HEADS,
)
from models.common.lightweightmodule import LightweightModule

NEG_INF = -1e30  # additive attention-mask fill for masked-out (future) positions


def _to_device(torch_tensor, device):
    """torch -> ttnn bf16 tile tensor on device."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


class TtXttsGptBlock(LightweightModule):
    def __init__(
        self,
        state_dict,
        device,
        layer_idx=0,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx

        prefix = f"gpt.gpt.h.{layer_idx}."

        # Load layer norm parameters
        self.ln_1_weight = _to_device(state_dict[prefix + "ln_1.weight"], device)
        self.ln_1_bias = _to_device(state_dict[prefix + "ln_1.bias"], device)
        self.ln_2_weight = _to_device(state_dict[prefix + "ln_2.weight"], device)
        self.ln_2_bias = _to_device(state_dict[prefix + "ln_2.bias"], device)

        # Load attention parameters
        self.attn_c_attn_weight = _to_device(state_dict[prefix + "attn.c_attn.weight"], device)
        self.attn_c_attn_bias = _to_device(state_dict[prefix + "attn.c_attn.bias"], device)
        self.attn_c_proj_weight = _to_device(state_dict[prefix + "attn.c_proj.weight"], device)
        self.attn_c_proj_bias = _to_device(state_dict[prefix + "attn.c_proj.bias"], device)

        # Load MLP parameters
        self.mlp_c_fc_weight = _to_device(state_dict[prefix + "mlp.c_fc.weight"], device)
        self.mlp_c_fc_bias = _to_device(state_dict[prefix + "mlp.c_fc.bias"], device)
        self.mlp_c_proj_weight = _to_device(state_dict[prefix + "mlp.c_proj.weight"], device)
        self.mlp_c_proj_bias = _to_device(state_dict[prefix + "mlp.c_proj.bias"], device)

    def _qkv(self, x):  # [b, s, hidden] -> q, k, v each [b, heads, s, head_dim]
        # One fused op replaces the slice x3 + reshape/permute x3 head-split: it splits the
        # [b, s, 3*hidden] c_attn output (GPT-2 [Q|K|V] block layout) into per-head Q, K, V.
        # transpose_key=False keeps K as [b, heads, s, head_dim] — SDPA and the decode KV
        # cache expect that layout, not K^T.
        qkv = ttnn.linear(x, self.attn_c_attn_weight, bias=self.attn_c_attn_bias)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=NUM_HEADS, transpose_key=False)
        ttnn.deallocate(qkv)
        return q, k, v

    def _attn_out(self, attn):  # [b, heads, s, head_dim] -> [b, s, hidden]
        out = ttnn.transformer.concatenate_heads(attn)  # fused permute + reshape
        ttnn.deallocate(attn)
        proj = ttnn.linear(out, self.attn_c_proj_weight, bias=self.attn_c_proj_bias)
        ttnn.deallocate(out)
        return proj

    def _attention(self, x):
        """Non-cached causal attention (full-sequence path). Consumes ``x``."""
        q, k, v = self._qkv(x)
        ttnn.deallocate(x)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return self._attn_out(attn)

    def _mlp(self, x):
        """c_fc -> gelu -> c_proj. Consumes ``x``."""
        h = ttnn.linear(x, self.mlp_c_fc_weight, bias=self.mlp_c_fc_bias)
        ttnn.deallocate(x)
        g = ttnn.gelu(h)  # tanh-approx GELU ~= GPT-2 "gelu_new" (validated by PCC)
        ttnn.deallocate(h)
        out = ttnn.linear(g, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias)
        ttnn.deallocate(g)
        return out

    def _residual_ffn(self, x):
        """Shared post-attention half: ``x + mlp(ln_2(x))``. Consumes and replaces ``x``."""
        h = ttnn.layer_norm(x, weight=self.ln_2_weight, bias=self.ln_2_bias, epsilon=LAYER_NORM_EPS)
        m = self._mlp(h)  # consumes h
        y = ttnn.add(x, m)
        ttnn.deallocate(x)
        ttnn.deallocate(m)
        return y

    def forward(self, x):
        """Run one XTTS GPT decoder block. ``x`` is ``[batch, seq, hidden]`` on device.

        Intermediates are freed as soon as they are consumed so at most one residual
        tensor and the current op's output coexist — this lowers the transient
        high-water mark (weights stay resident in DRAM; only activations churn)."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS)
        a = self._attention(h)  # consumes h
        xa = ttnn.add(x, a)
        ttnn.deallocate(x)
        ttnn.deallocate(a)
        return self._residual_ffn(xa)

    def forward_prefill(self, x):
        """Prefill: full causal attention over the prompt, plus the per-layer K, V
        (each ``[b, heads, seq, head_dim]``) used to seed the decode KV cache. K/V are
        kept (returned for the cache); every other intermediate is deallocated."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS)
        q, k, v = self._qkv(h)
        ttnn.deallocate(h)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
        ttnn.deallocate(q)  # k, v kept for the cache
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa), k, v

    def forward_decode_static(self, x, k_cache, v_cache, onehot, keep, add_mask):
        """Trace-compatible decode step over a FIXED-size KV cache (no concat growth).

        ``k_cache``/``v_cache`` are ``[1, heads, MAX, head_dim]`` PERSISTENT buffers updated
        IN PLACE at the current position with a device one-hot (``onehot`` ``[1, 1, MAX, 1]``,
        ``keep = 1 - onehot``); attention then runs over the whole cache with an additive
        position mask (``add_mask`` ``[1, 1, 1, MAX]``: 0 for cached positions, -inf ahead).
        In-place cache writes + static shapes + device-only ops mean one capture replays at
        any position and the caches accumulate across replays. Returns the FFN output."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS)
        q, k, v = self._qkv(h)  # each [1, heads, 1, head_dim]
        ttnn.deallocate(h)
        # In-place cache write at the current position: cache = cache*keep + newKV*onehot.
        # keep is 0 at the write row (1 elsewhere); onehot is 1 at the write row. The single
        # token's K/V [1,h,1,d] broadcasts over MAX; onehot/keep [1,1,MAX,1] over heads/head_dim.
        ttnn.multiply(k_cache, keep, output_tensor=k_cache)
        ttnn.multiply(v_cache, keep, output_tensor=v_cache)
        kw = ttnn.multiply(k, onehot)
        vw = ttnn.multiply(v, onehot)
        ttnn.add(k_cache, kw, output_tensor=k_cache)
        ttnn.add(v_cache, vw, output_tensor=v_cache)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(kw)
        ttnn.deallocate(vw)
        # Masked attention over the full cache: softmax(q·Kᵀ/√d + mask) · V.
        kT = ttnn.permute(k_cache, (0, 1, 3, 2))  # [1, heads, head_dim, MAX]
        scores = ttnn.multiply(ttnn.matmul(q, kT), 1.0 / math.sqrt(HEAD_DIM))  # [1, heads, 1, MAX]
        ttnn.deallocate(kT)
        ttnn.deallocate(q)
        scores = ttnn.add(scores, add_mask)
        p = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        attn = ttnn.matmul(p, v_cache)  # [1, heads, 1, head_dim]
        ttnn.deallocate(p)
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa)

    def forward_decode(self, x, k_cache, v_cache):
        """Decode one token. ``x`` is ``[b, 1, hidden]``; ``k_cache``/``v_cache`` are
        ``[b, heads, cur_len, head_dim]``. Appends this token's K, V to the cache and
        attends the single query against the whole cache (all cached positions are
        causal-valid), so no mask is needed. Returns the grown cache; the previous
        cache buffers are freed once concatenated into the new ones."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS)
        q, k, v = self._qkv(h)  # each [b, heads, 1, head_dim]
        ttnn.deallocate(h)
        new_k = ttnn.concat([k_cache, k], dim=2)
        new_v = ttnn.concat([v_cache, v], dim=2)
        ttnn.deallocate(k_cache)
        ttnn.deallocate(v_cache)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.transformer.scaled_dot_product_attention(q, new_k, new_v, is_causal=False)
        ttnn.deallocate(q)
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa), new_k, new_v
