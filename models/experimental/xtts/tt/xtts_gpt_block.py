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

KV caching:
  The block exposes three entry points sharing the same weights:
    * ``forward``          — full-sequence, stateless (no cache). Unchanged.
    * ``forward_prefill``  — full-sequence, but stores this block's per-head
                             K/V into ``self.k_cache`` / ``self.v_cache``.
    * ``forward_decode``   — one new token (``seq == 1``); appends its K/V to the
                             cache and attends the single query over the whole
                             cached history (non-causal, since every cached key
                             is a past position).
  The cache holds ``[batch, heads, seq, head_dim]`` tensors and grows by one row
  per decode step via ``ttnn.concat`` along the sequence dim.
"""

import torch
import ttnn

from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NUM_HEADS,
)
from models.common.lightweightmodule import LightweightModule


def _to_device(torch_tensor, device):
    """torch -> ttnn bf16 tile tensor on device."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


# Activation memory placement. Per-op intermediates (the "inputs" flowing through
# the block: layer-norm/linear/gelu/add/SDPA outputs) live in L1 for fast on-chip
# reuse. Weights stay in DRAM (loaded via _to_device, the default), and the KV
# cache stays in DRAM too — it grows to ~250 MB across the 30 blocks, far beyond
# the ~195 MB of on-chip L1, so it cannot be resident there.
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


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

        # Per-block KV cache (populated by forward_prefill, grown by forward_decode).
        self.k_cache = None
        self.v_cache = None

    def _split_heads(self, x, mem):  # [b, s, hidden] -> [b, heads, s, head_dim]
        b, s, _ = x.shape
        x = ttnn.reshape(x, (b, s, NUM_HEADS, HEAD_DIM))
        return ttnn.permute(x, (0, 2, 1, 3), memory_config=mem)

    def _merge_heads(self, x):  # [b, heads, s, head_dim] -> [b, s, hidden] (L1)
        b, _, s, _ = x.shape
        x = ttnn.permute(x, (0, 2, 1, 3), memory_config=L1)
        return ttnn.reshape(x, (b, s, HIDDEN_SIZE))

    def _qkv(self, x):  # ln_1'd hidden -> per-head (q, k, v), each [b, heads, s, head_dim]
        # q flows straight into SDPA -> L1; k/v become the KV cache -> DRAM.
        qkv = ttnn.linear(x, self.attn_c_attn_weight, bias=self.attn_c_attn_bias, memory_config=L1)
        b, s = qkv.shape[0], qkv.shape[1]
        q = ttnn.slice(qkv, [0, 0, 0], [b, s, HIDDEN_SIZE], memory_config=L1)
        k = ttnn.slice(qkv, [0, 0, HIDDEN_SIZE], [b, s, 2 * HIDDEN_SIZE], memory_config=DRAM)
        v = ttnn.slice(qkv, [0, 0, 2 * HIDDEN_SIZE], [b, s, 3 * HIDDEN_SIZE], memory_config=DRAM)
        ttnn.deallocate(qkv)
        return self._split_heads(q, L1), self._split_heads(k, DRAM), self._split_heads(v, DRAM)

    def _attn_out(self, attn):  # [b, heads, s, head_dim] -> [b, s, hidden] after c_proj (L1)
        out = self._merge_heads(attn)
        return ttnn.linear(out, self.attn_c_proj_weight, bias=self.attn_c_proj_bias, memory_config=L1)

    def _attention(self, x):
        q, k, v = self._qkv(x)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, memory_config=L1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return self._attn_out(attn)

    def _attention_prefill(self, x):
        """Like ``_attention`` but stores the per-head K/V (DRAM) into the KV cache."""
        q, k, v = self._qkv(x)
        self.k_cache, self.v_cache = k, v  # DRAM cache; kept alive, do NOT deallocate
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, memory_config=L1)
        ttnn.deallocate(q)
        return self._attn_out(attn)

    def _attention_decode(self, x):
        """Attention for one new token: append its K/V to the cache, attend over all of it."""
        q, k, v = self._qkv(x)  # q in L1; k, v in DRAM

        # Grow the DRAM cache by one position along the sequence dim.
        new_k = ttnn.concat([self.k_cache, k], dim=2, memory_config=DRAM)
        new_v = ttnn.concat([self.v_cache, v], dim=2, memory_config=DRAM)
        ttnn.deallocate(self.k_cache)
        ttnn.deallocate(self.v_cache)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        self.k_cache, self.v_cache = new_k, new_v

        # Single query attends to the whole cached history; every cached key is a
        # past (<=current) position, so no causal masking is needed here.
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, self.k_cache, self.v_cache, is_causal=False, memory_config=L1
        )
        ttnn.deallocate(q)
        return self._attn_out(attn)

    def _mlp(self, x):
        h = ttnn.linear(x, self.mlp_c_fc_weight, bias=self.mlp_c_fc_bias, memory_config=L1)
        h = ttnn.gelu(h, memory_config=L1)  # tanh-approx GELU ~= GPT-2 "gelu_new" (validated by PCC)
        return ttnn.linear(h, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias, memory_config=L1)

    def _residual_ffn(self, x, attn_out):
        """Shared post-attention residual + MLP tail of the block (activations in L1)."""
        x = ttnn.add(x, attn_out, memory_config=L1)
        h = ttnn.layer_norm(x, weight=self.ln_2_weight, bias=self.ln_2_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        return ttnn.add(x, self._mlp(h), memory_config=L1)

    def forward(self, x):
        """Run one XTTS GPT decoder block, stateless. ``x`` is ``[batch, seq, hidden]``."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        return self._residual_ffn(x, self._attention(h))

    def forward_prefill(self, x):
        """Run the block on the full prompt and seed this block's KV cache.

        ``x`` is ``[batch, seq, hidden]``; identical output to ``forward`` but the
        K/V computed here are retained for subsequent ``forward_decode`` steps.
        """
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        return self._residual_ffn(x, self._attention_prefill(h))

    def forward_decode(self, x):
        """Run the block on one new token (``x`` is ``[batch, 1, hidden]``).

        Requires ``forward_prefill`` to have been called first. Appends the new
        token's K/V to the cache and attends over the full history.
        """
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        return self._residual_ffn(x, self._attention_decode(h))
