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
L1 = ttnn.L1_MEMORY_CONFIG  # keep activations in L1 (weights stay in DRAM); the profiler flags the
# decode matmuls' input-0 as DRAM-resident — an L1 activation avoids that per-matmul DRAM read.


def _to_device(torch_tensor, device):
    """torch -> ttnn bf16 tile tensor on device."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def _to_device_w8(torch_tensor, device):
    """torch -> ttnn bfloat8_b tile weight. The decode matmuls are batch-1 (M=32, one token padded
    to a tile) so they are memory-bound — dominated by streaming the weight from DRAM. bfloat8_b
    (block-float, ~half the bytes of bf16) roughly halves that load, and holds accuracy on these
    GPT-2 weights (validated by the block-decode PCC/Frobenius gate)."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
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

        # Attention/MLP weights in bfloat8_b (memory-bound decode matmuls — see _to_device_w8);
        # biases stay bf16 (tiny, added in the fused epilogue).
        self.attn_c_attn_weight = _to_device_w8(state_dict[prefix + "attn.c_attn.weight"], device)
        self.attn_c_attn_bias = _to_device(state_dict[prefix + "attn.c_attn.bias"], device)
        self.attn_c_proj_weight = _to_device_w8(state_dict[prefix + "attn.c_proj.weight"], device)
        self.attn_c_proj_bias = _to_device(state_dict[prefix + "attn.c_proj.bias"], device)

        self.mlp_c_fc_weight = _to_device_w8(state_dict[prefix + "mlp.c_fc.weight"], device)
        self.mlp_c_fc_bias = _to_device(state_dict[prefix + "mlp.c_fc.bias"], device)
        self.mlp_c_proj_weight = _to_device_w8(state_dict[prefix + "mlp.c_proj.weight"], device)
        self.mlp_c_proj_bias = _to_device(state_dict[prefix + "mlp.c_proj.bias"], device)

    def _qkv(self, x):  # [b, s, hidden] -> q, k, v each [b, heads, s, head_dim]
        # Split the [b, s, 3*hidden] c_attn output (GPT-2 [Q|K|V] block layout) into per-head Q, K, V.
        # ttnn.experimental.nlp_create_qkv_heads is measurably faster than the transformer-namespace
        # split_query_key_value_and_split_heads wrapper (~43 vs ~63 us/call at decode shape, identical
        # output) — it wants a 4D [b, 1, s, 3*hidden] input, so add the leading singleton dim (a
        # metadata reshape). transpose_k_heads=False keeps K as [b, heads, s, head_dim] (SDPA + the
        # decode KV cache expect that layout, not K^T).
        qkv = ttnn.linear(x, self.attn_c_attn_weight, bias=self.attn_c_attn_bias, memory_config=L1)
        b, s, three_h = qkv.shape
        qkv = ttnn.reshape(qkv, (b, 1, s, three_h))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=NUM_HEADS, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        return q, k, v

    def _attn_out(self, attn):  # [b, heads, s, head_dim] -> [b, s, hidden]
        out = ttnn.transformer.concatenate_heads(attn)  # fused permute + reshape
        ttnn.deallocate(attn)
        proj = ttnn.linear(out, self.attn_c_proj_weight, bias=self.attn_c_proj_bias, memory_config=L1)
        ttnn.deallocate(out)
        return proj

    def _mlp(self, x):
        """c_fc (+ GELU fused into the matmul epilogue) -> c_proj. Consumes ``x``."""
        # activation="gelu" runs the GELU in the c_fc matmul's fused epilogue instead of a separate
        # ttnn.gelu op (one fewer op + one fewer intermediate). ~= GPT-2 "gelu_new" (validated by PCC).
        h = ttnn.linear(x, self.mlp_c_fc_weight, bias=self.mlp_c_fc_bias, activation="gelu", memory_config=L1)
        ttnn.deallocate(x)
        out = ttnn.linear(h, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias, memory_config=L1)
        ttnn.deallocate(h)
        return out

    def _residual_ffn(self, x):
        """Shared post-attention half: ``x + mlp(ln_2(x))``. Consumes and replaces ``x``."""
        h = ttnn.layer_norm(x, weight=self.ln_2_weight, bias=self.ln_2_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        m = self._mlp(h)  # consumes h
        y = ttnn.add(x, m, memory_config=L1)
        ttnn.deallocate(x)
        ttnn.deallocate(m)
        return y

    def forward_prefill(self, x):
        """PREFILL — one of the block's two forwards (the other is ``forward_decode``).

        Full causal attention over the prompt, plus the per-layer K, V (each
        ``[b, heads, seq, head_dim]``) used to seed the decode KV cache. K/V are kept
        (returned for the cache); every other intermediate is deallocated. Also serves the
        full teacher-forced pass (callers that want only the hidden state take ``[0]``)."""
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

    def forward_decode(self, x, k_cache, v_cache, onehot, add_mask, write_idx=None):
        """DECODE — one of the block's two forwards. One token over a FIXED-size KV cache
        (no concat growth: concat on a tile-misaligned seq dim forces untilize->concat->retilize,
        ~15% of the step — this path avoids all of it).

        ``k_cache``/``v_cache`` are ``[1, heads, MAX, head_dim]`` PERSISTENT buffers updated IN
        PLACE at the current position; attention then runs over the whole cache with an additive
        position mask (``add_mask`` ``[1, 1, 1, MAX]``: 0 for cached positions, -inf ahead). Two
        cache-write modes:
          * EAGER (``write_idx`` = Python int): ``ttnn.update_cache`` writes ONLY that row — O(1),
            ~2x faster than touching the whole cache.
          * TRACED (``write_idx`` None): a device one-hot select ``where(onehot, newKV, cache)``
            ([1,1,MAX,1], 1 at the write row) — data-driven, so one capture replays at any position.
        Returns the FFN output."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS, memory_config=L1)
        q, k, v = self._qkv(h)  # each [1, heads, 1, head_dim]
        ttnn.deallocate(h)
        if write_idx is not None:
            ttnn.update_cache(k_cache, k, write_idx)  # O(1): write only row write_idx
            ttnn.update_cache(v_cache, v, write_idx)
        else:
            # data-driven select at the one-hot row (trace-safe; whole-cache elementwise).
            ttnn.where(onehot, k, k_cache, output_tensor=k_cache)
            ttnn.where(onehot, v, v_cache, output_tensor=v_cache)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # Masked attention over the full fixed cache, fused into ONE SDPA op (scale + q·Kᵀ + additive
        # mask + softmax + ·V) instead of permute+matmul+mul+add+softmax+matmul. ``add_mask``
        # [1, 1, 1, MAX] is 0 for cached positions, -inf ahead (broadcasts over heads and the 1 query).
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k_cache, v_cache, attn_mask=add_mask, is_causal=False, scale=1.0 / math.sqrt(HEAD_DIM)
        )  # [1, heads, 1, head_dim]
        ttnn.deallocate(q)
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao, memory_config=L1)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa)
