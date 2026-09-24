# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Bidirectional attention for ModernBERT, shared by the global and local layers.

Op sequence:

    linear(Wqkv) -> split_query_key_value_and_split_heads -> rotary
                 -> scaled_dot_product_attention -> concatenate_heads -> linear(Wo)

Global and local layers differ only in the rotary theta they select and the mask
the caller supplies; see modernbert_masks.

SDPA's `sliding_window_size` argument is deliberately unused: passing it with a
`compute_kernel_config` hangs the device (each alone is fine). This is upstream
issue #51223 item 2 - the mask generator's K-chunk loop is bounded by the causal
diagonal even for a non-causal windowed call, so it under-fills cb_mask_in. The
window is handed to SDPA as an ordinary additive attn_mask instead, which costs
the sliding layers 49% more than the full-attention ones.

The sliding mask fills outside the band with -1e30, not -inf. Under chunked
attention some query rows have an entire k-chunk masked, and -inf would make that
chunk's running max -inf and produce NaN; -1e30 underflows exp() to zero.

The 1/sqrt(head_dim) scale is folded into Wq at load time and `scale=1.0` passed
here, because sdpa.cpp rescales attn_mask on every call unless scale is exactly
1.0. See `_qkv_linear` in weights.py.
"""

import ttnn
from models.experimental.modernbert.tt import model_config as _cfg
from models.experimental.modernbert.tt.model_config import compute_kernel_config


class TtnnModernBertAttention:
    def __init__(
        self,
        parameters,
        config,
        layer_type,
        down_core_grid=None,
        qkv_program_config=None,
        sdpa_program_config=None,
    ):
        self.Wqkv = parameters["Wqkv"]
        self.Wo = parameters["Wo"]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_type = layer_type
        self.compute_kernel_config = compute_kernel_config()
        self.down_core_grid = down_core_grid
        # None for shapes that were not measured, in which case ttnn chooses.
        self.qkv_program_config = qkv_program_config
        self.sdpa_program_config = sdpa_program_config
        # Built once: __call__ runs 22 times per pass and rebuilding these is host
        # work. None keeps each op's default; L1 carries the chain, see _L1_ATTENTION.
        self._mem = _cfg.attention_interleaved() if _cfg._L1_ATTENTION else None
        self._qkv_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if qkv_program_config is not None:
            self._qkv_kwargs["program_config"] = qkv_program_config
        if self._mem is not None:
            self._qkv_kwargs["memory_config"] = self._mem
        self._sdpa_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if sdpa_program_config is not None:
            self._sdpa_kwargs["program_config"] = sdpa_program_config
        if self._mem is not None:
            self._sdpa_kwargs["memory_config"] = self._mem
        self._split_kwargs = {} if self._mem is None else {"memory_config": self._mem}
        self._wo_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if down_core_grid is not None:
            self._wo_kwargs["core_grid"] = down_core_grid
        if self._mem is not None:
            self._wo_kwargs["memory_config"] = self._mem

    def __call__(self, hidden_states, rotary, attn_mask):
        """hidden_states: (B, S, hidden). attn_mask: additive (B, 1, S, S) mask for
        this layer type, from build_masks."""
        qkv = ttnn.linear(hidden_states, self.Wqkv, **self._qkv_kwargs)

        # transpose_key=False because rotary embeddings must be applied to K while
        # it is still (B, n_heads, S, head_dim). SDPA transposes it internally.
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, transpose_key=False, **self._split_kwargs
        )
        ttnn.deallocate(qkv)

        rotated_q = rotary(query, self.layer_type)
        rotated_k = rotary(key, self.layer_type)
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        context = ttnn.transformer.scaled_dot_product_attention(
            rotated_q,
            rotated_k,
            value,
            attn_mask=attn_mask,
            # ModernBERT is bidirectional. is_causal defaults to True, and leaving
            # it would silently return a causally masked result.
            is_causal=False,
            # Folded into Wq at load time; see the module docstring.
            scale=1.0,
            **self._sdpa_kwargs,
        )
        ttnn.deallocate(rotated_q)
        ttnn.deallocate(rotated_k)
        ttnn.deallocate(value)

        merged = ttnn.transformer.concatenate_heads(context, **self._split_kwargs)
        ttnn.deallocate(context)

        out = ttnn.linear(merged, self.Wo, **self._wo_kwargs)
        ttnn.deallocate(merged)
        return out
