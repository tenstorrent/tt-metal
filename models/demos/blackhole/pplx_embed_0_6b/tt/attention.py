# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
pplx-embed-v1-0.6B bidirectional attention subclass.

pplx-embed uses full bidirectional attention (every token attends to every
other token) rather than the causal mask used by standard Qwen3.  The only
change required at the TT graph level is flipping ``is_causal=True`` to
``is_causal=False`` in the ``ttnn.transformer.scaled_dot_product_attention``
call inside ``forward_prefill``.

Also incorporates the bs=1/ISL=512 Q-BFP8 skip optimisation from
Qwen3EmbeddingAttention: when ``TT_SKIP_KV_CACHE_FILL=1`` is set, K/V stay
native bf16, so the Q→BFP8 typecast is pure overhead and can be skipped.

Implementation uses a temporary wrapper around ``ttnn.transformer.scaled_dot_product_attention``
to inject ``is_causal=False`` without duplicating the entire 280-line
``forward_prefill`` method, keeping the subclass resilient to upstream changes.
"""

import functools

import ttnn
from models.tt_transformers.tt.attention import Attention

_OPTIMIZED_BATCH = 1
_OPTIMIZED_SEQ_LEN = 512


def _wrap_sdpa_bidirectional(original_fn):
    """Return a wrapper that forces ``is_causal=False`` on every call."""

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        kwargs["is_causal"] = False
        return original_fn(*args, **kwargs)

    return wrapper


class PplxBidirectionalAttention(Attention):
    """Drop-in replacement for ``tt_transformers.tt.attention.Attention``
    that uses bidirectional (non-causal) SDPA for pplx-embed models."""

    def _prepare_q_for_sdpa(self, q_heads_1QSD: ttnn.Tensor) -> ttnn.Tensor:
        if (
            self.max_batch_size == _OPTIMIZED_BATCH
            and self.max_seq_len == _OPTIMIZED_SEQ_LEN
            and getattr(self.args, "skip_kv_cache_fill", False)
        ):
            return q_heads_1QSD
        return super()._prepare_q_for_sdpa(q_heads_1QSD)

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        original_sdpa = ttnn.transformer.scaled_dot_product_attention
        ttnn.transformer.scaled_dot_product_attention = _wrap_sdpa_bidirectional(original_sdpa)
        try:
            return super().forward_prefill(
                x_11SH,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        finally:
            ttnn.transformer.scaled_dot_product_attention = original_sdpa


PplxBidirectionalAttention.__name__ = "Attention"
