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
import os

import ttnn
from models.tt_transformers.tt.attention import Attention

_OPTIMIZED_BATCH = 1
_OPTIMIZED_SEQ_LEN = 512

# Optional additive padding mask injected into the bidirectional prefill SDPA.
# Shape [b, nqh, S, S] (batch/head broadcastable). Real key columns are 0, padded
# key columns are a large negative value so padding is excluded from attention.
# A single shared device buffer is reused across all layers; the serving harness
# updates its contents per request before replaying the trace.
_PAD_ATTN_MASK = None


def set_pad_attn_mask(mask):
    """Set (or clear with ``None``) the shared bidirectional SDPA padding mask."""
    global _PAD_ATTN_MASK
    _PAD_ATTN_MASK = mask


def _wrap_sdpa_bidirectional(original_fn):
    """Return a wrapper that forces ``is_causal=False`` and injects the padding mask."""

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        kwargs["is_causal"] = False
        if _PAD_ATTN_MASK is not None and kwargs.get("attn_mask") is None:
            kwargs["attn_mask"] = _PAD_ATTN_MASK
        return original_fn(*args, **kwargs)

    return wrapper


class PplxBidirectionalAttention(Attention):
    """Drop-in replacement for ``tt_transformers.tt.attention.Attention``
    that uses bidirectional (non-causal) SDPA for pplx-embed models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ablation knob (DO NOT ENABLE): skipping the trained Q/K RMSNorm shaves
        # ~0.5ms (7.8ms -> 7.2ms on bs1/ISL512) but collapses STS-B Spearman from
        # 0.848 -> 0.236 — the per-head Q/K norm is load-bearing, not redundant.
        # Kept gated/off as documentation of the measured trade-off.
        if os.getenv("QWEN_SKIP_QK_NORM", "0") == "1":
            self.q_norm = lambda x, mode, norm_config: x
            self.k_norm = lambda x, mode, norm_config: x

    def _mllama_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        # The rotary_embedding_llama op defaults to MathFidelity::HiFi4 (4 math
        # passes). RoPE is just a cos/sin rotation (operands in [-1, 1]), so a
        # lower fidelity is near-lossless while cutting the kernel's math passes.
        # QWEN_ROPE_FIDELITY: lofi | hifi2 | hifi4 (unset -> stock HiFi4).
        fidelity = os.getenv("QWEN_ROPE_FIDELITY", "").lower()
        if fidelity == "lofi":
            ckc = self.args.compute_kernel_config_lofi
        elif fidelity == "hifi2":
            ckc = self.args.compute_kernel_config_hifi2
        else:
            return super()._mllama_rope_prefill(q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
            compute_kernel_config=ckc,
        )
        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
            compute_kernel_config=ckc,
        )
        return q_heads_1QSD, k_heads_1KSD

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
