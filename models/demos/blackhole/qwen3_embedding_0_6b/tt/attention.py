# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-0.6B-local attention subclass.

Houses BS=1 / ISL=512 prefill graph optimizations that are model + shape
specific and would not be safe to land directly in
``models/tt_transformers/tt/attention.py``.

Currently implemented:

* Skip the pre-SDPA ``Q -> BFP8`` typecast at bs=1 / ISL=512 when the
  embedding workload has ``TT_SKIP_KV_CACHE_FILL=1`` set. With KV-cache fill
  skipped, K/V already stay native bf16 going into SDPA, so casting Q to BFP8
  is pure overhead. Removing it saves 28 typecast ops/iter (one per decoder
  layer). Measured 7.8 -> 7.7 ms best prefill on P150 / ISL=512.

The optimization is gated on the model-level ``(max_batch_size, max_seq_len)``
configuration; any other shape falls through to the unmodified parent path.
``max_seq_len`` here is the configured prefill length on ``ModelArgs``, NOT
the per-call chunk size. ISL=512 prefill at bs=1 is internally executed as
four 128-token chunks, so a per-call shape gate would never fire.
"""

import ttnn
from models.tt_transformers.tt.attention import Attention

_OPTIMIZED_BATCH = 1
_OPTIMIZED_SEQ_LEN = 512


class Qwen3EmbeddingAttention(Attention):
    """Drop-in replacement for ``tt_transformers.tt.attention.Attention`` that
    applies Qwen3-Embedding-0.6B BS=1/ISL=512 prefill optimizations."""

    def _prepare_q_for_sdpa(self, q_heads_1QSD: ttnn.Tensor) -> ttnn.Tensor:
        """Override the pre-SDPA Q preparation hook.

        Conditions for the skip-typecast fast path:

        1. ``max_batch_size == _OPTIMIZED_BATCH`` (= 1)
        2. ``max_seq_len == _OPTIMIZED_SEQ_LEN`` (= 512) -- the model-level
           configured prefill length, not the per-call chunk seq.
        3. ``args.skip_kv_cache_fill`` is True so K/V stay native bf16.

        If any condition is unmet, fall back to the parent's typecast.
        """
        if (
            self.max_batch_size == _OPTIMIZED_BATCH
            and self.max_seq_len == _OPTIMIZED_SEQ_LEN
            and getattr(self.args, "skip_kv_cache_fill", False)
        ):
            return q_heads_1QSD
        return super()._prepare_q_for_sdpa(q_heads_1QSD)


# The base ``Attention`` class uses ``self.__class__.__name__`` at construction
# time to look up the state-dict prefix (e.g. ``"layers.0.attention."``) via
# the model_config module-name map (which only contains ``"Attention"``). To
# keep this subclass a true drop-in -- and to avoid touching shared code --
# alias the reported class name to ``"Attention"`` so the lookup matches.
# This is purely a metadata patch; instances still report the real class via
# ``type(obj).__qualname__`` and ``isinstance`` checks against either class
# continue to work as expected.
Qwen3EmbeddingAttention.__name__ = "Attention"
