# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA (DeepSeek sparse attention) indexer contract of the GLM-5.3-Flash MLA layers (modeling:772-1060).

Per query the indexer keeps the top `index_topk / index_kpool = 512` complete 4-token pools plus the
`visible_count mod 4` newest tokens: at most 2048 + 3 = 2051 attended tokens. When the visible context has at most
2051 tokens every visible token is selected, so the main attention is exactly dense causal attention and the indexer
output does not affect the result (modeling_glm5_next.py:883-913: the top-k width 2048 + 3 covers the visible set).

This decoder implements the MLA/DSA layer kind in this dense-exact regime only.
`check_dense_exact` is the single guard used by prefill, decode-position validation and cache allocation; a longer
context raises instead of silently computing dense attention where HF attends sparsely. The sparse path
(k-pooled indexer keys cache, `ttnn.experimental.indexer_score_dsa`, `topk_large_indices`, sparse latent attention)
is the deferred extension that lifts this bound.
"""

from __future__ import annotations

from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV


def dense_selection_is_exact(kv_len: int) -> bool:
    """True when the DSA top-k selection of every query is the full visible set."""
    return kv_len <= DSA_DENSE_EXACT_MAX_KV


def check_dense_exact(layer_idx: int, kv_len: int) -> None:
    if not dense_selection_is_exact(kv_len):
        raise NotImplementedError(
            f"MLA/DSA layer {layer_idx}: kv_len {kv_len} > {DSA_DENSE_EXACT_MAX_KV} needs the DSA indexer top-k path "
            "(outside the functional-decoder scope); dense attention would not match HF"
        )
