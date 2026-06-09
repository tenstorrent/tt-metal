# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused attention output projection: head concat + w_o linear in one pipelined path."""

from __future__ import annotations

import os
from typing import Callable

import ttnn


def attn_o_proj_from_heads(
    v: ttnn.Tensor,
    w_o: ttnn.Tensor,
    *,
    num_heads: int,
    v_head_dim: int,
    total_seq: int,
    batch: int,
    seq_len: int,
    linear_fn: Callable[..., ttnn.Tensor],
    tp_linear_fn: Callable[..., ttnn.Tensor] | None = None,
    tp_enabled: bool = False,
) -> ttnn.Tensor:
    """Concatenate per-head V and apply output projection with minimal layout churn.

    Replaces the separate nlp_concat_heads → reshape → w_o sequence.  For batch=1
    nlp_concat_heads already emits [1,1,S,H*D], so the extra reshape is skipped.
    Output is staged in L1 so w_o reads the concat result without a DRAM round-trip.
    """
    in_mc = ttnn.L1_MEMORY_CONFIG
    hidden_in = int(num_heads * v_head_dim)
    use_nlp = os.environ.get("GLM4_MOE_LITE_NLP_CONCAT_HEADS", "1").strip() != "0"

    if use_nlp:
        v_flat = ttnn.experimental.nlp_concat_heads(v, memory_config=in_mc)
        ttnn.deallocate(v, force=False)
    else:
        v = ttnn.permute(v, (0, 2, 1, 3))
        v_flat = ttnn.reshape(v, (batch, 1, seq_len, hidden_in))
        ttnn.deallocate(v, force=False)
        if batch > 1:
            v_flat = ttnn.reshape(v_flat, (1, 1, total_seq, hidden_in))
        v_flat = ttnn.to_memory_config(v_flat, in_mc)

    if tp_enabled and tp_linear_fn is not None:
        out = tp_linear_fn(v_flat, w_o)
    else:
        out = linear_fn(v_flat, w_o)
    ttnn.deallocate(v_flat, force=False)
    return out
