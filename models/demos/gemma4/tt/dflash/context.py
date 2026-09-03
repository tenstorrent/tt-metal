# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the DFlash drafter's "context" from the target's tapped hidden states
(models/demos/gemma4/docs/dflash_design.md section 3, "Context tap mechanism").

This is the simple, non-trace-safe form: it assumes the caller already has all
6 tapped hidden-state tensors in hand (e.g. via Gemma4Model's layer_probe hook,
untraced) and concatenates + projects them in one call. The trace-safe,
FC-decomposed accumulate-as-you-go version (needed for production decode-loop
performance, ported from
models/demos/deepseek_v3_d_p/tt/dflash_prefill/tt_dflash_drafter.py's tap()/
_finalize_sharded_partial()) is a later hardening step -- see dflash_design.md.
"""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.dflash.weights import Gemma4DFlashWeights


def compute_context(weights: Gemma4DFlashWeights, tapped_hidden_states: list[ttnn.Tensor]) -> ttnn.Tensor:
    """concat(6 tapped hidden states) -> fc -> hidden_norm -> context.

    Each tapped hidden state: [1,1,seq,hidden_size], full-width replicated
    (Gemma4's residual stream between layers is allreduced back to full width
    after every layer's row-parallel O-proj/down-proj -- see weights.py's note
    on why fc is replicated rather than TP-sharded). Returns [1,1,seq,hidden_size],
    also full-width replicated -- exactly what every drafter layer's k_proj/
    v_proj (column-parallel, needs a full-width input) expects.
    """
    concat = ttnn.concat(tapped_hidden_states, dim=-1)  # [1,1,seq,6*hidden_size]
    proj = ttnn.linear(concat, weights.fc)
    ttnn.deallocate(concat)
    context = weights.hidden_norm(proj)
    ttnn.deallocate(proj)
    return context
