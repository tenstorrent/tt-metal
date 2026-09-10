# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the DFlash drafter's "context" from the target's tapped hidden states
(models/demos/gemma4/docs/dflash_design.md section 3, "Context tap mechanism").

``compute_context`` is the simple, non-trace-safe form: it assumes the caller already
has all 6 tapped hidden-state tensors in hand (e.g. via Gemma4Model's layer_probe hook,
untraced) and concatenates + projects them in one call.

``ContextAccumulator`` is the trace-safe, FC-decomposed accumulate-as-you-go form,
ported from models/demos/deepseek_v3_d_p/tt/dflash_prefill/tt_dflash_drafter.py's
tap()/_finalize_sharded_partial() -- simplified for this drafter's case, where ``fc``
is REPLICATED (not TP-sharded), so no reduce_scatter combine step is needed; the
Kimi-K2.6 version also handles pipeline-parallel ranks importing/exporting partial
sums, which doesn't apply here either. It exploits linearity of matmul over
concatenation: ``fc(concat[h_1..h_6]) == sum_i fc_slice_i(h_i)``, where
``fc_slice_i`` is the ``[hidden_size, hidden_size]`` row-block of ``fc`` corresponding
to tap ``i``'s columns. This sums ONE tap's contribution into a running total as each
layer's hidden state becomes available during the SAME forward pass, instead of
collecting all 6 into a Python list and concatenating afterward -- a python dict/list
that grows across calls is exactly what a Metal trace can't replay (see
tt/dflash/verify.py's module docstring); an accumulator that always holds at most one
running total is the shape a trace CAN replay, once wired into one (a later step --
this class is not yet used inside begin_trace_capture/end_trace_capture anywhere,
matching the same "shaped for tracing, not yet proven inside one" caveat the Kimi-K2.6
version itself carries)."""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.dflash.weights import Gemma4DFlashWeights
from models.demos.gemma4.tt.rms_norm import dflash_context_hidden_norm


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
    context = dflash_context_hidden_norm(weights.hidden_norm, proj)
    ttnn.deallocate(proj)
    return context


def split_fc_slices(weights: Gemma4DFlashWeights, target_layer_ids, hidden_size: int) -> dict[int, ttnn.Tensor]:
    """Slice ``weights.fc`` (``[1,1,num_taps*hidden_size,hidden_size]``) into one
    ``[1,1,hidden_size,hidden_size]`` slice per tap, keyed by the target layer id it
    corresponds to (taps are concatenated in ``target_layer_ids`` order, matching
    ``compute_context``'s ``tapped_hidden_states`` ordering convention)."""
    return {
        layer_id: ttnn.slice(
            weights.fc, [0, 0, i * hidden_size, 0], [1, 1, (i + 1) * hidden_size, weights.fc.shape[-1]]
        )
        for i, layer_id in enumerate(target_layer_ids)
    }


class ContextAccumulator:
    """Trace-safe-shaped FC-decomposed context accumulator: call ``tap()`` once per
    target layer as its hidden state becomes available (e.g. from ``model.layer_probe``),
    then ``finalize()`` once to get the same result ``compute_context`` would have given
    the full ``tapped_hidden_states`` list -- without ever holding more than one running
    total plus the current tap's partial."""

    def __init__(self, fc_slices: dict[int, ttnn.Tensor], hidden_norm):
        self.fc_slices = fc_slices
        self.hidden_norm = hidden_norm
        self._accum = None

    def tap(self, hidden_states: ttnn.Tensor, layer_idx: int) -> None:
        if layer_idx not in self.fc_slices:
            return
        # model.py's own layer_probe docstring requires any snapshot of the live
        # mid-graph hidden state to land in DRAM ("holding a sharded L1 copy starves
        # later programs' circular buffers") -- the earlier, validated dict-based probe
        # (test_dflash_verify.py's _probe) did this via ttnn.to_memory_config before
        # storing the tap; this accumulate-as-you-go version omitted it, silently
        # feeding a possibly-L1-sharded tensor straight into ttnn.linear. Root-caused as
        # the source of an intermittent (worse over longer generations), non-traced-AND
        # -traced correctness divergence: confirmed via extensive real-hardware testing
        # that ruled out the drafter's own K/V cache, Metal trace capture itself, and
        # general hardware/CCL non-determinism (plain decode, which never installs
        # layer_probe, was perfectly stable across repeated long runs) before landing
        # here.
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        partial = ttnn.linear(hidden_states, self.fc_slices[layer_idx])
        if self._accum is None:
            self._accum = partial
        else:
            summed = ttnn.add(self._accum, partial)
            ttnn.deallocate(self._accum)
            ttnn.deallocate(partial)
            self._accum = summed

    def finalize(self) -> ttnn.Tensor:
        assert self._accum is not None, "finalize() called before any tap()"
        accum = self._accum
        self._accum = None
        context = dflash_context_hidden_norm(self.hidden_norm, accum)
        ttnn.deallocate(accum)
        return context
