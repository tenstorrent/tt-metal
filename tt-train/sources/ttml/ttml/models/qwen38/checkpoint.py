# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Activation recomputation for the Gated DeltaNet mixer.

The DeltaNet's chunked delta rule is by far the largest consumer of activation
memory in the stack.  Measured at ``seq_len=256`` on the 27B widths:

====================  ====================
component             activations retained
====================  ====================
DeltaNet block        115.3 MB
  ...of which mixer    92.0 MB
  ...of which MLP      11.4 MB
attention block        62.5 MB
====================  ====================

The mixer is 80% of a DeltaNet block, and DeltaNet is 48 of the 64 layers.  At
``seq_len=1024`` (activations scale linearly) the stack needs ~26 GB of
activations on top of ~16 GB of weights, against ~34 GB of DRAM -- it does not
fit.  Recomputing just the mixer frees ~17.7 GB and brings the total to ~24 GB.

Why only the mixer, and not the whole block
-------------------------------------------
Recomputation costs one extra forward pass of whatever is wrapped.  The MLP is
the FLOP-heavy part of a block (three 5120x17408 matmuls) but retains only
11.4 MB, so wrapping the whole block would re-run the expensive part to save
almost nothing.  Wrapping the mixer alone buys ~97% of the memory for roughly a
third of the extra compute.

How it works
------------
This mirrors ``ttml::models::common::transformer::memory_efficient_runner``
(``sources/ttml/models/common/transformer_common.hpp``), which is a C++ template
over the block's forward and so cannot be bound to Python directly.  Every
primitive it relies on is exposed, though, so the same three steps are done here:

1. run the forward with gradient mode DISABLED, so no graph is built and no
   intermediates are kept -- only the output survives;
2. register a backward that re-runs the forward on a *detached* input, this time
   with gradients enabled, rebuilding the graph on demand;
3. seed the recomputed output with the saved gradient and back-propagate through
   it, which deposits gradients into the mixer's parameters (LoRA adapters
   included) and into the detached input, whose gradient is handed back upstream.

The C++ runner also snapshots the RNG generator so dropout draws identically in
both passes.  That is unnecessary here: the mixer contains no dropout.
"""

from __future__ import annotations

import ttml
from ttml.autograd import Function

__all__ = ["RecomputeMixer", "recompute"]


class RecomputeMixer(Function):
    """Run a mixer without taping it, and rebuild the tape during backward.

    ``forward`` deliberately returns a raw ttnn tensor. ``Function.apply``
    inspects whether the outputs already carry autograd nodes; returning an
    untaped value is what tells it to install this class's ``backward`` instead
    of assuming ttml ops already built the graph.
    """

    @staticmethod
    def forward(ctx, mixer, hidden_states):
        ctx.mixer = mixer
        ctx.save_for_backward(hidden_states)

        auto = ttml.autograd.AutoContext.get_instance()
        previous = auto.get_gradient_mode()
        auto.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        try:
            out = mixer(hidden_states)
        finally:
            # Restore rather than force ENABLED: this may be nested inside an
            # inference-mode forward, where re-enabling would tape the rest.
            auto.set_gradient_mode(previous)
        return out.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        (saved,) = ctx.saved_tensors

        # Detach so the recomputed graph terminates here instead of running back
        # into the graph that produced `saved` -- that part is handled by the
        # gradient this function returns.
        detached = ttml.autograd.create_tensor(saved.get_value(), True)
        recomputed = ctx.mixer(detached)
        recomputed.set_grad(grad_output)
        recomputed.backward(False)
        return detached.get_grad()


def recompute(mixer, hidden_states):
    """Apply ``mixer`` to ``hidden_states``, recomputing it in the backward pass."""
    return RecomputeMixer.apply(mixer, hidden_states)
