# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Profiler marker that tracks both forward and backward passes.

Usage (model-level, with backward tracking)::

    from ttml.common.profiler_utils import profiler_marker

    x = profiler_marker(x, "[Motif] after tok_emb")
    # Emits "[FWD] [Motif] after tok_emb" during forward
    # Emits "[BWD] [Motif] after tok_emb" during backward

Usage (training-loop, forward-only)::

    profiler_marker(None, "dataloader_step_done")
    # Emits "[FWD] dataloader_step_done" (no backward node)
"""

from __future__ import annotations

import ttnn

import ttml

_FLOAT_DTYPES = frozenset(
    {
        ttnn.DataType.BFLOAT16,
        ttnn.DataType.FLOAT32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.DataType.BFLOAT4_B,
    }
)


class _BackwardMarker(ttml.autograd.Function):
    """Pass-through autograd node that emits a profiler marker during backward."""

    @staticmethod
    def forward(ctx, x, marker_name, dump_results):
        ctx.marker_name = marker_name
        ctx.dump_results = dump_results
        return x.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        profiler = autograd_ctx.get_profiler()
        device = autograd_ctx.get_device()
        profiler.read_results(device, f"[BWD] {ctx.marker_name}", dump_results=ctx.dump_results)
        return grad_output


def profiler_marker(x, name, dump_results=False):
    """Insert a profiler marker that fires in both forward and backward.

    Forward: always emits ``[FWD] <name>`` via profiler.read_results.
    Backward: emits ``[BWD] <name>`` when the backward graph reaches this node.

    When ``x`` is ``None`` the marker is forward-only: no backward node is
    created and ``None`` is returned.  This is useful for training-loop
    markers (e.g. "dataloader_step_done") that don't correspond to a tensor
    flowing through the autograd graph.

    The backward node is only inserted when profiling is active to avoid
    adding autograd graph overhead during normal training.

    Args:
        x: autograd tensor to pass through unchanged, or ``None`` for a
           forward-only marker.
        name: marker name (automatically prefixed with [FWD]/[BWD]).
        dump_results: if True, flush device profiling data to disk at this
            marker.  Expensive — use sparingly to prevent device-memory
            overflow on long profiling sessions.

    Returns:
        The same tensor value (potentially wrapped in a new autograd node),
        or ``None`` when ``x`` is ``None``.
    """
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    profiler = autograd_ctx.get_profiler()
    device = autograd_ctx.get_device()

    if x is None:
        profiler.read_results(device, name, dump_results=dump_results)
        return None

    profiler.read_results(device, f"[FWD] {name}", dump_results=dump_results)

    # Integer tensors (e.g. UINT32 token IDs) don't carry gradients;
    # inserting a backward node would crash in zeros_like during backward.
    if x.get_value().dtype in _FLOAT_DTYPES:
        return _BackwardMarker.apply(x, name, dump_results)
    return x
