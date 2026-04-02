# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Profiler marker that tracks both forward and backward passes.

Usage (model-level, with backward tracking)::

    from ttml.common.profiler_utils import profiler_marker

    x = profiler_marker(x, "[START] [Motif] after tok_emb")
    # Emits "[FWD] [START] [Motif] after tok_emb" during forward
    # Emits "[BWD] [START] [Motif] after tok_emb" during backward

Usage (training-loop, forward-only)::

    profiler_marker(None, "dataloader_step_done")
    # Emits "dataloader_step_done" (no backward node)
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


def profiler_marker(x, name, dump_results=False):
    """Insert a profiler marker that fires in both forward and backward.

    Forward: always emits ``[FWD] <name>`` via profiler.read_results.
    Backward: emits ``[BWD] <name>`` when the backward graph reaches this
    node.

    **Callers must use the return value** -- the returned tensor wraps the
    same device data but carries the backward marker node.  Discarding it
    means the backward marker will never fire.

    When ``x`` is ``None`` the marker is forward-only: no backward node is
    created and ``None`` is returned.  This is useful for training-loop
    markers (e.g. "dataloader_step_done") that don't correspond to a tensor
    flowing through the autograd graph.

    Args:
        x: autograd tensor to pass through, or ``None`` for a forward-only
           marker.
        name: marker name (automatically prefixed with [FWD]/[BWD]).
        dump_results: if True, flush device profiling data to disk at this
            marker.  Expensive — use sparingly to prevent device-memory
            overflow on long profiling sessions.

    Returns:
        A new autograd tensor wrapping the same data with a backward marker
        node, or ``None`` when ``x`` is ``None``.
    """
    import _ttml as cpp

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    profiler = autograd_ctx.get_profiler()
    device = autograd_ctx.get_device()

    if x is None:
        profiler.read_results(device, name, dump_results=dump_results)
        return None

    profiler.read_results(device, f"[FWD] {name}", dump_results=dump_results)

    # Integer tensors (e.g. UINT32 token IDs) don't carry gradients;
    # inserting a backward node would crash in zeros_like during backward.
    if x.get_value().dtype not in _FLOAT_DTYPES:
        return x

    grad_mode = autograd_ctx.get_gradient_mode()
    if grad_mode != cpp.autograd.GradMode.ENABLED:
        return x

    output = cpp.autograd.create_tensor(x.get_value(), requires_grad=True)

    input_tensor = x
    marker_name = name

    def backward_fn():
        ctx = ttml.autograd.AutoContext.get_instance()
        p = ctx.get_profiler()
        d = ctx.get_device()
        p.read_results(d, f"[BWD] {marker_name}", dump_results=dump_results)

        if output.is_grad_initialized():
            grad = output.get_grad()
        else:
            grad = cpp.core.zeros_like(output.get_value())

        if input_tensor.get_requires_grad():
            input_tensor.add_grad(grad)

    links = []
    input_node = x.get_node()
    if input_node is not None:
        links.append(input_node)

    node_id = autograd_ctx.add_backward_node(backward_fn, links)
    if node_id is not None:
        output.set_node(node_id)

    return output
