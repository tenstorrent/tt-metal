# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient-checkpointing utilities.

Provides two checkpointing strategies:

* :func:`checkpoint` — standard recomputation; saves the first autograd
  tensor and re-runs the forward pass on the backward pass.

* :func:`checkpoint_scattered` — like :func:`checkpoint` but also scatters
  the saved activation across TP devices so that each device only keeps
  ``1/tp_size`` of the tensor, reducing peak activation memory.
"""

import ttnn
import ttml

from utils.tensor_utils import get_tp_size


# ---------------------------------------------------------------------------
# CheckpointFunction & checkpoint
# ---------------------------------------------------------------------------


class CheckpointFunction(ttml.autograd.Function):
    """Gradient checkpointing as a ``ttml.autograd.Function``.

    Forward:  run ``forward_fn(*args)`` with gradients **disabled** — no
              intermediate activations or backward graph are stored.
    Backward: re-run ``forward_fn`` with gradients **enabled** (rebuilds the
              local sub-graph), inject the upstream gradient, back-propagate,
              and return the input gradients.

    Usage via the convenience wrapper::

        output = checkpoint(layer, hidden_states, attention_mask)
    """

    @staticmethod
    def forward(ctx, forward_fn, *args):
        ctx.forward_fn = forward_fn
        ctx.args = args
        ctx.num_tensor_args = sum(1 for a in args if hasattr(a, "get_requires_grad"))

        auto_ctx = ttml.autograd.AutoContext.get_instance()
        prev_mode = auto_ctx.get_gradient_mode()
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        try:
            out = forward_fn(*args)
        finally:
            auto_ctx.set_gradient_mode(prev_mode)

        return out.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        new_args = []
        detached_input = None
        first_tensor_found = False
        for a in ctx.args:
            if not first_tensor_found and hasattr(a, "get_requires_grad"):
                detached_input = ttml.autograd.create_tensor(a.get_value())
                new_args.append(detached_input)
                first_tensor_found = True
            else:
                new_args.append(a)

        recomputed = ctx.forward_fn(*new_args)

        recomputed.set_grad(grad_output)
        recomputed.backward(False)

        input_grad = detached_input.get_grad()
        if ctx.num_tensor_args <= 1:
            return input_grad
        return (input_grad,) + (None,) * (ctx.num_tensor_args - 1)


def checkpoint(forward_fn, *args):
    """Gradient-checkpointing wrapper: trade compute for memory.

    Args:
        forward_fn: callable(*args) -> output tensor (e.g. a decoder layer).
        *args: Arguments forwarded to *forward_fn*.  The first ttml autograd
               tensor is the differentiable input; the rest are treated as
               constants (no gradient returned).

    Returns:
        Output tensor whose backward will recompute the forward pass.
    """
    return CheckpointFunction.apply(forward_fn, *args)


# ---------------------------------------------------------------------------
# checkpoint_scattered
# ---------------------------------------------------------------------------


_scatter_fallback_warned = False


def checkpoint_scattered(forward_fn, scatter_dim, shard_dim, *args):
    """Gradient-checkpointing with scattered intermediates.

    Like :func:`checkpoint` but scatters the saved hidden state across TP
    devices to reduce per-device activation memory by ``tp_size``.

    Instead of using ``Function.apply()`` (which captures full input tensors
    in its backward closure, defeating the scatter optimisation), this
    function directly manipulates the autograd graph — mirroring the C++
    ``memory_efficient_runner`` pattern in ``transformer_common.hpp``.

    After the forward pass the first tensor arg's device-memory value is
    **replaced in-place** with the scattered shard via ``set_value()``.
    The full tensor is recovered via all-gather only when backward runs.

    When ``args[0].shape[scatter_dim] % tp_size != 0``, transparently
    falls back to regular checkpoint behaviour (saves full tensor, prints
    a one-time warning).

    Args:
        forward_fn: callable(*args) -> output tensor (e.g. a decoder layer).
        scatter_dim: Dimension along which to scatter the saved input
            (typically 0 for batch when ``batch_size % tp_size == 0``).
        shard_dim: Mesh dimension for TP communication (same as the model's
            ``shard_dim``).
        *args: Arguments forwarded to *forward_fn*.  The first ttml autograd
               tensor is scattered; the rest are saved as-is.

    Returns:
        Output tensor whose backward will all-gather + recompute.
    """
    auto_ctx = ttml.autograd.AutoContext.get_instance()

    if auto_ctx.get_gradient_mode() == ttml.autograd.GradMode.DISABLED:
        return forward_fn(*args)
    # The rest assumes gradient mode is enabled

    # --- forward with gradients disabled (no intermediate activations) ---
    auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        out = forward_fn(*args)
    finally:
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)

    # Locate the first autograd tensor in args (the differentiable input).
    first_tensor = None
    first_idx = -1
    for i, a in enumerate(args):
        if hasattr(a, "get_requires_grad"):
            first_tensor = a
            first_idx = i
            break

    # Determine whether we can scatter along the requested dim.
    global _scatter_fallback_warned
    tp_size = get_tp_size(shard_dim)
    can_scatter = (
        first_tensor is not None and first_tensor.shape()[scatter_dim] % tp_size == 0
    )

    if first_tensor is not None and not can_scatter and not _scatter_fallback_warned:
        dim_size = first_tensor.shape()[scatter_dim]
        print(
            f"  [scatter_intermediates] dim {scatter_dim} "
            f"(size={dim_size}) not divisible by "
            f"tp_size={tp_size}, keeping full tensor "
            f"(no memory saving for this call)"
        )
        _scatter_fallback_warned = True

    if can_scatter:
        # Scatter the value, replace it in-place, and explicitly free the
        # old full-size device buffer.  set_value() alone is not enough
        # because tt::tt_metal::Tensor uses shared storage — any shallow
        # copy (e.g. the temp autograd tensor) keeps the buffer alive.
        # ttnn.deallocate() forces the shared device buffer free.
        #
        # Gradients MUST be disabled for the helper scatter/all-gather ops,
        # otherwise they register orphan backward nodes in the global graph
        # whose closures keep full-size tensors alive until reset_graph().
        full_value = first_tensor.get_value()
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        temp = ttml.autograd.create_tensor(full_value, requires_grad=False)
        scattered = ttml.ops.distributed.scatter(temp, scatter_dim, shard_dim)
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        first_tensor.set_value(scattered.get_value())
        del temp, scattered
        ttnn.deallocate(full_value)
        del full_value

    # --- register backward node directly (no Function.apply overhead) ---
    def _backward():
        if can_scatter:
            # Reconstruct full input from the scattered shard.
            # Disable gradients to avoid creating orphan graph nodes.
            auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
            temp = ttml.autograd.create_tensor(
                first_tensor.get_value(), requires_grad=False
            )
            gathered = ttml.ops.distributed.all_gather(temp, scatter_dim, shard_dim)
            auto_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            full_value = gathered.get_value()
            # Restore full value so that add_grad() shape check passes.
            first_tensor.set_value(full_value)
            input_detached = ttml.autograd.create_tensor(full_value)
            del temp, gathered, full_value
        else:
            input_detached = ttml.autograd.create_tensor(first_tensor.get_value())

        # Rebuild args with the detached copy so the recomputed graph
        # is isolated from the outer graph.
        new_args = list(args)
        new_args[first_idx] = input_detached

        # Re-run forward WITH gradients (builds local backward graph).
        recomputed = forward_fn(*new_args)
        recomputed.set_grad(out.get_grad())
        recomputed.backward(False)

        first_tensor.add_grad(input_detached.get_grad())

        if can_scatter:
            # Re-scatter and deallocate the full value so it doesn't stay
            # alive via the previous layer's backward closure (which
            # captures this tensor as `out` but only reads its gradient).
            full_val = first_tensor.get_value()
            auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
            temp = ttml.autograd.create_tensor(full_val, requires_grad=False)
            scattered = ttml.ops.distributed.scatter(temp, scatter_dim, shard_dim)
            auto_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            first_tensor.set_value(scattered.get_value())
            del temp, scattered
            ttnn.deallocate(full_val)
            del full_val

    links = ttml.autograd.get_links([first_tensor])
    out.set_node(auto_ctx.add_backward_node(_backward, links))
    return out
