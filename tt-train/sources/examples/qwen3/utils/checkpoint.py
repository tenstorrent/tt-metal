# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient-checkpointing utilities.

* :func:`checkpoint` — standard recomputation; saves the first autograd
  tensor and re-runs the forward pass on the backward pass.
"""

import ttml


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
