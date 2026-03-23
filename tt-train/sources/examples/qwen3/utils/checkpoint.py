# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient-checkpointing utilities.

* :func:`checkpoint` — standard recomputation; saves the first autograd
  tensor and re-runs the forward pass on the backward pass.
"""

import ttml


def checkpoint(forward_fn, *args):
    """Gradient-checkpointing wrapper: trade compute for memory.

    Uses :func:`ttml.models.memory_efficient_runner` which disables gradients
    on the forward pass and recomputes activations during backward.

    Args:
        forward_fn: callable(input, mask, *extra) -> output tensor.
        *args: Positional arguments: (input, mask, …extra).

    Returns:
        Output tensor whose backward will recompute the forward pass.
    """
    input_tensor, mask, *extra = args
    return ttml.models.memory_efficient_runner(forward_fn, input_tensor, mask, *extra)
