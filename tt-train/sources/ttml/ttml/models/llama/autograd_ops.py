# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd ops used by the Python Llama model."""

from __future__ import annotations

import ttnn
import ttml


class SliceLastDim(ttml.autograd.Function):
    """Differentiable truncation of the last dimension: ``y = x[..., :width]``."""

    @staticmethod
    def forward(ctx, x, width):
        shape = x.shape()
        ctx.full_last = shape[-1]
        ctx.width = width
        return ttnn.slice(
            x.get_value(),
            [0, 0, 0, 0],
            [shape[0], shape[1], shape[2], width],
        )

    @staticmethod
    def backward(ctx, grad_output):
        pad_amount = ctx.full_last - ctx.width
        padding = [(0, 0), (0, 0), (0, 0), (0, pad_amount)]
        return ttnn.pad(grad_output, padding=padding, value=0.0)
