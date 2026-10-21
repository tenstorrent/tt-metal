# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd ops used by the Python Llama model."""

from __future__ import annotations

import ttnn
import ttml


class Truncate(ttml.autograd.Function):
    """Differentiable truncation of a 4-D tensor's dimension ``dim``: its first ``width`` entries."""

    @staticmethod
    def forward(ctx, x, dim, width):
        shape = list(x.shape())
        ctx.dim, ctx.pad_amount = dim, shape[dim] - width
        end = list(shape)
        end[dim] = width
        return ttnn.slice(x.get_value(), [0, 0, 0, 0], end)

    @staticmethod
    def backward(ctx, grad_output):
        padding = [(0, 0)] * 4
        padding[ctx.dim] = (0, ctx.pad_amount)
        return ttnn.pad(grad_output, padding=padding, value=0.0)
