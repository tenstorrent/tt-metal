# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd operations for Qwen3.

ConcatLastDim: concatenation with correct backward gradient split.
"""

from __future__ import annotations

import ttnn
import ttml


class ConcatLastDim(ttml.autograd.Function):
    """Concatenate two tensors along the last dimension with correct backward split."""

    @staticmethod
    def forward(ctx, a, b):
        a_shape = a.shape()
        b_shape = b.shape()
        ctx.save_for_backward(a_shape, b_shape)
        result = ttnn.concat([a.get_value(), b.get_value()], dim=-1)
        return ttml.autograd.create_tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        a_last = a_shape[-1]
        grad_a = ttnn.slice(grad_output, [0, 0, 0, 0], [a_shape[0], a_shape[1], a_shape[2], a_last])
        grad_b = ttnn.slice(
            grad_output,
            [0, 0, 0, a_last],
            [b_shape[0], b_shape[1], b_shape[2], a_last + b_shape[-1]],
        )
        return grad_a, grad_b
