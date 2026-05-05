# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd operations for Qwen3.

ConcatLastDim: concatenation with correct backward gradient split.
RMSNormFunction: explicit rsqrt-based RMSNorm that works around L1 size
limits on Tenstorrent devices for large hidden dimensions.
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


class RMSNormFunction(ttml.autograd.Function):
    """RMSNorm via explicit rsqrt for direct control of forward/backward.

    Works around L1 size limits that the fused rmsnorm_bw hits on large
    hidden dimensions (14B/32B scale).
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_val = x.get_value()
        w_val = weight.get_value()

        x_sq = ttnn.mul(x_val, x_val)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        variance = ttnn.add(mean_sq, eps)
        rrms = ttnn.rsqrt(variance)

        x_hat = ttnn.mul(x_val, rrms)
        out = ttnn.mul(x_hat, w_val)

        ctx.save_for_backward(x_hat, rrms, w_val)
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, rrms, w_val = ctx.saved_tensors

        grad_w = ttnn.mul(grad_output, x_hat)
        for d in range(3):
            grad_w = ttnn.sum(grad_w, dim=d, keepdim=True)

        gw = ttnn.mul(grad_output, w_val)
        dot = ttnn.mul(gw, x_hat)
        dot_mean = ttnn.mean(dot, dim=-1, keepdim=True)
        correction = ttnn.mul(x_hat, dot_mean)
        grad_x = ttnn.mul(ttnn.subtract(gw, correction), rrms)

        return grad_x, grad_w
