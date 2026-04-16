# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom autograd functions and RMSNorm module for Qwen3.

- ConcatLastDim: concatenate two tensors along the last dimension
- RMSNormFunction: RMSNorm via explicit rsqrt (avoids L1 overflow in
  the built-in rmsnorm_bw for 14B/32B models)
- Qwen3RMSNorm: module wrapping RMSNormFunction
- MemorySnapshotFunction / memory_snapshot: identity op with memory tracking
"""

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter


# =====================================================================
# ConcatLastDim
# =====================================================================


class ConcatLastDim(ttml.autograd.Function):
    """Concatenate two tensors along the last dimension."""

    @staticmethod
    def forward(ctx, a, b):
        a_shape = a.shape()
        b_shape = b.shape()
        ctx.save_for_backward(a_shape, b_shape)
        a_ttnn = a.get_value()
        b_ttnn = b.get_value()
        result = ttnn.concat([a_ttnn, b_ttnn], dim=-1)
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


# =====================================================================
# RMSNormFunction
# =====================================================================


class RMSNormFunction(ttml.autograd.Function):
    """RMSNorm via rsqrt: y = (x * rsqrt(mean(x^2, dim=-1) + eps)) * weight.

    Replaces rmsnorm_composite with explicit rsqrt (instead of sqrt +
    division) for direct control of forward/backward on Tenstorrent devices.

    Works with any 4-D shape as long as the last dim matches the weight.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_val = x.get_value()
        w_val = weight.get_value()

        # variance = mean(x^2, dim=-1) + eps  ->  rrms = rsqrt(variance)
        x_sq = ttnn.mul(x_val, x_val)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        variance = ttnn.add(mean_sq, eps)
        rrms = ttnn.rsqrt(variance)

        # x_hat = x * rrms,  y = x_hat * w
        x_hat = ttnn.mul(x_val, rrms)
        out = ttnn.mul(x_hat, w_val)

        ctx.save_for_backward(x_hat, rrms, w_val)
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, rrms, w_val = ctx.saved_tensors

        # grad_w: dL/dw = sum over all dims except last -> (1, 1, 1, D)
        grad_w = ttnn.mul(grad_output, x_hat)
        for d in range(3):
            grad_w = ttnn.sum(grad_w, dim=d, keepdim=True)

        # grad_x = rrms * (g*w - x_hat * mean(g*w*x_hat, dim=-1))
        gw = ttnn.mul(grad_output, w_val)
        dot = ttnn.mul(gw, x_hat)
        dot_mean = ttnn.mean(dot, dim=-1, keepdim=True)
        correction = ttnn.mul(x_hat, dot_mean)
        grad_x = ttnn.mul(ttnn.subtract(gw, correction), rrms)

        return grad_x, grad_w


# =====================================================================
# Qwen3RMSNorm module
# =====================================================================


class Qwen3RMSNorm(AbstractModuleBase):
    """RMSNorm using custom RMSNormFunction autograd op.

    Uses explicit rsqrt instead of the built-in rmsnorm op to avoid
    L1 memory overflow in backward for 14B/32B models.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        gamma_shape = (1, 1, 1, hidden_size)
        self.gamma = Parameter(ttml.init.ones()(gamma_shape))

    def forward(self, hidden_states: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return RMSNormFunction.apply(hidden_states, self.gamma.tensor, self.eps)


# =====================================================================
# MemorySnapshotFunction
# =====================================================================

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


class MemorySnapshotFunction(ttml.autograd.Function):
    """Identity op that captures a MemoryUsageTracker snapshot on forward
    and/or backward.  Zero computational overhead.
    """

    @staticmethod
    def forward(ctx, x, fwd_label, bwd_label):
        ctx.bwd_label = bwd_label
        if fwd_label:
            MemoryUsageTracker.snapshot(fwd_label)
        return x.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.bwd_label:
            MemoryUsageTracker.snapshot(ctx.bwd_label)
        return grad_output


def memory_snapshot(x, fwd_label="", bwd_label=""):
    """Identity wrapper that records memory snapshots during forward/backward."""
    if not fwd_label and not bwd_label:
        return x
    return MemorySnapshotFunction.apply(x, fwd_label, bwd_label)
