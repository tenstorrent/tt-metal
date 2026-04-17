# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm for Qwen3.

Replaces ``ttml.ops.rmsnorm.rmsnorm`` (composite) with an explicit rsqrt-based
autograd function. The composite backward blows past L1 on 14B/32B configs —
see the error note on ``Qwen3RMSNorm.forward`` below for the exact symptom.

Works with any 4-D shape as long as the last dim matches the weight.
"""

from __future__ import annotations

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter


class RMSNormFunction(ttml.autograd.Function):
    """RMSNorm via rsqrt: ``y = (x · rsqrt(mean(x², dim=-1) + eps)) · weight``."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_val = x.get_value()
        w_val = weight.get_value()

        # variance = mean(x², dim=-1) + eps  →  rrms = rsqrt(variance)
        x_sq = ttnn.mul(x_val, x_val)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)  # (B, 1, S, 1)
        variance = ttnn.add(mean_sq, eps)  # (B, 1, S, 1)
        rrms = ttnn.rsqrt(variance)  # (B, 1, S, 1)

        # x_hat = x · rrms,  y = x_hat · w
        x_hat = ttnn.mul(x_val, rrms)  # (B, 1, S, D)
        out = ttnn.mul(x_hat, w_val)  # (B, 1, S, D)

        ctx.save_for_backward(x_hat, rrms, w_val)
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, rrms, w_val = ctx.saved_tensors

        # grad_w = Σ (g · x_hat) over all dims except last  →  (1, 1, 1, D)
        grad_w = ttnn.mul(grad_output, x_hat)
        for d in range(3):
            grad_w = ttnn.sum(grad_w, dim=d, keepdim=True)

        # grad_x = rrms · (g⊙w − x_hat · mean(g⊙w⊙x_hat, dim=-1))
        gw = ttnn.mul(grad_output, w_val)  # (B, 1, S, D)
        dot = ttnn.mul(gw, x_hat)  # (B, 1, S, D)
        dot_mean = ttnn.mean(dot, dim=-1, keepdim=True)  # (B, 1, S, 1)
        correction = ttnn.mul(x_hat, dot_mean)  # (B, 1, S, D)
        grad_x = ttnn.mul(ttnn.subtract(gw, correction), rrms)  # (B, 1, S, D)

        return grad_x, grad_w


class Qwen3RMSNorm(AbstractModuleBase):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, hidden_size)))

    def forward(self, hidden_states: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        # Custom autograd is required on 14B/32B — the composite backward
        # (ttml::metal::rmsnorm_bw) allocates CBs that exceed L1:
        #   "Statically allocated circular buffers on core range
        #    [(x=0,y=0) - (x=4,y=3)] grow to 1764640 B which is beyond max L1
        #    size of 1499136 B"
        return RMSNormFunction.apply(hidden_states, self.weight.tensor, self.eps)
