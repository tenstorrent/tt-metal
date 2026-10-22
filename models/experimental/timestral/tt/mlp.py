# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN pre-norm SwiGLU MLP (no residual; the block adds it).

reference : models/experimental/timestral/reference/src/timestral/layers.py
    gate, up = w12(RMSNorm(x)).chunk(2)   # gain folded into both halves
    return w3(gate * silu(up))            # gate * silu(up) is one fused minimal_matmul
"""

from __future__ import annotations

from models.experimental.timestral.tt.common import (
    TtTimestralPrecision,
    linear,
    rms_norm,
    swiglu,
    upload_linear,
    upload_swiglu,
)
from models.experimental.timestral.tt.weights import MlpWeights


class TtSwiGLU:
    def __init__(self, device, weights: MlpWeights, precision: TtTimestralPrecision):
        self.precision = precision
        self.eps = weights.eps
        dtype = precision.weight_dtype()
        self._w_up_gate = upload_swiglu(device, weights.w_up, weights.w_gate, dtype)
        self._w_down = upload_linear(device, weights.w_down, dtype)

    def __call__(self, x):
        """Flat device ``(1, 1, tokens, D)`` -> same shape; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self.eps, p)
        hidden = swiglu(xn, self._w_up_gate, p)
        ttnn.deallocate(xn)
        out = linear(hidden, self._w_down, p)
        ttnn.deallocate(hidden)
        return out
