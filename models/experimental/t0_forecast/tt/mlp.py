# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN pre-norm SwiGLU MLP of t0.

reference : t0/torch/model/layers/transformer.py (``TransformerLayer.mlp``), feed_forward.py (``SwiGLU``)
    gate, up = (RMSNorm(x) * norm) @ W1 + b1 split in halves   # the GATE is the FIRST half
    return x + (silu(gate) * up) @ W2 + b2
"""

from __future__ import annotations

from models.experimental.t0_forecast.tt.common import (
    TtT0Precision,
    add,
    linear,
    rms_norm,
    upload_bias,
    upload_linear,
    upload_norm,
)
from models.experimental.t0_forecast.tt.weights import MlpWeights


class TtSwiGLU:
    def __init__(self, device, weights: MlpWeights, precision: TtT0Precision):
        self.precision = precision
        self.eps = 1e-8
        dtype = precision.weight_dtype()
        h = weights.hidden
        self._norm = upload_norm(device, weights.norm)
        self._w_gate = upload_linear(device, weights.w1[:h], dtype)
        self._b_gate = upload_bias(device, weights.b1[:h])
        self._w_up = upload_linear(device, weights.w1[h:], dtype)
        self._b_up = upload_bias(device, weights.b1[h:])
        self._w_down = upload_linear(device, weights.w2, dtype)
        self._b_down = upload_bias(device, weights.b2)

    def __call__(self, x):
        """Flat ``(1, 1, tokens, d)`` -> ``x + mlp(x)``; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self._norm, self.eps, p)
        gate = linear(xn, self._w_gate, p, role="ff1", bias=self._b_gate, activation="silu")
        up = linear(xn, self._w_up, p, role="ff1", bias=self._b_up)
        ttnn.deallocate(xn)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = linear(hidden, self._w_down, p, role="ff2", bias=self._b_down)
        ttnn.deallocate(hidden)
        y = add(x, out)
        ttnn.deallocate(out)
        return y
