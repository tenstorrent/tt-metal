# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ResidualBlock (patch embedding and quantile head).

reference : models/experimental/timestral/reference/src/timestral/layers.py
    out = output(silu(hidden(x)))   # silu fused into the hidden matmul
    return out + residual(x)
"""

from __future__ import annotations

from models.experimental.timestral.tt.common import (
    TtTimestralPrecision,
    add_residual,
    linear,
    upload_bias,
    upload_linear,
)
from models.experimental.timestral.tt.weights import ResidualBlockWeights


class TtResidualBlock:
    """Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: ResidualBlockWeights, precision: TtTimestralPrecision):
        self.precision = precision
        dtype = precision.weight_dtype()
        self.in_features = weights.hidden_w.shape[1]
        self.out_features = weights.output_w.shape[0]
        self._hidden_w = upload_linear(device, weights.hidden_w, dtype)
        self._hidden_b = upload_bias(device, weights.hidden_b)
        self._output_w = upload_linear(device, weights.output_w, dtype)
        self._output_b = upload_bias(device, weights.output_b)
        self._residual_w = upload_linear(device, weights.residual_w, dtype)
        self._residual_b = upload_bias(device, weights.residual_b)

    def __call__(self, x):
        """Device ``(..., in)`` -> device ``(..., out)``; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        hidden = linear(x, self._hidden_w, p, bias=self._hidden_b, activation="silu")
        main = linear(hidden, self._output_w, p, bias=self._output_b)
        ttnn.deallocate(hidden)
        skip = linear(x, self._residual_w, p, bias=self._residual_b)
        return add_residual(main, skip)
