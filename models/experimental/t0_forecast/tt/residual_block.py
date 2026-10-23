# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ResidualBlock (patch encoder and decoder).

reference : t0/torch/model/layers/mlp.py (``ResidualBlock``, ReLU)
    out = output(relu(hidden(x))) + residual(x)    # the two biases summed into one; ``extra_bias`` folds in
                                                   # the patch encoder's type embedding of target rows
"""

from __future__ import annotations

import torch

from models.experimental.t0_forecast.tt.common import TtT0Precision, add, linear, upload_bias, upload_linear
from models.experimental.t0_forecast.tt.weights import ResidualBlockWeights


class TtResidualBlock:
    """Weights move host -> device once in ``__init__``."""

    def __init__(
        self,
        device,
        weights: ResidualBlockWeights,
        precision: TtT0Precision,
        *,
        role: str,
        extra_bias: torch.Tensor | None = None,
        out_dtype=None,
    ):
        """``role``: the matmul role for the precision policy; ``out_dtype``: dtype of the output and residual
        linears and their sum (default bf16)."""
        self.role = role
        self.precision = precision
        self.out_dtype = out_dtype
        dtype = precision.weight_dtype()
        self._hidden_w = upload_linear(device, weights.hidden_w, dtype)
        self._hidden_b = upload_bias(device, weights.hidden_b)
        self._output_w = upload_linear(device, weights.output_w, dtype)
        bias = weights.output_b + weights.residual_b
        if extra_bias is not None:
            bias = bias + extra_bias
        self._output_b = upload_bias(device, bias)
        self._residual_w = upload_linear(device, weights.residual_w, dtype)

    def __call__(self, x):
        """Device ``(..., in)`` -> ``(..., out)``; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        hidden = linear(x, self._hidden_w, p, role=self.role, bias=self._hidden_b, activation="relu")
        out = linear(hidden, self._output_w, p, role=self.role, bias=self._output_b, dtype=self.out_dtype)
        ttnn.deallocate(hidden)
        skip = linear(x, self._residual_w, p, role=self.role, dtype=self.out_dtype)
        y = add(out, skip)
        ttnn.deallocate(out)
        ttnn.deallocate(skip)
        return y
