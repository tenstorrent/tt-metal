# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Toto 2.0 feed-forward blocks.

reference : toto2/toto2/model.py @ 9213d2c, folded by ``tt/weights.py``
    TtSwiGLU (``GatedLinearUnitFeedForwardNetwork`` + tau residual):
        xn = RMSNorm(x);  g, u = xn Wg, xn Wu;  x <- beta * x + (g * silu(u)) Wfc2      (SiLU on the second half)
        d_ff is zero-padded to whole tiles (5464 -> 5472): the padded g, u columns are 0 and so is their product
    TtResidualMLP (``InputResidualMLP`` / ``OutputResidualMLP``):  silu(x W1 + b1) W2 + x Ws + b
"""

from __future__ import annotations

from models.experimental.toto2_forecast.tt.common import (
    Toto2Precision,
    linear,
    pad_to_tile,
    rms_norm,
    scaled_residual,
    upload_bias,
    upload_linear,
)
from models.experimental.toto2_forecast.tt.weights import LayerWeights, ResidualMLPWeights, Toto2Config


class TtSwiGLU:
    def __init__(self, device, lw: LayerWeights, config: Toto2Config, precision: Toto2Precision):
        self.precision = precision
        self.eps = config.norm_eps
        self.beta = lw.beta_mlp
        f = config.d_ff
        fp = pad_to_tile(f)
        dtype = precision.weight_dtype("ffn")
        self._w_g = upload_linear(device, lw.w_fc1[:f], dtype, pad_out=fp)
        self._w_u = upload_linear(device, lw.w_fc1[f:], dtype, pad_out=fp)
        self._w_down = upload_linear(device, lw.w_fc2, dtype, pad_in=fp)

    def __call__(self, x):
        """Flat ``(1, 1, tokens, d)`` -> same; consumes ``x``."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self.eps, p)
        g = linear(xn, self._w_g, p)
        u = linear(xn, self._w_u, p, activation="silu")
        ttnn.deallocate(xn)
        hidden = ttnn.mul(g, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = linear(hidden, self._w_down, p)
        ttnn.deallocate(hidden)
        y = scaled_residual(out, x, self.beta)
        ttnn.deallocate(x)
        return y


class TtResidualMLP:
    def __init__(self, device, w: ResidualMLPWeights, precision: Toto2Precision, *, out_dtype=None):
        """``out_dtype``: of the output (default bf16)."""
        self.precision = precision
        self.out_dtype = out_dtype
        dtype = precision.weight_dtype()
        self._w1 = upload_linear(device, w.w1, dtype)
        self._b1 = upload_bias(device, w.b1)
        self._w2 = upload_linear(device, w.w2, dtype)
        self._ws = upload_linear(device, w.ws, dtype)
        self._b = upload_bias(device, w.b)

    def __call__(self, x):
        """Device ``(1, 1, tokens, in)`` -> ``(1, 1, tokens, out)``; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        out_dtype = self.out_dtype or ttnn.bfloat16
        hidden = linear(x, self._w1, p, bias=self._b1, activation="silu", dtype=ttnn.bfloat16)
        main = linear(hidden, self._w2, p, dtype=out_dtype)
        ttnn.deallocate(hidden)
        skip = linear(x, self._ws, p, bias=self._b, dtype=out_dtype)
        out = ttnn.add(main, skip, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(main)
        ttnn.deallocate(skip)
        return out
