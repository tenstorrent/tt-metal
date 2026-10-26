# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Toto 2.0 feed-forward blocks.

reference : toto2/toto2/model.py @ 9213d2c, folded by ``tt/weights.py``
    TtSwiGLU (``GatedLinearUnitFeedForwardNetwork`` + tau residual):
        xn = RMSNorm(x);  g, u = xn Wg, xn Wu;  x <- beta * x + (g * silu(u)) Wfc2      (SiLU on the second half)
        d_ff is zero-padded to whole tiles (5464 -> 5472): the padded g, u columns are 0 and so is their product
        g * silu(u) runs as one ``minimal_matmul(fuse_swiglu=True)`` (round 1, R1-3; blocks from
        doc/optimize/iter_01/swiglu_sweep_*.json, job 720)
    TtResidualMLP (``InputResidualMLP`` / ``OutputResidualMLP``):  silu(x W1 + b1) W2 + x Ws + b
"""

from __future__ import annotations

import torch

from models.experimental.toto2_forecast.tt.common import (
    TILE,
    Toto2Precision,
    linear,
    pad_to_tile,
    rms_norm,
    upload,
    upload_bias,
    upload_linear,
)
from models.experimental.toto2_forecast.tt.attention import epilogue_config, mm_config, mm_linear_residual
from models.experimental.toto2_forecast.tt.weights import LayerWeights, ResidualMLPWeights, Toto2Config

# Fused-SwiGLU blocks (M, K, N, subblock_h, subblock_w) in tiles, by d_model; the fastest of job 720's sweep at the
# paper M (98304 tokens): 2.5B-FT 69.98 -> 19.21 ms, 313m 18.00 -> 5.69 ms per call (accuracy). K_block is cut to a
# divisor of the K tiles for the small checkpoints.
_SWIGLU_BLOCKS = {2048: (8, 32, 2, 2, 2), 1024: (8, 16, 4, 1, 4)}


def packed_fc1(w_fc1: torch.Tensor, f: int, fp: int) -> torch.Tensor:
    """``(2f, d)`` [g | u] -> the ``(d, 2fp)`` fused-SwiGLU weight: silu(u) * g per tile pair, zero d_ff padding."""
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    pad = lambda w: torch.nn.functional.pad(w, (0, 0, 0, fp - f))
    stacked = torch.cat([pad(w_fc1[:f]), pad(w_fc1[f:])]).T.contiguous()  # [g (mult) | u (silu)]
    return prepare_for_fused_swiglu(stacked, 1, gate_is_first=False)


def swiglu_config(device, d_model: int):
    import ttnn

    mb, kb, nb, sh, sw = _SWIGLU_BLOCKS.get(d_model, _SWIGLU_BLOCKS[1024])
    k_tiles = d_model // 32
    kb = max(b for b in range(1, min(kb, k_tiles) + 1) if k_tiles % b == 0)
    grid = device.compute_with_storage_grid_size()
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
    )


class TtSwiGLU:
    def __init__(self, device, lw: LayerWeights, config: Toto2Config, precision: Toto2Precision):
        self.precision = precision.for_role("ffn")
        self.eps = config.norm_eps if lw.eps_mlp is None else lw.eps_mlp
        self.beta = lw.beta_mlp
        f = config.d_ff
        fp = pad_to_tile(f)
        dtype = precision.weight_dtype("ffn")
        self._w_gu = upload(device, packed_fc1(lw.w_fc1, f, fp), dtype=dtype)
        self._swiglu_config = swiglu_config(device, config.d_model)
        self._w_down = upload_linear(device, lw.w_fc2, precision.weight_dtype("ffn_out"), pad_in=fp)
        self._down_config = mm_config(device, config.d_model, "down", k=fp)
        self._ones = upload(device, torch.ones(1, config.d_model)) if self.beta == 1.0 else None
        self._down_config_ep = epilogue_config(self._down_config, fp // TILE)

    def __call__(self, x):
        """Flat ``(1, 1, tokens, d)`` -> same; consumes ``x``."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self.eps, p)
        hidden = ttnn.experimental.minimal_matmul(
            xn,
            self._w_gu,
            config=self._swiglu_config,
            fuse_swiglu=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=p.matmul_config(),
        )
        ttnn.deallocate(xn)
        y = mm_linear_residual(hidden, self._w_down, p, (self._down_config, self._down_config_ep), x, self.beta, self._ones)
        ttnn.deallocate(hidden)
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
