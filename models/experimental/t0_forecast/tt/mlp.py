# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN pre-norm SwiGLU MLP of t0.

reference : t0/torch/model/layers/transformer.py (``TransformerLayer.mlp``), feed_forward.py (``SwiGLU``)
    gate, up = (RMSNorm(x) * norm) @ W1 + b1 split in halves   # the GATE is the FIRST half
    return x + (silu(gate) * up) @ W2 + b2

The front half is one ``minimal_matmul(fuse_swiglu=True)``: W1 and b1 are packed into gate / up tile pairs with
``prepare_for_fused_swiglu(gate_is_first=True)``. Blocks M 8 / K 16 / N 2, subblock 2 x 2, from
``tests/perf/sweep_swiglu.py`` at M = 98304 (``doc/optimize/iter_01/swiglu_sweep*.json``): 6.35 ms against 15.07 ms
for gate + SiLU + up + MUL (accuracy), 5.50 against 14.61 ms (performance); PCC against the unfused path 0.999993.
"""

from __future__ import annotations

from models.experimental.t0_forecast.tt.common import (
    TtT0Precision,
    linear_residual,
    rms_norm,
    upload,
    upload_bias,
    upload_linear,
    upload_norm,
    upload_ones,
)
from models.experimental.t0_forecast.tt.weights import MlpWeights

# w_down on minimal_matmul with the residual add in its epilogue (R1-6); blocks from iter_01/matmul_sweep_ff2.json.
FF2_BLOCKS = (16, 8, 4, 4, 1)


class TtSwiGLU:
    def __init__(self, device, weights: MlpWeights, precision: TtT0Precision):
        self.precision = precision
        self.eps = 1e-8
        from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

        dtype, dtype_down = precision.weight_dtype("ff1"), precision.weight_dtype("ff2")
        self._norm = upload_norm(device, weights.norm)
        pack = lambda t: prepare_for_fused_swiglu(t, 1, gate_is_first=True)
        self._w_gate_up = upload(device, pack(weights.w1.T.contiguous()), dtype=dtype)
        self._b_gate_up = upload(device, pack(weights.b1.reshape(1, -1)))
        self._grid = device.compute_with_storage_grid_size()
        self._w_down = upload_linear(device, weights.w2, dtype_down)
        self._b_down = upload_bias(device, weights.b2)
        self._ones = upload_ones(device, weights.w2.shape[0])

    def __call__(self, x):
        """Flat ``(1, 1, tokens, d)`` -> ``x + mlp(x)``; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self._norm, self.eps, p)
        m_tiles = xn.padded_shape[-2] // 32
        config = ttnn.MinimalMatmulConfig(
            M_block_size=min(8, m_tiles),
            K_block_size=16,
            N_block_size=2,
            subblock_h=2 if m_tiles % 2 == 0 else 1,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(self._grid.x, self._grid.y),
        )
        hidden = ttnn.experimental.minimal_matmul(
            xn,
            self._w_gate_up,
            bias_tensor=self._b_gate_up,
            config=config,
            fuse_swiglu=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=p.compute_kernel_config(role="ff1"),
        )
        ttnn.deallocate(xn)
        y = linear_residual(hidden, self._w_down, x, self._ones, p, role="ff2", blocks=FF2_BLOCKS, bias=self._b_down)
        ttnn.deallocate(hidden)
        return y
