# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 engram projection (reference/ds41_ref.py "engram_kv" = fp8_linear(engram_rows, engram.wkv) [S, 5H]) on
the SP x TP mesh. Adapted from tt/q_a.py (bf16 rows -> one ttnn.linear on the weights dequantized at load).

Per chip (r, c) the input is the engram rows [1, 1, S/SP, 6144] (tt/engram.py:TtEngramRows: rows split over axis 0,
replicated over axis 1; bfloat8_b with ENGRAM_ROWS=bfp8, else bf16; fp32 from the harness):

    gemm_input                  GEMM_INPUTS (FIX.7, F208): bfloat8_b, the device's form of the released fp8_linear's
                                act quant of the rows; rows that arrive as bfloat8_b are read as they are
    ttnn.linear                 rows [S/SP, 6144] @ the chip's wkv columns [6144, 5 x H/TP] (DENSE_WEIGHTS),
                                MATMUL_FIDELITY + fp32 accumulation, fp32 out. wkv is split by
                                output: chip column c holds its hidden slice [c H/TP, (c+1) H/TP) of each of the five
                                H-wide blocks (4 stream keys, the value), so engram_add needs no reshard (plan.md)

Output [1, 1, S/SP, 5 x H/TP] fp32 TILE: block b (0..3 = key of stream b, 4 = value) at columns [b H/TP, (b+1) H/TP),
the chip's hidden slice. No exchange, no host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

from . import mm_configs as mm
from .ccl import dealloc
from .common import DRAM, TP_AXIS, act_dtype, compute_config, dense_dtype, gemm_input
from .cores import from_torch
from .layout import streams_cols_to_chip_major

BLOCKS = 5  # hc_mult keys + one value


class TtEngramKv:
    """engram_kv of one layer. Call with the engram rows [1, 1, S/SP, K] (bf16, fp32 or bfloat8_b TILE); returns
    [1, 1, S/SP, 5 x H/TP] fp32 TILE. Does not consume its input."""

    def __init__(self, mesh, wkv: torch.Tensor, hidden: int):
        n, k = wkv.shape  # [5H, K] (nn.Linear layout)
        tp = mesh.shape[TP_AXIS]
        assert n == BLOCKS * hidden, (wkv.shape, hidden)
        assert hidden % (32 * tp) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        assert k % 32 == 0, wkv.shape
        self.mesh, self.hidden, self.k = mesh, hidden, k
        self.width = BLOCKS * hidden // tp
        self.ckc = compute_config(mesh)
        w_t = wkv.float().t().contiguous()  # [K, 5H], block order
        assert torch.equal(w_t.bfloat16().float(), w_t), "fp8 weights must be bf16-exact"
        w_t = streams_cols_to_chip_major(w_t, hidden, tp, BLOCKS)  # [K, tp x 5 x H/TP]: chip column c's block
        dims = (None, 3) if TP_AXIS == 1 else (3, None)
        w_t = w_t.reshape(1, 1, k, BLOCKS * hidden).contiguous()
        self.weight = from_torch(
            w_t,
            dtype=dense_dtype(w_t, "engram.wkv"),  # DENSE_WEIGHTS (P.13): bf16 (exact) or bfloat8_b
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )

    def __call__(self, rows: ttnn.Tensor) -> ttnn.Tensor:
        assert rows.shape[-1] == self.k, (rows.shape, self.k)
        xq = gemm_input(rows)  # GEMM_INPUTS (FIX.7): bfloat8_b (ENGRAM_ROWS=bfp8: arrives in it, no typecast)
        # BF16_ACTIVATIONS=on (P.20): written bf16 (the GPU code's fp8 GEMM output), the engram add reads it as is
        out = mm.linear(
            xq, self.weight, site="engram_kv", dtype=act_dtype(), compute_kernel_config=self.ckc, memory_config=DRAM
        )
        if xq is not rows:
            dealloc(xq)
        return out


def build_engram_kv(mesh, ck, cfg, layer: int) -> TtEngramKv:
    """Layer ``layer``'s engram projection (fp8 wkv dequantized at load)."""
    assert cfg.is_engram(layer), f"layer {layer} has no engram"
    wkv = ck.weight(f"layers.{layer}.engram.wkv.weight", torch.float32)
    return TtEngramKv(mesh, wkv, cfg.dim)
