# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 engram projection (reference/ds41_ref.py "engram_kv" = fp8_linear(engram_rows, engram.wkv) [S, 5H]) on
the SP x TP mesh. Adapted from tt/q_a.py (fp8 act quant -> one ttnn.linear on bf16 weights dequantized at load).

Per chip (r, c) the input is the engram rows [1, 1, S/SP, 6144] (tt/engram.py:TtEngramRows: rows split over axis 0,
replicated over axis 1; bf16 there, fp32 from the harness):

    tt/quant.fp8_qdq            act_quant_dequant(rows, 32), bit-exact (DS41-13). On real table rows (e4m3 x 2^k per
                                group) it is the identity (0 / 12.6M values changed on the layer-1 golden), except
                                groups with amax < 1e-4; kept by default (setting ENGRAM_KV_QDQ) so any input is right
    ttnn.linear                 q(rows) [S/SP, 6144] bf16 @ the chip's wkv columns [6144, 5 x H/TP] bf16 (fp8 x ue8m0
                                dequantized at load: bf16-exact), HiFi4 + fp32 accumulation, fp32 out. wkv is split by
                                output: chip column c holds its hidden slice [c H/TP, (c+1) H/TP) of each of the five
                                H-wide blocks (4 stream keys, the value), so engram_add needs no reshard (plan.md)

Output [1, 1, S/SP, 5 x H/TP] fp32 TILE: block b (0..3 = key of stream b, 4 = value) at columns [b H/TP, (b+1) H/TP),
the chip's hidden slice. No exchange, no host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, TP_AXIS, compute_config
from .layout import streams_cols_to_chip_major
from .quant import fp8_qdq
from .settings import settings

BLOCKS = 5  # hc_mult keys + one value


class TtEngramKv:
    """engram_kv of one layer. Call with the engram rows [1, 1, S/SP, K] (bf16 or fp32 TILE); returns
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
        self.qdq = settings.get("ENGRAM_KV_QDQ") == "qdq"
        w_t = wkv.float().t().contiguous()  # [K, 5H], block order
        assert torch.equal(w_t.bfloat16().float(), w_t), "fp8 weights must be bf16-exact"
        w_t = streams_cols_to_chip_major(w_t, hidden, tp, BLOCKS)  # [K, tp x 5 x H/TP]: chip column c's block
        dims = (None, 3) if TP_AXIS == 1 else (3, None)
        self.weight = ttnn.from_torch(
            w_t.reshape(1, 1, k, BLOCKS * hidden).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )

    def __call__(self, rows: ttnn.Tensor) -> ttnn.Tensor:
        assert rows.shape[-1] == self.k, (rows.shape, self.k)
        if self.qdq:
            x32 = rows if rows.dtype == ttnn.float32 else ttnn.typecast(rows, ttnn.float32, memory_config=DRAM)
            xq = fp8_qdq(x32, dtype=ttnn.bfloat16)
            if x32 is not rows:
                ttnn.deallocate(x32)
        elif rows.dtype == ttnn.bfloat16:
            xq = rows
        else:
            xq = ttnn.typecast(rows, ttnn.bfloat16, memory_config=DRAM)
        out = ttnn.linear(xq, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        if xq is not rows:
            ttnn.deallocate(xq)
        return out


def build_engram_kv(mesh, ck, cfg, layer: int) -> TtEngramKv:
    """Layer ``layer``'s engram projection (fp8 wkv dequantized at load, exact)."""
    assert cfg.is_engram(layer), f"layer {layer} has no engram"
    wkv = ck.weight(f"layers.{layer}.engram.wkv.weight", torch.float32)
    return TtEngramKv(mesh, wkv, cfg.dim)
