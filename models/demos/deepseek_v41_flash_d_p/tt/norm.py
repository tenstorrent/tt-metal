# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 distributed RMSNorm (attn_norm / ffn_norm / final norm; reference/ds41_ref.py:rms_norm, inference
RMSNorm) on the SP x TP mesh, fp32 throughout:

    out [S, H] = w * (x * rsqrt(mean(x^2) + eps))          eps = norm_eps (1e-20)

Per chip (r, c) the input is the chip's rows and hidden columns [1, 1, S/SP, H/TP] fp32 TILE (tt/collapse.py output);
the output has the same split. Composition (all fp32 on the SFPU, so every product and sum is an exact fp32 operation;
K-30: the FPU, which rms_norm_pre/post_all_gather use, reads fp32 at about TF32; K-12: those ops also leave junk in the
stats tile on an fp32 input):

    ttnn.bringup.mhc_pre_xing_pack(n=1)   the chip's exact sum x^2 over its H/TP columns -> column 3 of a [S/SP, 32]
                                          row (the op overwrites that column of the ``mix`` it is given; any fp32
                                          [S/SP, 32] tile does, so x's first 32 columns serve, no per-call constant)
    slice column 3 -> tt/common.py:tp_sum all_gather over axis 1 + ttnn.add in mesh-column order (exact, bit-identical
                                          on every chip of the row)
    x (1/H) + eps -> ttnn.rsqrt           [S/SP, 1]
    x * rstd ([S/SP, 1] column broadcast) * w ([1, H/TP] row broadcast; the chip's slice of w, fp32, built at load)

The fp8 act quant of the output (blocks of 32) belongs to the fp8 linears that read it (q_a / kv: reference
fp8_linear quantizes its input); the compressor and the indexer's weights_proj read this fp32 output as is.

ffn_norm (``gather=True``) adds one ttnn.all_gather (dim 3, cluster_axis 1; a data move, exact) of the fp32 output
-> [1, 1, S/SP, H] fp32 per chip, replicated over the mesh row: the router reads these whole rows unquantized, and
``quant(y)`` (tt/quant.py:fp8_qdq, blocks of 32 along H, bf16 exact) gives q(x), which the shared and the routed
experts both read (reference _swiglu act_quant's the same ffn_norm output), computed once per layer.
No host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, TP_AXIS, tp_sum
from .quant import fp8_qdq

PACK_N = 1  # mhc_pre_xing_pack with one "stream": the sum of squares lands in column n (n + 2) = 3
SS_COL = PACK_N * (PACK_N + 2)


class TtDistributedRmsNorm:
    """One column-split RMSNorm. Call with the per-chip [1, 1, S/SP, H/TP] fp32 TILE; returns [1, 1, S/SP, H/TP]
    fp32 TILE (same split). Does not consume its input. ``split=False``: the rows are whole on every chip ([.., H],
    replicated over TP, e.g. q_norm / kv_norm): the same chain with a replicated weight and no exchange."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float, split: bool = True, gather: bool = False):
        assert split or not gather, "gather needs a column-split norm"
        hidden = weight.numel()
        self.gather = gather and mesh.shape[TP_AXIS] > 1
        tp = mesh.shape[TP_AXIS] if split else 1
        assert hidden % (32 * tp) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        self.mesh, self.hidden, self.eps, self.split, self.tp = mesh, hidden, float(eps), split, tp
        self.inv_h = 1.0 / float(hidden)
        # w as [1, 1, 1, H] fp32 TILE, the last dim split over the TP axis (chip column c holds w[c H/TP, (c+1) H/TP)),
        # replicated over the SP axis; split=False: replicated everywhere.
        dims = ((None, 3) if TP_AXIS == 1 else (3, None)) if split else (None, None)
        self.weight = ttnn.from_torch(
            weight.float().reshape(1, 1, 1, hidden).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )

    def rstd(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, R, H/TP] fp32 -> rsqrt(mean over the full row of x^2 + eps) [1, 1, R, 1] fp32, replicated over TP."""
        rows = x.shape[-2]
        mix = ttnn.slice(x, [0, 0, 0, 0], [1, 1, rows, 32], memory_config=DRAM)
        packed = ttnn.bringup.mhc_pre_xing_pack(mix, x, n=PACK_N)
        ttnn.deallocate(mix)
        part = ttnn.slice(packed, [0, 0, 0, SS_COL], [1, 1, rows, SS_COL + 1], memory_config=DRAM)
        ttnn.deallocate(packed)
        if self.split:
            ss = tp_sum(self.mesh, part)
            ttnn.deallocate(part)
        else:
            ss = part
        ms = ttnn.multiply(ss, self.inv_h, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(ss)
        t = ttnn.add(ms, self.eps, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(ms)
        rs = ttnn.rsqrt(t, memory_config=DRAM)
        ttnn.deallocate(t)
        return rs

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert x.dtype == ttnn.float32, x.dtype
        assert x.shape[-1] * self.tp == self.hidden, (x.shape, self.hidden)
        rs = self.rstd(x)
        xn = ttnn.multiply(x, rs, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(rs)
        y = ttnn.multiply(xn, self.weight, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(xn)
        if not self.gather:
            return y
        # ffn_norm: whole token rows on every chip of the mesh row (router, dispatch, shared expert); chip column c's
        # slice lands at columns [c H/TP, (c+1) H/TP) (mesh-column order), so the rows are the reference's.
        full = ttnn.all_gather(y, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        ttnn.deallocate(y)
        return full

    @staticmethod
    def quant(y: ttnn.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        """q(y): the fp8 act quant-dequant (blocks of 32, ue8m0, e4m3 RNE) the experts read; bit-exact (DS41-13)."""
        return fp8_qdq(y, dtype=dtype)


# norm steps -> checkpoint weight under layers.<i>.
NORM_WEIGHTS = {"attn_norm": "attn_norm.weight", "ffn_norm": "ffn_norm.weight"}
# norm steps whose output is gathered over the mesh columns (whole rows, replicated within a mesh row).
GATHERED_NORMS = {"ffn_norm"}


def build_norm(mesh, ck, cfg, layer: int, step: str) -> TtDistributedRmsNorm:
    w = ck.weight(f"layers.{layer}.{NORM_WEIGHTS[step]}", torch.float32)
    return TtDistributedRmsNorm(mesh, w, cfg.eps, gather=step in GATHERED_NORMS)
