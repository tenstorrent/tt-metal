# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers: the compute-kernel config every matmul uses, and the exact partial-sum exchange over the
TP axis (all_gather + SFPU ttnn.add, K-30: the FPU in all_reduce reads fp32 at about TF32)."""

from __future__ import annotations

import torch

import ttnn

from .settings import settings

TP_AXIS = 1  # mesh columns: heads / hidden
SP_AXIS = 0  # mesh rows: sequence
DRAM = ttnn.DRAM_MEMORY_CONFIG


def compute_config(mesh):
    """HiFi4 + fp32 accumulation by default (owner rule 7; tt/settings.py MATMUL_FIDELITY / FP32_DEST_ACC)."""
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=bool(settings.get("FP32_DEST_ACC")),
        packer_l1_acc=False,
    )


def fpu_weight(w: torch.Tensor) -> torch.Tensor:
    """An fp32 matmul weight (matmul in1) as the FPU will multiply it (load time, host). The unpacker reads fp32 at
    TF32 (10 mantissa bits) by truncation (K-30), and at HiFi4 the FPU multiplies only 9 explicit mantissa bits of
    in1 (srcA: 2 fidelity phases x 5 bits incl. the hidden bit; in0 / srcB keeps all 10): a coherent truncation bias
    over a long dot product (hc_fn: mixes -3.46e-4 with a TF32 weight, DS41-42). FP32_WEIGHT_ROUND rounds the weight
    to nearest-even at load so what the FPU multiplies is exact and the error unbiased: srca = to the 9 bits srcA
    keeps at HiFi4 (mixes +3.0e-5), rne = to TF32 (its 10th bit is still truncated on the device); truncate leaves
    it all to the FPU."""
    w = w.float().contiguous()
    mode = settings.get("FP32_WEIGHT_ROUND")
    if mode == "truncate":
        return w
    sh = 23 - {"srca": 9, "rne": 10}[mode]  # dropped low mantissa bits
    i = w.view(torch.int32)
    lsb = (i >> sh) & 1
    r = ((i + ((1 << (sh - 1)) - 1) + lsb) >> sh) << sh
    return torch.where(torch.isfinite(w), r.view(torch.float32), w)


def tp_sum(mesh, t: ttnn.Tensor, how: str | None = None) -> ttnn.Tensor:
    """Sum ``t`` ([1, 1, R, W], fp32 TILE) over the TP axis, the result replicated on every chip of the row. Does not
    consume ``t``. gather_add: all_gather on dim 1 -> [1, TP, R, W], then the TP slices added in mesh-column order
    on the SFPU, so every chip computes the bit-identical fp32 sum (K-30). all_reduce: one CCL (FPU, ~TF32)."""
    tp = mesh.shape[TP_AXIS]
    if tp == 1:
        return ttnn.clone(t, memory_config=DRAM)
    how = how or settings.get("HC_TP_SUM")
    if how == "all_reduce":
        return ttnn.all_reduce(t, cluster_axis=TP_AXIS, memory_config=DRAM)
    r, w = t.shape[-2], t.shape[-1]
    g = ttnn.all_gather(t, dim=1, cluster_axis=TP_AXIS, memory_config=DRAM)
    acc = ttnn.slice(g, [0, 0, 0, 0], [1, 1, r, w])
    for c in range(1, tp):
        part = ttnn.slice(g, [0, c, 0, 0], [1, c + 1, r, w])
        nxt = ttnn.add(acc, part, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(acc)
        ttnn.deallocate(part)
        acc = nxt
    ttnn.deallocate(g)
    return acc
