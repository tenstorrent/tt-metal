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
    """An fp32 matmul weight as the FPU will read it (load time, host). The FPU reads fp32 operands at TF32 (10
    mantissa bits) by truncation (K-30): a coherent -2^-11 bias over a long dot product (hc_fn, host: mixes rel
    3.5e-4, all of it bias). FP32_WEIGHT_ROUND=rne rounds the weight to TF32 to nearest-even here, so the FPU's read
    is exact and the error unbiased (host 1.5e-5; DS41-31); truncate leaves it to the FPU."""
    w = w.float().contiguous()
    if settings.get("FP32_WEIGHT_ROUND") != "rne":
        return w
    i = w.view(torch.int32)
    lsb = (i >> 13) & 1
    r = ((i + 0xFFF + lsb) >> 13) << 13
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
