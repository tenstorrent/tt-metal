# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers: the compute-kernel config every matmul uses, and the exact partial-sum exchange over the
TP axis (all_gather + SFPU ttnn.add, K-30: the FPU in all_reduce reads fp32 at about TF32)."""

from __future__ import annotations

import torch
from loguru import logger

import ttnn

from . import ccl
from .settings import settings

TP_AXIS = 1  # mesh columns: heads / hidden
SP_AXIS = 0  # mesh rows: sequence
DRAM = ttnn.DRAM_MEMORY_CONFIG


def fp32_acc(needed: bool = True) -> bool:
    """Whether a program accumulates in fp32 DEST (setting FP32_ACC): all = every program (the bring-up precision,
    max precision); needed = only the sites that pass ``needed=True``: fp32 data paths (SFPU fp32 sums / Sinkhorn /
    norm statistics, whose values would be cut to bf16 in a 16-bit DEST), long reductions measured to matter and the
    softmax statistics (P.13; the list and the numbers: tt/settings.py FP32_ACC)."""
    return settings.get("FP32_ACC") == "all" or bool(needed)


def compute_config(mesh, needed: bool = True):
    """MATMUL_FIDELITY; fp32 accumulation per FP32_ACC (``needed``: this site keeps it under FP32_ACC=needed)."""
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc(needed),
        packer_l1_acc=False,
    )


def attn_compute_config(mesh, needed: bool = True):
    """The attention-side matmuls' config (P.7, owner comparison): fidelity ATTN_FIDELITY, fp32 accumulation per
    FP32_ACC (``needed`` as in compute_config). Used by the attention input projection [wq_a | wkv] (tt/q_a.py), wq_b,
    the window / sparse sparse_sdpa, wo_a, wo_b (tt/attention.py), the indexer's weights_proj / wq_b and its ring score
    (tt/indexer.py) and the compressor's projections (tt/compressor.py). mHC, norms, router, experts, shared expert and
    every non-matmul op keep compute_config (MATMUL_FIDELITY)."""
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("ATTN_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc(needed),
        packer_l1_acc=False,
    )


def bfp8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """t as ttnn's host bfloat8_b packing holds it (TILE, groups of 16 along the last dim), fp32; no device."""
    return ttnn.to_torch(ttnn.from_torch(t.float().contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


BFP8_STATS: list[dict] = []  # DENSE_WEIGHTS=bfp8: one entry per dense fp8 matrix loaded (dense_dtype), for the A/B
_LOADING = {"layer": None}  # the layer whose weights are being loaded (set_load_layer): dense_dtype's log label


def set_load_layer(layer: int | None) -> None:
    """Names the layer the next dense_dtype calls load (the block / build functions set it; logging only)."""
    _LOADING["layer"] = layer


def dense_dtype(w: torch.Tensor, label: str):
    """The device dtype of a dense weight the checkpoint stores in fp8 (e4m3 x ue8m0 per 32 x 32; setting
    DENSE_WEIGHTS): bf16 = the dequantized values exactly (e4m3 x 2^k is bf16-exact); bfp8 = bfloat8_b, the
    checkpoint's own 8-bit width (agent rule 7): 7 magnitude bits under one exponent per 16 values along the last dim,
    so a value more than 3 binades under its group's largest loses bits. ``w`` is the tensor as it will be uploaded
    (its last dim is the one bfp8 groups); with bfp8 the per-matrix change is logged (values changed, ||dW|| / ||W||)
    and kept in BFP8_STATS."""
    if settings.get("DENSE_WEIGHTS") != "bfp8":
        return ttnn.bfloat16
    wf = w.float()
    d = bfp8_roundtrip(wf).reshape(wf.shape).sub_(wf)
    layer = _LOADING["layer"]
    label = label if layer is None else f"L{layer:02d} {label}"
    st = {
        "matrix": label,
        "shape": tuple(wf.shape),
        "changed_frac": float(torch.count_nonzero(d)) / max(1, d.numel()),
        "rel": float(d.norm() / wf.norm().clamp_min(1e-30)),
    }
    BFP8_STATS.append(st)
    logger.info(
        f"DENSE_BFP8 {label} {st['shape']} values_changed={100 * st['changed_frac']:.2f} % "
        f"||dW||/||W||={st['rel']:.4g}"
    )
    return ttnn.bfloat8_b


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
    consume ``t``. gather_add: the exact fp32 sum, bit-identical on every chip (K-30), by tt/ccl.py:tp_sum (CCL_IMPL
    stock: all_gather on dim 1 + the TP slices added in mesh-column order on the SFPU; fabric: the bring-up fabric
    reduce-scatter with its exact SFPU add + fabric all_gather). all_reduce: one CCL (FPU, ~TF32)."""
    tp = mesh.shape[TP_AXIS]
    if tp == 1:
        return ttnn.clone(t, memory_config=DRAM)
    how = how or settings.get("HC_TP_SUM")
    if how == "all_reduce":
        return ttnn.all_reduce(t, cluster_axis=TP_AXIS, memory_config=DRAM)
    return ccl.tp_sum(mesh, t)


def sum_cols(mesh, t: ttnn.Tensor) -> ttnn.Tensor:
    """The exact fp32 sum of ``t`` ([1, 1, R, H] fp32 TILE) over the TP axis, kept as the chip's hidden columns
    [1, 1, R, H/TP] (the gather_add of ATTN_O_SUM / EXPERTS_SUM; tt/ccl.py:sum_cols). Does not consume ``t``."""
    if mesh.shape[TP_AXIS] == 1:
        return ttnn.clone(t, memory_config=DRAM)
    return ccl.sum_cols(mesh, t)
