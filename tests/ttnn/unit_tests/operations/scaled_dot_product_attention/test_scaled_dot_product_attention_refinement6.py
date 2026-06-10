# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 6 unit tests — S=8192 fp32 precision lift via UnpackToDestFp32
on running-state CBs + untagged-intermediate FPU final divide.

The two named failing cells from op_requirements.md are:

    Q1x1x8192x64 fp32 self mha none × {auto, explicit}

Post-R4 / R5 measurement: PCC=0.999631, RMS=0.0272 vs target (0.999, 0.02)
— PCC met, RMS over by 0.007. The residual was the per-K-iter SFPU unpack
of cb_cur_sum_exp / cb_cur_mm_out routing through srcA/srcB TF32 (~10
mantissa bits) — for Kt=256 the sqrt(Kt) × TF32-ULP floor lands at the
observed 0.027 RMS.

R6 closes it by tagging cb_cur_sum_exp and cb_cur_mm_out with
UnpackToDestFp32 (preserves the full 24-bit FP32 mantissa across the
per-iter SFPU reload) and introducing untagged intermediate CBs
(cb_cur_sum_exp_for_divide, cb_cur_mm_out_for_divide) for the final FPU
mul_tiles_bcast_cols (incompatible with UnpackToDestFp32).

This file:
  - Directly exercises the two named cells against the PyTorch reference.
  - Probe data shows the post-R6 result lands at RMS=0.000291 — over an
    order of magnitude under the 0.02 target, plenty of margin.
  - Regression-guards the R5 fp32 + D=1024 path (UnpackToDestFp32 is now
    on for ANY fp32_dest_acc_en=True call; D=1024 hits the divide CB
    copies for Dt=32 — verifies no L1 overflow or wrong-answer).
  - Regression-guards the bf16 path (USE_UNTAGGED_DIVIDE=0; the direct
    divide branch must still produce correct answers).
  - Regression-guards composition with R2 GQA/MQA, R3 non-aligned,
    R4 mid-context (S=4096), and an fp32_dest_acc_en=False audit (which
    the validate() guard refuses — we expect an UnsupportedAxisValue).
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, scale: float | None):
    """Float32 reference matching the op contract."""
    qf, kf, vf = q.float(), k.float(), v.float()
    if qf.shape[1] != kf.shape[1]:
        reps = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(reps, dim=1)
        vf = vf.repeat_interleave(reps, dim=1)
    s = scale if scale is not None else 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    if mask is not None:
        scores = scores + mask.float()
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, vf).to(q.dtype)


def _pcc_rms(reference: torch.Tensor, actual: torch.Tensor):
    """Cosine-similarity-derived PCC + relative RMS (normalized by max(|ref|))."""
    rf = reference.float().flatten()
    af = actual.float().flatten()
    pcc = torch.nn.functional.cosine_similarity(rf - rf.mean(), af - af.mean(), dim=0).item()
    rms = (
        (reference.float() - actual.float()).pow(2).mean().sqrt() / reference.float().abs().clamp_min(1e-12).max()
    ).item()
    return pcc, rms


def _additive_causal_mask(B: int, S_q: int, S_kv: int, dtype: torch.dtype) -> torch.Tensor:
    """Phase-0-compatible causal mask: 0 on lower triangle, -inf on upper."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    triu = torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1)
    mask.masked_fill_(triu, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# Named failing cells: Q1x1x8192x64 fp32 self mha none × {auto, explicit}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_fp32_s8192_named_cell(device, scale_mode):
    """The two exact failing cells the R6 verifier named.

    Pre-R6 (post-R5 baseline) measurement on this exact shape:
        PCC=0.999631, RMS=0.0272 — target (0.999, 0.02); RMS over.

    Post-R6 expected (per probe_014): PCC ≈ 1.000006, RMS ≈ 0.000291 —
    massive margin under target, driven by:

      1) UnpackToDestFp32 on cb_cur_sum_exp (slot 12) and cb_cur_mm_out
         (slot 21) → per-K-iter SFPU reload preserves the full 24-bit
         FP32 mantissa instead of dropping to 10-bit TF32 through
         srcA/srcB.
      2) FPU final divide reads from untagged intermediate CBs
         (cb_cur_*_for_divide), filled via a single SFPU copy_tile per
         tile — adds at most one ULP per tile, vs. eliminating the
         sqrt(Kt)=16-way TF32-ULP cascade across Kt=256 K-iterations.
    """
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 8192, 64

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    if scale_mode == "explicit":
        kwargs["scale"] = 1.0 / math.sqrt(D)

    out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, kwargs.get("scale"))

    pcc, rms = _pcc_rms(reference, result)
    # fp32 target from the golden suite is PCC ≥ 0.999, RMS ≤ 0.02.
    assert pcc >= 0.999, f"PCC {pcc} below fp32 target 0.999 (scale_mode={scale_mode})"
    assert rms <= 0.02, f"RMS {rms} above fp32 target 0.02 (scale_mode={scale_mode})"


# ---------------------------------------------------------------------------
# fp32 long-context regression: S=4096 (R4 territory) must still pass
# under the new R6 divide path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S", [4096])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_fp32_s4096_no_regression(device, S, scale_mode):
    """R4 closed S=4096 fp32 cells via the two-pass algorithmic restructure.
    R6 changes the final-divide route — these must not regress."""
    torch.manual_seed(1)
    B, H, D = 1, 1, 64

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    if scale_mode == "explicit":
        kwargs["scale"] = 1.0 / math.sqrt(D)

    out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, kwargs.get("scale"))

    pcc, rms = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc} below target (S={S}, scale_mode={scale_mode})"
    assert rms <= 0.02, f"RMS {rms} above target (S={S}, scale_mode={scale_mode})"


# ---------------------------------------------------------------------------
# fp32 D=1024 (R5 territory) — must still fit L1 and produce correct
# results under the R6 CB additions. cb_cur_mm_out_for_divide adds 128 KB
# at Dt=32 single-buffered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_mode", ["none", "causal"])
def test_fp32_d1024_no_regression(device, mask_mode):
    """R5 unblocked fp32 D=1024 by reclaiming dead CBs (~287 KB). R6 reuses
    two of those reclaimed slots for the divide intermediates (~128 KB +
    4 KB at Dt=32) — should still fit with margin."""
    torch.manual_seed(2)
    B, H, S, D = 1, 1, 128, 1024

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    mask_pt = None
    if mask_mode == "causal":
        mask_pt = _additive_causal_mask(B, S, S, torch.float32)
        kwargs["attention_mask"] = ttnn.from_torch(mask_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, mask_pt, None)

    pcc, rms = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc} (mask_mode={mask_mode})"
    assert rms <= 0.02, f"RMS {rms} (mask_mode={mask_mode})"


# ---------------------------------------------------------------------------
# bf16 path: USE_UNTAGGED_DIVIDE=0 (direct mul branch). When the caller
# does not enable fp32_dest_acc_en, the running-state CBs are bf16,
# UnpackToDestFp32 is not applied, and the compute kernel takes the
# direct-divide branch (same as pre-R6).
#
# Note: the op file's validate() refuses fp32_dest_acc_en=False with a
# clear error (the fused-scale-exp SFPU path requires fp32 DEST). So this
# bf16 "direct path" coverage runs with the DEFAULT compute_kernel_config
# (fp32_dest_acc_en=True) — which IS the path R6 changes. The true
# USE_UNTAGGED_DIVIDE=0 branch is only reachable if a future refinement
# enables bf16 DEST. We cover that the existing config-rejection still
# holds AND that the bf16 input path (with default fp32 DEST) still works.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_mode", ["none", "causal"])
def test_bf16_input_default_config(device, mask_mode):
    """bf16 inputs with default config (fp32_dest_acc_en=True) — running-
    state CBs are still fp32, still UnpackToDestFp32-tagged, USE_UNTAGGED_
    DIVIDE=1 path. Must produce a correct bf16 answer."""
    torch.manual_seed(3)
    B, H, S, D = 1, 4, 128, 64

    q_pt = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k_pt = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v_pt = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    mask_pt = None
    if mask_mode == "causal":
        mask_pt = _additive_causal_mask(B, S, S, torch.bfloat16)
        kwargs["attention_mask"] = ttnn.from_torch(mask_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, mask_pt, None)

    pcc, _ = _pcc_rms(reference, result)
    assert pcc >= 0.995, f"PCC {pcc} (mask_mode={mask_mode})"


def test_validate_rejects_fp32_dest_acc_en_false(device):
    """R1's validate() refuses fp32_dest_acc_en=False (the fused-scale-exp
    SFPU path requires fp32 DEST). R6 must not have broken this guard —
    the direct-divide branch (USE_UNTAGGED_DIVIDE=0) is not yet reachable
    via a user-exposed config; this assertion documents that.

    When a future refinement enables bf16 DEST, this test will start
    XPASSing, and the implementer will need to flip the assertion."""
    torch.manual_seed(4)
    B, H, S, D = 1, 1, 32, 64
    q_pt = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    qt = ttnn.from_torch(q_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(q_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(q_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    )

    with pytest.raises(UnsupportedAxisValue, match="fp32_dest_acc_en=False"):
        scaled_dot_product_attention(qt, kt, vt, compute_kernel_config=cfg)


# ---------------------------------------------------------------------------
# Composition smoke tests — R6 must not have broken cross-refinement axes.
# ---------------------------------------------------------------------------


def test_fp32_s8192_compose_with_explicit_config(device):
    """S=8192 fp32 with an explicit compute_kernel_config — confirms the
    descriptor's unpack_to_dest_mode wiring fires when fp32_dest_acc_en
    is set explicitly (not just via the default)."""
    torch.manual_seed(5)
    B, H, S, D = 1, 1, 8192, 64

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )

    out = scaled_dot_product_attention(qt, kt, vt, compute_kernel_config=cfg)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, rms = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc}"
    assert rms <= 0.02, f"RMS {rms}"


def test_fp32_gqa_s4096_compose(device):
    """R2 (GQA) composition at the fp32 long-context floor — verifies the
    R6 divide path works for H_q != H_kv shapes."""
    torch.manual_seed(6)
    B, H_q, H_kv, S, D = 1, 4, 1, 4096, 64

    q_pt = torch.randn(B, H_q, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H_kv, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H_kv, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, rms = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc}"
    assert rms <= 0.02, f"RMS {rms}"


def test_fp32_non_aligned_compose(device):
    """R3 (W non-aligned at D=50) composition — verifies the R6 divide
    intermediate CBs allocate correctly when Dt is computed via
    ceil-divide on a non-aligned head dim."""
    torch.manual_seed(7)
    B, H, S, D = 1, 1, 32, 50  # Dt = 2 (ceil 50/32)

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, _ = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc}"


def test_fp32_s8192_with_causal_mask(device):
    """S=8192 fp32 + causal mask — verifies the precision lift composes
    with the existing R3 alignment-mask / R0 user-mask infrastructure."""
    torch.manual_seed(8)
    B, H, S, D = 1, 1, 8192, 64

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    mask_pt = _additive_causal_mask(B, S, S, torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mt = ttnn.from_torch(mask_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt, attention_mask=mt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, mask_pt, None)

    pcc, rms = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc}"
    assert rms <= 0.02, f"RMS {rms}"


# ---------------------------------------------------------------------------
# Tiny canary — exercises the descriptor's allocation + tag wiring on the
# smallest possible fp32 path (Dt=2). If unpack_to_dest_mode wiring is
# broken or the new divide CBs were mis-sized, this is the cheapest cell
# to catch it.
# ---------------------------------------------------------------------------


def test_fp32_tiny_smoke(device):
    """1-tile-everywhere fp32 path — verifies the R6 wiring fires without
    L1 issues on the smallest shape that exercises Dt > 1."""
    torch.manual_seed(9)
    B, H, S, D = 1, 1, 32, 64  # Dt=2, Kt=1, Qt=1

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, _ = _pcc_rms(reference, result)
    assert pcc >= 0.999, f"PCC {pcc}"
