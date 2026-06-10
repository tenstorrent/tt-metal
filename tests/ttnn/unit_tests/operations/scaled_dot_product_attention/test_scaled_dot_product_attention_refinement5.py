# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 5 unit tests — fp32 + large head_dim L1 capacity via CB reclaim.

The four named failing cells from op_requirements.md are:

    Q1x1x128x1024 fp32 self × {auto, explicit} × mask={none, causal}

R4-iter3's two-pass restructure stopped using cb_prev_max,
cb_prev_sum_exp, cb_exp_max_diff and cb_prev_mm_out, but left them
allocated. At fp32 D=1024 the unused `cb_prev_mm_out` alone was 256 KB
(2 × Dt × fp32 tile_size with Dt=32), pushing the program's per-core L1
footprint to 1,598,816 B vs the 1,499,136 B cap.

R5 deletes the four unused CB descriptors (and the dead helper functions
that referenced them in the compute kernel) — saving ~287 KB at
fp32 D=1024 and unblocking the four golden cells with ~187 KB headroom.

This file:
  - directly exercises the four named cells against the PyTorch reference.
  - guards the per-cell precision floor at fp32 + D=1024 (well-conditioned
    random inputs PCC ~1.0; the looser ≥0.999 / RMS≤0.02 target leaves
    plenty of margin).
  - regression-guards bf16 + D=1024 (unchanged behavior — bf16 fit in L1
    even before R5).
  - regression-guards the small fp32 + small-D path (the R5 CB cleanup
    must not change small-D semantics).
  - regression-guards composition with Refinement 1's compute_kernel_config,
    Refinement 2's GQA/MQA, and Refinement 3's non-aligned shapes (the R5
    cleanup only deletes CBs, but the compute kernel removed the dead
    helpers that those refinements indirectly survived against, so smoke-
    test the chain).
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

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
    rf = reference.float().flatten()
    af = actual.float().flatten()
    pcc = torch.nn.functional.cosine_similarity(rf - rf.mean(), af - af.mean(), dim=0).item()
    rms = ((reference.float() - actual.float()).pow(2).mean().sqrt() / reference.float().abs().clamp_min(1e-12).max()).item()
    return pcc, rms


def _additive_causal_mask(B: int, S_q: int, S_kv: int, dtype: torch.dtype) -> torch.Tensor:
    """Phase-0-compatible causal mask: 0 on lower triangle, -inf on upper."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    triu = torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1)
    mask.masked_fill_(triu, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# Named failing cells: Q1x1x128x1024 fp32 × {auto, explicit} × {none, causal}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
def test_fp32_d1024_named_cell(device, scale_mode, mask_mode):
    """Exercise the exact four golden cells the R5 verifier named.

    The pre-R5 program descriptor OOMed here at static CB allocation time:
        1,598,816 B vs 1,499,136 B cap — over by 99,680 B.
    The R5 CB-reclaim drops 287 KB → ~1,312 KB; plenty of headroom.
    """
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 1024

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    if scale_mode == "explicit":
        kwargs["scale"] = 1.0 / math.sqrt(D)

    mask_pt = None
    if mask_mode == "causal":
        mask_pt = _additive_causal_mask(B, S, S, torch.float32)
        kwargs["attention_mask"] = ttnn.from_torch(mask_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, mask_pt, kwargs.get("scale"))

    pcc, rms = _pcc_rms(reference, result)
    # fp32 target from the golden suite is PCC ≥ 0.999, RMS ≤ 0.02.
    assert pcc >= 0.999, f"PCC {pcc} below fp32 target 0.999 (scale_mode={scale_mode}, mask_mode={mask_mode})"
    assert rms <= 0.02, f"RMS {rms} above fp32 target 0.02 (scale_mode={scale_mode}, mask_mode={mask_mode})"


# ---------------------------------------------------------------------------
# Regression guards: bf16 + D=1024 (would already fit pre-R5; must not regress)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_mode", ["none", "causal"])
def test_bf16_d1024_no_regression(device, mask_mode):
    """bf16 + D=1024 fit in L1 even before R5 because bf16 page sizes are
    half of fp32 — half the per-Dt-tile bytes. R5 must not have broken
    these cells while shrinking the CB set."""
    torch.manual_seed(1)
    B, H, S, D = 1, 1, 128, 1024

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
        kwargs["attention_mask"] = ttnn.from_torch(
            mask_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    out = scaled_dot_product_attention(qt, kt, vt, **kwargs)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, mask_pt, None)

    pcc, _ = _pcc_rms(reference, result)
    assert pcc >= 0.995, f"PCC {pcc} below bf16 target 0.995 (mask_mode={mask_mode})"


# ---------------------------------------------------------------------------
# Regression guards: smaller-D shapes must still work after CB reclaim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,S,D,dtype",
    [
        (1, 1, 32, 64, ttnn.bfloat16),  # smallest single-tile shape
        (1, 1, 64, 64, ttnn.float32),  # smallest fp32 multi-tile-Sq
        (1, 4, 128, 128, ttnn.bfloat16),  # mid-size bf16
        (2, 2, 128, 64, ttnn.float32),  # multi-batch fp32
    ],
)
def test_small_d_no_regression(device, B, H, S, D, dtype):
    """The CB reclaim removed descriptors and dead helpers; the live
    K-loop path is unchanged. Smoke-test that small-D shapes still
    produce the right answer."""
    torch.manual_seed(2)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    q_pt = torch.randn(B, H, S, D, dtype=torch_dtype)
    k_pt = torch.randn(B, H, S, D, dtype=torch_dtype)
    v_pt = torch.randn(B, H, S, D, dtype=torch_dtype)

    qt = ttnn.from_torch(q_pt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, _ = _pcc_rms(reference, result)
    target = 0.999 if dtype == ttnn.float32 else 0.995
    assert pcc >= target, f"PCC {pcc} below target {target} (dtype={dtype}, shape=({B},{H},{S},{D}))"


# ---------------------------------------------------------------------------
# R1 / R2 / R3 composition smoke tests at D=1024 (verify reclaim didn't
# break the compose-with-other-refinements axes)
# ---------------------------------------------------------------------------


def test_fp32_d1024_compose_with_compute_kernel_config(device):
    """Refinement-1 composition: caller-supplied compute_kernel_config
    threads through to the program descriptor; the CB widening + the R5
    reclaim must still produce correct results."""
    torch.manual_seed(3)
    B, H, S, D = 1, 1, 128, 1024

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
    assert pcc >= 0.999
    assert rms <= 0.02


def test_fp32_d128_gqa_no_regression(device):
    """Refinement-2 (GQA) at moderate-D fp32 — covers the
    'shrunk CB set must still address h_kv != h_q correctly' path."""
    torch.manual_seed(4)
    B, H_q, H_kv, S, D = 1, 4, 2, 64, 128

    q_pt = torch.randn(B, H_q, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H_kv, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H_kv, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, _ = _pcc_rms(reference, result)
    assert pcc >= 0.999


def test_fp32_non_aligned_no_regression(device):
    """Refinement-3 (non-aligned dims) — D=50 (W non-aligned) at fp32.
    Exercises the alignment-mask synthesizer's compatibility with the
    R5 CB set."""
    torch.manual_seed(5)
    B, H, S, D = 1, 1, 32, 50

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
    assert pcc >= 0.999


# ---------------------------------------------------------------------------
# Optional D=512 fp32 cell — sits BELOW the OOM line both before and after,
# but exercises the new CB layout at a non-trivial Dt without hitting the
# cap. Useful as a self-check that the descriptor edits didn't break smaller
# fp32-D shapes too.
# ---------------------------------------------------------------------------


def test_fp32_d512_intermediate_size(device):
    """fp32 + D=512 — well within budget but tests the K-loop on a
    larger-than-baseline Dt. Pre-R5 already worked; R5 must not regress."""
    torch.manual_seed(6)
    B, H, S, D = 1, 1, 64, 512

    q_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    k_pt = torch.randn(B, H, S, D, dtype=torch.float32)
    v_pt = torch.randn(B, H, S, D, dtype=torch.float32)

    qt = ttnn.from_torch(q_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_pt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(qt, kt, vt)
    result = ttnn.to_torch(out)
    reference = _torch_sdpa(q_pt, k_pt, v_pt, None, None)

    pcc, rms = _pcc_rms(reference, result)
    assert pcc >= 0.999
    assert rms <= 0.02
