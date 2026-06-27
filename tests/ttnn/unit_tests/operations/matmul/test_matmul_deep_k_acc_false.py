# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1b — bf16-output + fp32_dest_acc_en=False at extreme K.

The 2 residual golden fails after Refinement 1 were the deep-K (K=8192)
bf16-output acc=False cells: the default software K-spill re-quantizes the
running K-sum to bf16 on every K-block reload, so the deep-K error hit the
16-bit floor (relRMS ~0.128 vs the (bfloat16, False) golden band 0.10).

The fix (matmul_program_descriptor.py) generalizes Refinement 1's Lever B
from bf8b->bf16-interm to bf16->fp32-interm: a bf16 output with acc=False now
opts into HARDWARE packer-L1 accumulation with an fp32 cb_interm. The
cross-K-block running sum accumulates in fp32 in L1 (never reloaded into the
16-bit DEST until the final block), bounding the 16-bit in-DEST accumulation
run to ONE K-block. fp32_dest_acc_en=False (16-bit DEST per block) is still
honored — packer_l1_acc is an orthogonal hardware knob.

These tests assert the deep-K bf16 acc=False cells clear the golden band, and
guard against regression on the shallow/medium-K bf16 cells (which switch to
the same path) and the fp32 / acc=True paths (which are unchanged).

DO NOT DELETE — documents the Refinement 1b numerical fix.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.matmul import matmul


# ---------------------------------------------------------------------------
# metrics (mirror eval/metrics.py: relative RMS = abs_rms / reference stddev)
# ---------------------------------------------------------------------------
def _rel_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_rms = torch.nn.functional.mse_loss(actual_f, expected_f).sqrt().item()
    scale = expected_f.std().item()
    return abs_rms / scale if scale > 1e-12 else abs_rms


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float().flatten()
    e = expected.float().flatten()
    a = a - a.mean()
    e = e - e.mean()
    denom = (a.norm() * e.norm()).item()
    return (a @ e).item() / denom if denom > 1e-12 else 0.0


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}


def _run(device, a_shape, b_shape, *, dtype, weight_dtype, fp32_dest_acc_en):
    torch.manual_seed(0)
    A = torch.randn(a_shape, dtype=_TORCH_DTYPE[dtype])
    B = torch.randn(b_shape, dtype=_TORCH_DTYPE[weight_dtype])
    expected = torch.matmul(A.float(), B.float()).to(_TORCH_DTYPE[dtype])

    tA = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tB = ttnn.from_torch(B, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
    )
    out = matmul(tA, tB, compute_kernel_config=cfg)
    res = ttnn.to_torch(out).to(_TORCH_DTYPE[dtype])
    return res, expected


# golden (bfloat16, False) band
_BF16_ACCFALSE_PCC = 0.99
_BF16_ACCFALSE_RMS = 0.10


# ---------------------------------------------------------------------------
# The 2 target cells: K=8192 bf16 output + acc=False (the Refinement 1b goal)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "weight_dtype",
    [ttnn.bfloat16, ttnn.float32],
    ids=["wbf16", "wfp32"],
)
def test_deep_k8192_bf16_out_acc_false(device, weight_dtype):
    """A256x8192 — the formerly-failing residual cells. Must clear the band."""
    res, expected = _run(
        device,
        (256, 8192),
        (8192, 2048),
        dtype=ttnn.bfloat16,
        weight_dtype=weight_dtype,
        fp32_dest_acc_en=False,
    )
    rms = _rel_rms(res, expected)
    pcc = _pcc(res, expected)
    print(f"K=8192 bf16/{weight_dtype} acc=False : relRMS={rms:.4f} PCC={pcc:.5f}")
    assert pcc >= _BF16_ACCFALSE_PCC, f"PCC {pcc:.5f} < {_BF16_ACCFALSE_PCC}"
    assert rms <= _BF16_ACCFALSE_RMS, f"relRMS {rms:.4f} > band {_BF16_ACCFALSE_RMS}"


# ---------------------------------------------------------------------------
# K-depth ladder: error must stay well bounded (not grow ~O(sqrt(K)) as it did
# before the fix). At K=8192 the old path hit relRMS 0.128; the fp32-L1-acc
# path keeps every depth comfortably under the band.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [512, 2048, 4096, 8192], ids=lambda k: f"K{k}")
def test_k_depth_ladder_bf16_acc_false(device, K):
    res, expected = _run(
        device,
        (256, K),
        (K, 1024),
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        fp32_dest_acc_en=False,
    )
    rms = _rel_rms(res, expected)
    pcc = _pcc(res, expected)
    print(f"K={K} bf16 acc=False : relRMS={rms:.4f} PCC={pcc:.5f}")
    assert pcc >= _BF16_ACCFALSE_PCC, f"PCC {pcc:.5f} < {_BF16_ACCFALSE_PCC}"
    assert rms <= _BF16_ACCFALSE_RMS, f"relRMS {rms:.4f} > band {_BF16_ACCFALSE_RMS}"


# ---------------------------------------------------------------------------
# Non-regression: fp32 (acc=True) and bf16 acc=True paths are untouched by the
# packer-L1 change (the use_packer_l1_acc gate is acc=False-only).
# ---------------------------------------------------------------------------
def test_no_regression_fp32_acc_true(device):
    res, expected = _run(
        device,
        (256, 4096),
        (4096, 1024),
        dtype=ttnn.float32,
        weight_dtype=ttnn.float32,
        fp32_dest_acc_en=True,
    )
    pcc = _pcc(res, expected)
    rms = _rel_rms(res, expected)
    print(f"fp32 acc=True : relRMS={rms:.4f} PCC={pcc:.5f}")
    assert pcc >= 0.999 and rms <= 0.02


def test_no_regression_bf16_acc_true(device):
    res, expected = _run(
        device,
        (256, 8192),
        (8192, 1024),
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        fp32_dest_acc_en=True,
    )
    pcc = _pcc(res, expected)
    rms = _rel_rms(res, expected)
    print(f"bf16 acc=True : relRMS={rms:.4f} PCC={pcc:.5f}")
    assert pcc >= 0.997 and rms <= 0.04
