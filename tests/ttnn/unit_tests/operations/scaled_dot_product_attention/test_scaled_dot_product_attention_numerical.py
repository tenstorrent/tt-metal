# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — Numerical configurability tests for scaled_dot_product_attention.

Exercises the new R1 surface directly:
  * dtype ∈ {bfloat16, float32, bfloat8_b} across mask/scale/attention-kind
    combinations (the golden-relevant risk surface — esp. bf8b + causal mask,
    where the additive −inf must survive block-float storage).
  * compute_kernel_config plumbing (math_fidelity / fp32_dest_acc_en) — that a
    user-supplied config is honored and that None reproduces Phase-0 behavior.
  * test_scaled_dot_product_attention_precision_matrix — the authoritative
    precision characterization across dtype × fidelity × fp32_acc × shape.

All shapes are tile-aligned MHA (alignment/kv_heads refinements are R2/R3).
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Reference + helpers
# ---------------------------------------------------------------------------


def _pytorch_sdpa(Q, K, V, *, attention_mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf)


def _make_causal_mask(B, S_q, S_kv, *, torch_dtype):
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


# bf8b has no native torch dtype — reference in bf16, store on device as bf8b.
_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}

# Per-dtype PCC floors (mirrors the golden suite TOLERANCES, slightly tightened
# for the small/medium tile-aligned shapes used here).
_PCC = {
    ttnn.bfloat16: 0.995,
    ttnn.float32: 0.999,
    ttnn.bfloat8_b: 0.99,
}


def _to_device(t, device, dtype):
    if t is None:
        return None
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _metrics(expected, actual):
    abs_err = (actual.float() - expected.float()).abs()
    _, pcc = comp_pcc(expected.float(), actual.float())
    rms = abs_err.pow(2).mean().sqrt().item()
    ref_rms = expected.float().pow(2).mean().sqrt().clamp(min=1e-10).item()
    return pcc, abs_err.max().item(), rms / ref_rms


# ---------------------------------------------------------------------------
# 1. dtype × mask × scale × attention-kind  (the golden risk surface)
# ---------------------------------------------------------------------------

_DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat8_b, id="bf8b"),
]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
@pytest.mark.parametrize(
    "q_shape,kv_shape",
    [
        pytest.param((1, 2, 128, 64), (1, 2, 128, 64), id="self_128"),
        pytest.param((1, 4, 256, 64), (1, 4, 256, 64), id="self_256"),
        pytest.param((1, 4, 64, 64), (1, 4, 128, 64), id="cross_sq_lt_skv"),
        pytest.param((1, 4, 128, 64), (1, 4, 64, 64), id="cross_sq_gt_skv"),
    ],
)
def test_dtype_mask_scale(dtype, mask_mode, scale_mode, q_shape, kv_shape, device):
    """Every supported dtype across mask/scale/attention-kind.

    bf8b + causal is the sharp edge (additive −inf in block-float). If it fails
    here it becomes an EXCLUSIONS entry in the op file.
    """
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    B, Hq, S_q, D = q_shape
    _, Hkv, S_kv, _ = kv_shape

    Q = torch.randn(B, Hq, S_q, D, dtype=torch_dtype)
    K = torch.randn(B, Hkv, S_kv, D, dtype=torch_dtype)
    V = torch.randn(B, Hkv, S_kv, D, dtype=torch_dtype)

    mask = _make_causal_mask(B, S_q, S_kv, torch_dtype=torch_dtype) if mask_mode == "causal" else None
    scale = 0.125 if scale_mode == "explicit" else None

    expected = _pytorch_sdpa(Q, K, V, attention_mask=mask, scale=scale)

    out = scaled_dot_product_attention(
        _to_device(Q, device, dtype),
        _to_device(K, device, dtype),
        _to_device(V, device, dtype),
        attention_mask=_to_device(mask, device, dtype),
        scale=scale,
    )
    assert out.dtype == dtype, f"output dtype {out.dtype} != input {dtype}"
    actual = ttnn.to_torch(out)
    assert list(actual.shape) == list(expected.shape)

    pcc, max_abs, rel_rms = _metrics(expected, actual)
    print(
        f"\n[dtype_mask] dt={dtype} mask={mask_mode} scale={scale_mode} "
        f"q={q_shape} kv={kv_shape} PCC={pcc:.6f} max_abs={max_abs:.5f} rel_rms={rel_rms:.5f}"
    )
    assert pcc >= _PCC[dtype], f"PCC {pcc:.6f} < floor {_PCC[dtype]} (dt={dtype}, mask={mask_mode})"


# ---------------------------------------------------------------------------
# 2. compute_kernel_config plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
def test_compute_kernel_config_honored(dtype, math_fidelity, fp32_acc, device):
    """A user-supplied compute_kernel_config runs and produces a sane result.

    Default config (None) is covered by every other test; here we confirm the
    knobs are wired through without hanging and stay numerically reasonable.
    """
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    shape = (1, 2, 128, 64)
    Q = torch.randn(shape, dtype=torch_dtype)
    K = torch.randn(shape, dtype=torch_dtype)
    V = torch.randn(shape, dtype=torch_dtype)
    expected = _pytorch_sdpa(Q, K, V)

    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        None,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )
    out = scaled_dot_product_attention(
        _to_device(Q, device, dtype),
        _to_device(K, device, dtype),
        _to_device(V, device, dtype),
        compute_kernel_config=config,
    )
    actual = ttnn.to_torch(out)
    pcc, max_abs, rel_rms = _metrics(expected, actual)
    print(
        f"\n[ckc] dt={dtype} fid={math_fidelity} fp32_acc={fp32_acc} "
        f"PCC={pcc:.6f} max_abs={max_abs:.5f} rel_rms={rel_rms:.5f}"
    )
    # Loose floor — LoFi / bf16-acc legitimately lose precision; this only guards
    # against a config combo silently corrupting the result.
    assert pcc >= 0.97, f"PCC {pcc:.6f} unexpectedly low for fid={math_fidelity} fp32_acc={fp32_acc}"


# ---------------------------------------------------------------------------
# 3. test_scaled_dot_product_attention_precision_matrix
# ---------------------------------------------------------------------------

_PM_SHAPES = [
    pytest.param((1, 1, 32, 32), id="1x1x32x32"),
    pytest.param((1, 1, 128, 64), id="1x1x128x64"),
    pytest.param((1, 4, 256, 64), id="1x4x256x64"),
    pytest.param((1, 8, 512, 64), id="1x8x512x64"),
    pytest.param((1, 2, 128, 128), id="1x2x128x128"),
]


@pytest.mark.parametrize("distribution", [pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _PM_SHAPES)
def test_scaled_dot_product_attention_precision_matrix(shape, dtype, math_fidelity, fp32_acc, distribution, device):
    """Authoritative precision characterization for SDPA across the numeric surface.

    randn (the registry distribution) only — adversarial uniform/negative
    distributions are characterized in test_regression.py and are a known,
    out-of-R1-scope bf16-accumulator limitation. Asserts a permissive PCC floor;
    prints all metrics for the results file.
    """
    torch_dtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    B, H, S, D = shape
    gen = torch.randn if distribution == "randn" else torch.rand
    Q = gen(B, H, S, D, dtype=torch_dtype)
    K = gen(B, H, S, D, dtype=torch_dtype)
    V = gen(B, H, S, D, dtype=torch_dtype)
    expected = _pytorch_sdpa(Q, K, V)

    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        None,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )
    out = scaled_dot_product_attention(
        _to_device(Q, device, dtype),
        _to_device(K, device, dtype),
        _to_device(V, device, dtype),
        compute_kernel_config=config,
    )
    actual = ttnn.to_torch(out)
    pcc, max_abs, rel_rms = _metrics(expected, actual)
    print(
        f"\n[precision_matrix] shape={shape} dt={dtype} fid={math_fidelity} "
        f"fp32_acc={fp32_acc} dist={distribution} PCC={pcc:.6f} "
        f"max_abs={max_abs:.5f} rel_rms={rel_rms:.5f}"
    )
    # Permissive floor: LoFi + bf16-acc on the largest shapes legitimately lose
    # precision. The golden suite is the tight gate; this catches catastrophes.
    assert pcc >= 0.95, f"PCC {pcc:.6f} catastrophically low (shape={shape}, dt={dtype}, fid={math_fidelity})"
