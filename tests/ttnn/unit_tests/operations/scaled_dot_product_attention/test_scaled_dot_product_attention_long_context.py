# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — long-context precision via Bkv_t KV-chunk blocking.

The FlashAttention online-softmax recurrence rescales the running output
(corr*O) once per KV-chunk. At S>=4096 mask=none the output is a near-zero
average of thousands of V vectors, so the per-chunk 16-bit-DEST rounding of
that rescale accumulates into a large *relative* RMS. Folding Bkv_t key-tiles
per KV-chunk (host-chosen, gated to long / tile-aligned shapes) cuts the
rescale-round count by Bkv_t — the only lever once the golden pins
fp32_dest_acc_en / math_fidelity.

This file exercises the lever directly on the long-context cells that were
`supported_fail` after R1:

  - mask=none @ S in {4096, 8192} for bf16/bf8b acc=False (HiFi2) and
    fp32 acc=True (HiFi4) — the dominant near-zero-output failures.
  - mask=custom @ S=8192 acc=False — the triangular-mask long-context misses.
  - causal @ S in {4096, 8192} acc=False — exercises the Bkv_t-wide diagonal
    block (zeros for past tiles, triangular diag, -inf future), including
    qc positions that are not Bkv_t-aligned (qc ranges over every tile row).
  - a short (Bkv_t==1) control per mode: the kernel must be byte-for-byte the
    Phase-0 per-tile path there (no regression).

Tolerances come from the golden helpers so this test gates on exactly what the
golden suite gates on.
"""

import pytest
import torch

import ttnn
from eval.golden_tests.scaled_dot_product_attention.helpers import (
    TOLERANCES,
    make_causal_mask,
    pytorch_scaled_dot_product_attention,
)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def _rel_rms(out, exp):
    o, e = out.float(), exp.float()
    return (o - e).pow(2).mean().sqrt().item() / e.std().item()


def _pcc(out, exp):
    a, b = out.float().flatten(), exp.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _acc_config(fp32_dest_acc_en):
    # acc=True -> the op default (None -> HiFi4 + fp32 DEST acc); acc=False ->
    # the golden's pinned HiFi2 + 16-bit DEST config (the path that surfaced the
    # long-context floor).
    if fp32_dest_acc_en:
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _run(device, q_shape, k_shape, *, dtype, fp32_acc, mask_mode, scale=None):
    tdt = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=tdt)
    K = torch.randn(k_shape, dtype=tdt)
    V = torch.randn(k_shape, dtype=tdt)

    is_causal = mask_mode == "causal"
    torch_mask = None
    if mask_mode == "custom":
        torch_mask = make_causal_mask(q_shape[0], q_shape[2], k_shape[2], torch_dtype=tdt)

    expected = pytorch_scaled_dot_product_attention(Q, K, V, attn_mask=torch_mask, is_causal=is_causal, scale=scale)

    tq = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tm = (
        ttnn.from_torch(torch_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if torch_mask is not None
        else None
    )

    out = scaled_dot_product_attention(
        tq,
        tk,
        tv,
        attn_mask=tm,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=_acc_config(fp32_acc),
    )
    out_t = ttnn.to_torch(out)

    pcc_floor, rms_ceil = TOLERANCES[(dtype, fp32_acc)]
    pcc, rms = _pcc(out_t, expected), _rel_rms(out_t, expected)
    assert pcc >= pcc_floor and rms <= rms_ceil, (
        f"dtype={dtype} acc={fp32_acc} mask={mask_mode} S={q_shape[2]}: "
        f"pcc={pcc:.5f} (>= {pcc_floor}) rms={rms:.4f} (<= {rms_ceil})"
    )


# (dtype, fp32_dest_acc_en) — the three numeric paths that missed at long context.
_NUMERIC = [
    pytest.param(ttnn.bfloat16, False, id="bf16_acc=False"),
    pytest.param(ttnn.bfloat8_b, False, id="bf8b_acc=False"),
    pytest.param(ttnn.float32, True, id="fp32_acc=True"),
]


@pytest.mark.parametrize("dtype,fp32_acc", _NUMERIC)
@pytest.mark.parametrize("S", [4096, 8192])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_long_context_mask_none(device, dtype, fp32_acc, S, scale_mode):
    """mask=none long-context — the near-zero-output failures Bkv blocking fixes."""
    scale = 0.125 if scale_mode == "explicit" else None
    _run(device, (1, 1, S, 64), (1, 1, S, 64), dtype=dtype, fp32_acc=fp32_acc, mask_mode="none", scale=scale)


@pytest.mark.parametrize("dtype,fp32_acc", _NUMERIC)
@pytest.mark.parametrize("S", [4096, 8192])
def test_long_context_causal(device, dtype, fp32_acc, S):
    """causal long-context — exercises the Bkv_t-wide diagonal block."""
    _run(device, (1, 2, S, 64), (1, 2, S, 64), dtype=dtype, fp32_acc=fp32_acc, mask_mode="causal")


@pytest.mark.parametrize("dtype,fp32_acc", _NUMERIC)
def test_long_context_custom_mask_8192(device, dtype, fp32_acc):
    """mask=custom @ S=8192 — the triangular-mask long-context miss (acc=False)."""
    _run(device, (1, 1, 8192, 64), (1, 1, 8192, 64), dtype=dtype, fp32_acc=fp32_acc, mask_mode="custom")


@pytest.mark.parametrize("dtype,fp32_acc", _NUMERIC)
def test_long_context_gqa_d128(device, dtype, fp32_acc):
    """GQA + D=128 long context (Bkv_t blocks a d_t=4 head dim)."""
    _run(device, (1, 8, 4096, 128), (1, 2, 4096, 128), dtype=dtype, fp32_acc=fp32_acc, mask_mode="none")


@pytest.mark.parametrize("mask_mode", ["none", "custom", "causal"])
def test_short_bkv1_no_regression(device, mask_mode):
    """Short shape (Skv_t < 128) keeps Bkv_t==1 — the Phase-0 per-tile path."""
    _run(device, (1, 2, 256, 64), (1, 2, 256, 64), dtype=ttnn.bfloat16, fp32_acc=False, mask_mode=mask_mode)
