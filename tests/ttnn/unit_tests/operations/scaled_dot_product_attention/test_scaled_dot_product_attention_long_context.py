# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — long-context / near-uniform-attention precision tests.

Root cause fixed: HiFi2 skips the SrcB low-mantissa fidelity phase, so the
rowsum l (reduce SUM) and P@V (matmul) consumed the packed probs at different
effective precision; with bf16 cb_probs the V=ones invariant (output must be
exactly 1) was off by up to 11%. Fix: bf16/bf8b default math_fidelity=HiFi3 +
cb_probs follows fp32_dest_acc_en (Float32 by default).

Tests here lock both halves of the fix:
- long-context randn cells at golden tolerances (the 12 cells filed under R3)
- V=ones invariant (kernel-internal consistency, far tighter than golden)
- uniform/negative inputs at the bf16 output quantization floor (max-abs gate,
  see changelog — rel-RMS at these distributions is floor-limited).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _reference(Q, K, V):
    D = Q.shape[-1]
    Kf, Vf = K.float(), V.float()
    if Q.shape[1] != K.shape[1]:
        r = Q.shape[1] // K.shape[1]
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    return torch.softmax((Q.float() @ Kf.transpose(-2, -1)) / math.sqrt(D), -1) @ Vf


def _run(device, Q, K, V, dtype):
    t = lambda x: ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(scaled_dot_product_attention(t(Q), t(K), t(V))).float()


# (q_shape, kv_shape, dtype, rel_rms_target) — the R3 golden cells + inherited
LONG_CONTEXT_CASES = [
    ((1, 1, 4096, 64), (1, 1, 4096, 64), ttnn.bfloat16, 0.05),
    ((1, 1, 8192, 64), (1, 1, 8192, 64), ttnn.bfloat16, 0.05),
    ((1, 4, 4096, 64), (1, 4, 4096, 64), ttnn.bfloat16, 0.05),
    ((1, 4, 4096, 64), (1, 1, 4096, 64), ttnn.bfloat16, 0.05),  # MQA (R2 inherited)
    ((1, 8, 4096, 128), (1, 2, 4096, 128), ttnn.bfloat16, 0.05),  # GQA (R2 inherited)
    ((1, 1, 8192, 64), (1, 1, 8192, 64), ttnn.bfloat8_b, 0.12),  # R1 inherited
]
_IDS = [f"Q{'x'.join(map(str, q))}_KV{'x'.join(map(str, k))}_{d.name}" for q, k, d, _ in LONG_CONTEXT_CASES]


@pytest.mark.parametrize("q_shape,kv_shape,dtype,target", LONG_CONTEXT_CASES, ids=_IDS)
def test_long_context_randn(device, q_shape, kv_shape, dtype, target):
    """The R3 supported_fail cells: randn, mask=none, auto scale."""
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(kv_shape, dtype=torch.bfloat16)
    V = torch.randn(kv_shape, dtype=torch.bfloat16)
    ref = _reference(Q, K, V)
    out = _run(device, Q, K, V, dtype)
    rel_rms = ((out - ref).pow(2).mean().sqrt() / ref.std()).item()
    assert rel_rms <= target, f"rel_rms {rel_rms:.4f} > {target}"


@pytest.mark.parametrize("S", [128, 1024, 4096, 8192])
def test_v_ones_invariant(device, S):
    """V == 1 makes the exact output 1.0 for every element, independent of
    Q/K/m/l. Any deviation is rowsum-l vs P@V inconsistency in the kernel.
    Pre-R3 this was off by up to 0.11 abs (HiFi2 SrcB truncation + bf16
    probs); post-fix it is within ~3 bf16 ulps."""
    sh = (1, 1, S, 64)
    torch.manual_seed(0)
    Q = torch.randn(sh, dtype=torch.bfloat16)
    K = torch.randn(sh, dtype=torch.bfloat16)
    V = torch.ones(sh, dtype=torch.bfloat16)
    out = _run(device, Q, K, V, ttnn.bfloat16)
    max_err = (out - 1.0).abs().max().item()
    # 4 bf16 ulp @1.0 (6 at S=8192: more KV blocks → more recurrence rounding);
    # pre-R3 the deviation reached 28 ulp at S=4096.
    bound = 0.0157 if S <= 4096 else 0.0235
    assert max_err <= bound, f"V=ones invariant broken: max |O-1| = {max_err:.5f}"


@pytest.mark.parametrize("sign", ["uniform", "negative"])
@pytest.mark.parametrize("S", [128, 512])
def test_near_uniform_attention_ulp_floor(device, sign, S):
    """Uniform / negative-uniform inputs drive near-uniform attention: the
    output std is ~1 bf16 ulp, so relative-RMS is floor-limited (see
    changelog). Gate on max-abs <= 2 bf16 ulp of the output magnitude (~0.5)
    instead — pre-R3 this was up to 16 ulp at S=512 (severity=bug)."""
    sh = (1, 4, S, 64)
    torch.manual_seed(42)
    if sign == "uniform":
        Q, K, V = (torch.rand(sh, dtype=torch.bfloat16) for _ in range(3))
    else:
        Q, K, V = (-(torch.rand(sh, dtype=torch.bfloat16) + 0.5) for _ in range(3))
    ref = _reference(Q, K, V)
    out = _run(device, Q, K, V, ttnn.bfloat16)
    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 0.0079, f"max_abs {max_abs:.5f} > 2 bf16 ulp"
