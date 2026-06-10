# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — GQA / MQA head mapping tests.

Reader maps Q head h -> KV head h / (H_q / H_kv). These tests exercise that
mapping directly: random-data PCC against a repeat_interleave reference, plus
a deterministic head-identification test where each KV head's V is a constant
so any head-mapping error is an exact, visible value mismatch.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def torch_sdpa(q, k, v, mask=None, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if q.shape[1] != k.shape[1]:  # GQA/MQA head broadcast
        r = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(r, 1)
        v = v.repeat_interleave(r, 1)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        s = s + mask
    return torch.matmul(torch.softmax(s, dim=-1), v)


def causal_mask(B, H, S_q, S_kv):
    m = torch.zeros(B, 1, S_q, S_kv)
    m.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


def compute_pcc(golden, actual):
    g = golden.flatten().float()
    a = actual.flatten().float()
    if torch.allclose(g, a, atol=1e-8):
        return 1.0
    return torch.corrcoef(torch.stack([g, a]))[0, 1].item()


def run_case(device, q_shape, kv_shape, mask_mode="none", scale=None, dtype=ttnn.bfloat16):
    torch.manual_seed(42)
    q = torch.randn(q_shape)
    k = torch.randn(kv_shape)
    v = torch.randn(kv_shape)
    mask = causal_mask(q_shape[0], q_shape[1], q_shape[2], kv_shape[2]) if mask_mode == "causal" else None

    golden = torch_sdpa(q, k, v, mask=mask, scale=scale)

    tt_q = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {}
    if mask is not None:
        kwargs["attention_mask"] = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if scale is not None:
        kwargs["scale"] = scale

    out = ttnn.to_torch(scaled_dot_product_attention(tt_q, tt_k, tt_v, **kwargs))
    assert out.shape == golden.shape
    pcc = compute_pcc(golden, out)
    assert pcc >= PCC[dtype], f"PCC {pcc:.5f} < {PCC[dtype]} for q={q_shape} kv={kv_shape} mask={mask_mode}"


# --- GQA self / cross / batch / causal -------------------------------------


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 8, 128, 64), (1, 2, 128, 64)),  # 4:1 ratio
        ((1, 32, 128, 128), (1, 8, 128, 128)),  # llama3-ish 4:1, D=128
        ((2, 8, 128, 64), (2, 2, 128, 64)),  # batched
        ((1, 8, 64, 64), (1, 2, 128, 64)),  # cross-attention, S_q < S_kv
    ],
    ids=["gqa_4to1", "gqa_llama3", "gqa_batch", "gqa_cross"],
)
def test_gqa(device, q_shape, kv_shape):
    run_case(device, q_shape, kv_shape)


@pytest.mark.parametrize("mask_mode", ["causal"])
def test_gqa_causal(device, mask_mode):
    run_case(device, (1, 8, 128, 64), (1, 2, 128, 64), mask_mode=mask_mode)


def test_gqa_explicit_scale(device):
    run_case(device, (1, 8, 128, 64), (1, 4, 128, 64), scale=0.1)


# --- MQA self / cross / batch / causal -------------------------------------


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 8, 128, 64), (1, 1, 128, 64)),  # base
        ((1, 32, 128, 128), (1, 1, 128, 128)),  # large H ratio, D=128
        ((2, 8, 128, 64), (2, 1, 128, 64)),  # batched
        ((1, 8, 64, 64), (1, 1, 128, 64)),  # cross-attention
    ],
    ids=["mqa_base", "mqa_large", "mqa_batch", "mqa_cross"],
)
def test_mqa(device, q_shape, kv_shape):
    run_case(device, q_shape, kv_shape)


def test_mqa_causal(device):
    run_case(device, (1, 8, 128, 64), (1, 1, 128, 64), mask_mode="causal")


# --- dtype coverage on the new axis values ----------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_gqa_dtypes(device, dtype):
    run_case(device, (1, 8, 128, 64), (1, 2, 128, 64), dtype=dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
def test_mqa_dtypes(device, dtype):
    run_case(device, (1, 8, 128, 64), (1, 1, 128, 64), dtype=dtype)


# --- deterministic head-mapping identification -------------------------------
#
# Each KV head's V is a constant equal to its head index. Attention over a
# constant V returns that constant exactly, so output for Q head h must be
# h // (H_q // H_kv) everywhere. Any reader head-mapping bug is an exact,
# integer-level mismatch.


@pytest.mark.parametrize(
    "H_q, H_kv",
    [(8, 2), (8, 1), (12, 4), (8, 8)],
    ids=["gqa_4to1", "mqa", "gqa_3to1", "mha_control"],
)
def test_head_mapping_exact(device, H_q, H_kv):
    B, S, D = 1, 64, 64
    torch.manual_seed(0)
    q = torch.randn(B, H_q, S, D)
    k = torch.randn(B, H_kv, S, D)
    v = torch.zeros(B, H_kv, S, D)
    for kv_h in range(H_kv):
        v[:, kv_h] = float(kv_h)

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tt_q, tt_k, tt_v)).float()

    ratio = H_q // H_kv
    for h in range(H_q):
        expected = float(h // ratio)
        head = out[:, h]
        # bf16 probs quantization gives ~1.5% relative error on the constant
        # (e.g. 3.94 vs 4.0); head-mapping bugs are off by >= 1.0 — round to
        # identify the head, with a wide rel tolerance guard.
        assert torch.allclose(head, torch.full_like(head, expected), rtol=0.05, atol=0.1), (
            f"Q head {h}: expected KV head {h // ratio} constant {expected}, "
            f"got min={head.min():.3f} max={head.max():.3f}"
        )
