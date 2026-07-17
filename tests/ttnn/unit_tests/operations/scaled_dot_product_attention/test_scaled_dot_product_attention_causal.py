# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — causal masking (mask_mode = causal).

DO NOT DELETE — documents the R4 causal-masking surface.

The triangular −∞ bias is generated ON-DEVICE from the is_causal compile-time
flag (no mask tensor):
  * block-skip whole future KV chunks (≈half the KV work for causal self-attn);
  * per-element diagonal mask on the straddling chunk(s), applied before the
    row-max via the same additive-mask compute path R1's KV-padding mask rides.

Reference is torch.nn.functional.scaled_dot_product_attention(is_causal=True)
(the same oracle the golden harness uses). causal requires S_q == S_kv, so the
op refuses causal + cross via an EXCLUSION; is_causal ∧ attn_mask is a ValueError.
"""

import pytest
import torch

import ttnn
from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Golden bf16 + fp32-DEST tolerances (helpers.TOLERANCES[(bfloat16, True)]).
PCC = 0.995
REL_RMS = 0.05


def _check(ref, out):
    ref = ref.float()
    out = out.float()
    rf, of = ref.flatten() - ref.flatten().mean(), out.flatten() - out.flatten().mean()
    pcc = (rf @ of / (rf.norm() * of.norm() + 1e-12)).item()
    rel_rms = (torch.sqrt(((ref - out) ** 2).mean()) / (ref.std() + 1e-12)).item()
    assert pcc >= PCC, f"PCC {pcc:.5f} < {PCC}  (rel_rms={rel_rms:.4f})"
    assert rel_rms <= REL_RMS, f"rel_rms {rel_rms:.4f} > {REL_RMS}  (pcc={pcc:.5f})"


def causal_reference(q, k, v, scale=None):
    """fp32 torch causal SDPA reference (matches the golden oracle). Head-broadcasts
    K/V for GQA/MQA (torch F.sdpa does not auto-broadcast heads)."""
    q, k, v = q.float(), k.float(), v.float()
    H_q, H_kv = q.shape[1], k.shape[1]
    if H_q != H_kv:
        rep = H_q // H_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _to_device(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def run_causal(device, q_shape, k_shape, v_shape, *, dtype=ttnn.bfloat16, scale_mode="auto", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    v = torch.randn(v_shape)
    scale = None if scale_mode == "auto" else 0.25

    ref = causal_reference(q, k, v, scale=scale)

    tq = _to_device(q, device, dtype)
    tk = _to_device(k, device, dtype)
    tv = _to_device(v, device, dtype)

    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, scale=scale)
    assert list(out.shape) == list(q_shape), f"shape {list(out.shape)} != {list(q_shape)}"
    _check(ref, ttnn.to_torch(out))


# Causal self-attention only (S_q == S_kv). Covers single-tile, multi-tile (block
# skip active), multi-head, multi-batch, GQA, MQA, long-context, and the
# non-power-of-2 head counts (H > 1 core).
SELF = {
    "single_tile": ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    "two_tile": ((1, 1, 64, 64), (1, 1, 64, 64), (1, 1, 64, 64)),
    "multi_tile": ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    "s256": ((1, 1, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),
    "s512": ((1, 2, 512, 64), (1, 2, 512, 64), (1, 2, 512, 64)),
    "s1024": ((1, 1, 1024, 64), (1, 1, 1024, 64), (1, 1, 1024, 64)),
    "multi_head_batch": ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
    "d128": ((1, 4, 256, 128), (1, 4, 256, 128), (1, 4, 256, 128)),
    "gqa_4to1": ((1, 8, 256, 64), (1, 2, 256, 64), (1, 2, 256, 64)),
    "mqa": ((1, 8, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),
    "mqa_falcon": ((1, 71, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),  # H_q > cores
    "nonpow2_heads": ((1, 3, 192, 96), (1, 3, 192, 96), (1, 3, 192, 96)),
}


@pytest.mark.parametrize("key", list(SELF.keys()))
def test_causal_self(device, key):
    q, k, v = SELF[key]
    run_causal(device, q, k, v)


@pytest.mark.parametrize("key", ["multi_tile", "s512", "gqa_4to1"])
def test_causal_self_explicit_scale(device, key):
    q, k, v = SELF[key]
    run_causal(device, q, k, v, scale_mode="explicit")


def test_causal_all_ones(device):
    """All-ones causal self-attn: query i attends uniformly to keys 0..i, so the
    output is the running mean of V (= ones) = 1 everywhere. Deterministic."""
    q = torch.ones(1, 1, 128, 64, dtype=torch.bfloat16)
    tq = _to_device(q, device, ttnn.bfloat16)
    out = scaled_dot_product_attention(tq, tq, tq, is_causal=True)
    res = ttnn.to_torch(out).float()
    assert torch.allclose(
        res, torch.ones_like(res), atol=0.05
    ), f"max diff {(res - torch.ones_like(res)).abs().max().item()}"


def test_causal_cross_is_excluded(device, expect_error):
    """causal requires S_q == S_kv; a cross shape (S_q != S_kv) is refused."""
    q = _to_device(torch.randn(1, 4, 64, 64), device, ttnn.bfloat16)
    k = _to_device(torch.randn(1, 4, 128, 64), device, ttnn.bfloat16)
    with expect_error(ExcludedCell, ".*"):
        scaled_dot_product_attention(q, k, k, is_causal=True)


def test_causal_and_mask_mutually_exclusive(device, expect_error):
    """is_causal=True and attn_mask are mutually exclusive (ValueError)."""
    q = _to_device(torch.randn(1, 1, 64, 64), device, ttnn.bfloat16)
    m = _to_device(torch.zeros(1, 1, 64, 64), device, ttnn.bfloat16)
    with expect_error(ValueError, ".*mutually exclusive.*"):
        scaled_dot_product_attention(q, q, q, attn_mask=m, is_causal=True)
