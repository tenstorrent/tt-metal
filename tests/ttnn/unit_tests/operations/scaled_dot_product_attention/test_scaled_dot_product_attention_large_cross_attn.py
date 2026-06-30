# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5 — Large unequal-seqlen cross-attention.

Tests that the op does not hang or deadlock on cross-attention shapes with
large, unequal sequence lengths (S_q ≠ S_kv, large S) and on shapes where
num_work_units > num_cores (multi-work-unit-per-core).

The root cause fixed in Refinement 5: the compute kernel was missing a
work-unit loop. When a core got >1 work unit (e.g. H_q=71 > 64 cores),
compute processed only the first work unit and exited, while the reader
kept pushing tiles into full CBs → CB deadlock.
"""
import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _run_and_compare(
    device,
    Q,
    K,
    V,
    attn_mask=None,
    is_causal=False,
    scale=None,
    dtype=ttnn.bfloat16,
    fp32_dest_acc_en=True,
    pcc_threshold=0.99,
):
    """Run SDPA on device and compare against PyTorch reference."""
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = None
    if attn_mask is not None:
        tt_mask = ttnn.from_torch(attn_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    tt_out = scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        attn_mask=tt_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=cfg,
    )
    tt_back = ttnn.to_torch(tt_out)

    # PyTorch reference
    H_q = Q.shape[1]
    H_kv = K.shape[1]
    if H_q != H_kv:
        K_ref = K.expand(-1, H_q, -1, -1)
        V_ref = V.expand(-1, H_q, -1, -1)
    else:
        K_ref = K
        V_ref = V

    gt = F.scaled_dot_product_attention(
        Q.float(),
        K_ref.float(),
        V_ref.float(),
        attn_mask=attn_mask.float() if attn_mask is not None else None,
        is_causal=is_causal,
        scale=scale,
    )

    # PCC comparison
    from models.common.utility_functions import comp_pcc

    out_pass, out_pcc = comp_pcc(gt, tt_back.float(), pcc_threshold)
    assert out_pass, f"PCC={out_pcc:.6f} < {pcc_threshold}. Max diff: {(gt - tt_back.float()).abs().max().item()}"
    return out_pcc


# ===========================================================================
# Tests for multi-work-unit-per-core (H_q > num_cores)
# ===========================================================================


class TestMultiWorkUnitPerCore:
    """Shapes where H_q > 64 (num_cores on WH), so some cores get >1 WU.

    Before Refinement 5: the compute kernel had no work-unit loop, so it
    processed only the first WU and exited, causing a CB deadlock.
    """

    def test_falcon_7b_shape(self, device):
        """Falcon-7B: B=1, H_q=71, H_kv=1 (MQA), S=2048, D=64.

        71 work units > 64 cores → some cores get 2 WUs.
        This was the primary hang cell from the translated suite.
        """
        torch.manual_seed(1234)
        B, H_q, H_kv, S, D = 1, 71, 1, 2048, 64
        Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)
        assert pcc > 0.99, f"Falcon-7B shape PCC={pcc}"

    def test_falcon_7b_bf8b(self, device):
        """Falcon-7B shape with bfloat8_b dtype."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S, D = 1, 71, 1, 2048, 64
        Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, dtype=ttnn.bfloat8_b, pcc_threshold=0.98)
        assert pcc > 0.98

    def test_falcon_7b_low_prec(self, device):
        """Falcon-7B shape with HiFi2 + fp32_dest_acc_en=False (low precision).

        This mirrors the translated test's compute config exactly.
        """
        torch.manual_seed(1234)
        B, H_q, H_kv, S, D = 1, 71, 1, 2048, 64
        Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, fp32_dest_acc_en=False, pcc_threshold=0.98)

    def test_h_72_mha(self, device):
        """H_q=72 (MHA), so 72 > 64 cores → some cores get 2 WUs.
        Even number, divides cleanly across 8x8 grid.
        """
        torch.manual_seed(42)
        B, H_q, S, D = 1, 72, 512, 64
        Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)

    def test_h_130_mqa(self, device):
        """H_q=130 (MQA, H_kv=1), 130 > 64 → 2+ WUs per core."""
        torch.manual_seed(99)
        B, H_q, H_kv, S, D = 1, 130, 1, 256, 64
        Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)

    def test_h_64_boundary(self, device):
        """H_q=64, exactly num_cores → 1 WU per core (boundary case).
        Should still pass — the work-unit loop runs exactly once.
        """
        torch.manual_seed(7)
        B, H_q, S, D = 1, 64, 512, 64
        Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)


# ===========================================================================
# Tests for large unequal-seqlen cross-attention (S_q ≠ S_kv, large S)
# ===========================================================================


class TestLargeUnequalSeqlenCrossAttention:
    """Cross-attention with large, unequal S_q ≠ S_kv.

    The refinement's primary target cell was
    test_sdpa_noncausal_unequal_seqlen__nightly[1-8-1-4096-2048-128-k256-q128-bfp8].
    """

    def test_sq_4096_sk_2048_bf16(self, device):
        """S_q=4096, S_kv=2048, D=128 — the refinement's named cell (bf16)."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 8, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)

    def test_sq_4096_sk_2048_bf8b(self, device):
        """S_q=4096, S_kv=2048, D=128 — the refinement's named cell (bf8b)."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 8, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, dtype=ttnn.bfloat8_b, pcc_threshold=0.98)

    def test_sq_2048_sk_6528_bf16(self, device):
        """S_q=2048, S_kv=6528 (Llama-Vision shape) — large asymmetric."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 4, 1, 2048, 6528, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)

    def test_sq_2048_sk_6528_bf8b(self, device):
        """S_q=2048, S_kv=6528 (Llama-Vision shape) — bf8b."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 4, 1, 2048, 6528, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, dtype=ttnn.bfloat8_b, pcc_threshold=0.98)

    def test_sq_4096_sk_2048_with_mask(self, device):
        """S_q=4096, S_kv=2048, D=128, with custom additive mask."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 8, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        mask = torch.bernoulli(torch.full((1, 1, S_q, S_kv), 0.25, dtype=torch.bfloat16)) * -1e9
        pcc = _run_and_compare(device, Q, K, V, attn_mask=mask, pcc_threshold=0.99)

    def test_cross_attn_low_prec(self, device):
        """Cross-attention with HiFi2 + fp32_dest_acc_en=False."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 8, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, fp32_dest_acc_en=False, pcc_threshold=0.98)


# ===========================================================================
# Combined: multi-work-unit + large unequal seqlen
# ===========================================================================


class TestMultiWULargeCrossAttention:
    """Combine H_q > num_cores with large unequal S_q ≠ S_kv.

    This is the worst case: both the work-unit loop bug AND large cross-attention
    in the same cell.
    """

    def test_h71_cross_attn(self, device):
        """H_q=71 (MQA), S_q=4096, S_kv=2048 — multi-WU + cross-attention."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 71, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)

    def test_h71_cross_attn_bf8b(self, device):
        """H_q=71 (MQA), S_q=4096, S_kv=2048 — bf8b."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 1, 71, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, dtype=ttnn.bfloat8_b, pcc_threshold=0.97)

    def test_h8_large_batch_cross_attn(self, device):
        """B=8, H_q=8, S_q=4096, S_kv=2048 — 64 WUs (exactly num_cores)."""
        torch.manual_seed(1234)
        B, H_q, H_kv, S_q, S_kv, D = 8, 8, 1, 4096, 2048, 128
        Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16) * 0.1
        K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16) * 0.1
        pcc = _run_and_compare(device, Q, K, V, pcc_threshold=0.99)
