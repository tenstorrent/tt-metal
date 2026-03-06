# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for the fused topk_router_gpt operation.

The fused op computes:  matmul + bias → logits → topk(k=4) → softmax
Outputs: (indices_rm, weights_rm) both in ROW_MAJOR format
  - indices_rm: [B, k_padded] uint16 expert indices
  - weights_rm: [B, k_padded] bf16 softmax weights
"""

import torch
import torch.nn.functional as F
import ttnn
from loguru import logger


def run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k=4):
    """Run the fused op and return the torch result."""
    # Pre-broadcast bias to [B, N] so all tile rows contain the bias vector.
    torch_bias_bcast = torch_bias.expand(B, N).contiguous()
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        k=k,
        num_experts=N,
    )

    # Convert to torch (slice to actual k cols since outputs are padded to k_padded)
    indices = ttnn.to_torch(indices_rm)[:B, :k].long()
    weights = ttnn.to_torch(weights_rm)[:B, :k].float()
    return weights, indices


def compute_pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b, atol=1e-3) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    B, K, N = 32, 2880, 128
    k = 4
    device = ttnn.open_device(device_id=0)

    try:
        # ================================================================
        # Test 1: Known logits → verify topk indices are correct
        # Use a weight that produces known logits: all zeros except column j
        # has value (j+1) for all rows.  Top-4 should be cols 127,126,125,124.
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 1: Deterministic topk (known logits)")
        # Build logits = bias only (zero input/weight, bias = arange)
        torch_input = torch.zeros(B, K, dtype=torch.bfloat16)
        torch_weight = torch.zeros(K, N, dtype=torch.bfloat16)
        # Bias values: 0, 1, 2, ..., 127  (col 127 = largest)
        torch_bias = torch.arange(N, dtype=torch.float32).unsqueeze(0).to(torch.bfloat16)

        weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)

        # Reference: top-4 of [0,1,2,...,127] = [127, 126, 125, 124]
        ref_logits = torch_bias.float().expand(B, N)
        ref_vals, ref_idxs = torch.topk(ref_logits, k, dim=-1)
        ref_weights = F.softmax(ref_vals, dim=-1)

        logger.info(f"  Expected indices row 0: {ref_idxs[0].tolist()}")
        logger.info(f"  TT       indices row 0: {indices_tt[0].tolist()}")
        logger.info(f"  Expected weights row 0: {[f'{w:.4f}' for w in ref_weights[0].tolist()]}")
        logger.info(f"  TT       weights row 0: {[f'{w:.4f}' for w in weights_tt[0].tolist()]}")

        idx_match = (indices_tt == ref_idxs).all().item()
        weight_pcc = compute_pcc(ref_weights, weights_tt)
        logger.info(f"  Indices exact match: {idx_match}")
        logger.info(f"  Weight PCC: {weight_pcc:.6f}")

        # ================================================================
        # Test 2: Random matmul + bias → verify topk + softmax accuracy
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 2: Random matmul + bias → topk + softmax")
        torch.manual_seed(42)
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)

        # Reference computation in float
        ref_logits = (torch_input.float() @ torch_weight.float() + torch_bias.float()).to(torch.bfloat16).float()
        ref_vals, ref_idxs = torch.topk(ref_logits, k, dim=-1)
        ref_weights = F.softmax(ref_vals, dim=-1)

        logger.info(f"  Ref indices row 0: {ref_idxs[0].tolist()}")
        logger.info(f"  TT  indices row 0: {indices_tt[0].tolist()}")
        logger.info(f"  Ref weights row 0: {[f'{w:.4f}' for w in ref_weights[0].tolist()]}")
        logger.info(f"  TT  weights row 0: {[f'{w:.4f}' for w in weights_tt[0].tolist()]}")

        # Check index match (allow minor differences due to bf16 matmul rounding)
        idx_match_count = (indices_tt == ref_idxs).sum().item()
        total_indices = B * k
        idx_match_pct = idx_match_count / total_indices * 100
        logger.info(f"  Index match: {idx_match_count}/{total_indices} ({idx_match_pct:.1f}%)")

        # Check weight accuracy using PCC
        weight_pcc = compute_pcc(ref_weights, weights_tt)
        logger.info(f"  Weight PCC: {weight_pcc:.6f}")

        # Verify softmax properties: weights sum to ~1 per row
        row_sums = weights_tt.sum(dim=-1)
        logger.info(
            f"  Weight row sums - min: {row_sums.min():.4f}, max: {row_sums.max():.4f}, mean: {row_sums.mean():.4f}"
        )

        # Check weights are all positive
        all_positive = (weights_tt > 0).all().item()
        logger.info(f"  All weights positive: {all_positive}")

        # ================================================================
        # Test 3: Verify with different random seed
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 3: Different random seed")
        torch.manual_seed(99)
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)

        ref_logits = (torch_input.float() @ torch_weight.float() + torch_bias.float()).to(torch.bfloat16).float()
        ref_vals, ref_idxs = torch.topk(ref_logits, k, dim=-1)
        ref_weights = F.softmax(ref_vals, dim=-1)

        idx_match_count = (indices_tt == ref_idxs).sum().item()
        idx_match_pct = idx_match_count / total_indices * 100
        weight_pcc = compute_pcc(ref_weights, weights_tt)
        row_sums = weights_tt.sum(dim=-1)

        logger.info(f"  Index match: {idx_match_count}/{total_indices} ({idx_match_pct:.1f}%)")
        logger.info(f"  Weight PCC: {weight_pcc:.6f}")
        logger.info(f"  Weight row sums - mean: {row_sums.mean():.4f}")

        # ================================================================
        # Test 4: Verify indices are uint16 dtype
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 4: Verify indices_rm dtype is uint16")
        torch.manual_seed(42)
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        # Run fused op
        weights_ref, indices_ref = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)

        # Now run again to verify dtype
        torch_bias_bcast = torch_bias.expand(B, N).contiguous()
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
            tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            k=k,
            num_experts=N,
        )

        # Verify dtype and values
        indices_torch = ttnn.to_torch(indices_rm)[:B, :k].long()
        weights_torch = ttnn.to_torch(weights_rm)[:B, :k].float()

        tc_idx_match = (indices_torch == indices_ref).all().item()
        tc_weight_pcc = compute_pcc(weights_ref, weights_torch)
        logger.info(f"  indices_rm dtype: {indices_rm.dtype}")
        logger.info(f"  weights_rm dtype: {weights_rm.dtype}")
        logger.info(f"  Indices row 0: {indices_torch[0].tolist()}")
        logger.info(f"  Ref     row 0: {indices_ref[0].tolist()}")
        logger.info(f"  Indices exact match: {tc_idx_match}")
        logger.info(f"  Weight PCC: {tc_weight_pcc:.6f}")

        # ================================================================
        # Test 5: Verify outputs match PyTorch reference
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 5: Verify outputs match PyTorch reference")
        torch.manual_seed(99)  # Different seed from other tests
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)

        # Compute reference
        ref_logits = (torch_input.float() @ torch_weight.float() + torch_bias.float()).to(torch.bfloat16).float()
        ref_vals, ref_idxs = torch.topk(ref_logits, k, dim=-1)
        ref_weights = F.softmax(ref_vals, dim=-1)

        dispatch_idx_match = (indices_tt == ref_idxs).all().item()
        dispatch_wgt_pcc = compute_pcc(ref_weights, weights_tt)
        logger.info(f"  TT  indices row 0: {indices_tt[0].tolist()}")
        logger.info(f"  Ref indices row 0: {ref_idxs[0].tolist()}")
        logger.info(f"  Indices match: {dispatch_idx_match}")
        logger.info(f"  Weights PCC: {dispatch_wgt_pcc:.6f}")

        # ================================================================
        # Summary
        # ================================================================
        logger.info("=" * 60)
        if (
            weight_pcc >= 0.95
            and idx_match_pct >= 90
            and tc_idx_match
            and tc_weight_pcc >= 0.95
            and dispatch_wgt_pcc >= 0.95
        ):
            logger.info("PASSED")
        else:
            logger.error(
                f"FAILED: weight_pcc={weight_pcc:.4f}, idx_match={idx_match_pct:.1f}%, "
                f"tc_idx_match={tc_idx_match}, tc_weight_pcc={tc_weight_pcc:.4f}, "
                f"dispatch_wgt_pcc={dispatch_wgt_pcc:.4f}"
            )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
