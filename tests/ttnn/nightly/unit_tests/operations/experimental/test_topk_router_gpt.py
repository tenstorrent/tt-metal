# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for the fused topk_router_gpt operation.

The fused op computes:  matmul + bias → logits → topk(k=4) → softmax
Output: [B, 64] packed as 2 tiles:
  - Tile 0 (cols 0-31):  softmax weights in columns 0..k-1
  - Tile 1 (cols 32-63): expert indices (as bf16) in columns 0..k-1
"""

import torch
import torch.nn.functional as F
import ttnn
from loguru import logger


def run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k=4, untilize_output=False):
    """Run the fused op and return the torch result."""
    # Pre-broadcast bias to [B, N] so all tile rows contain the bias vector.
    torch_bias_bcast = torch_bias.expand(B, N).contiguous()
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    out_layout = ttnn.ROW_MAJOR_LAYOUT if untilize_output else ttnn.TILE_LAYOUT
    # Output: [B, 64] packed (tile 0 / cols 0-31 = weights, tile 1 / cols 32-63 = indices)
    tt_output = ttnn.from_torch(
        torch.zeros(B, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        layout=out_layout,
    )

    result, _, _, _ = ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        output_tensor=tt_output,
        k=k,
        num_experts=N,
        untilize_output=untilize_output,
    )

    tt_result = ttnn.to_torch(result)[:B, :64]
    # Unpack: first 32 cols = tile 0 (weights), next 32 cols = tile 1 (indices)
    weights = tt_result[:, :k].float()
    indices = tt_result[:, 32 : 32 + k].float().long()
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
        # Test 4: Untilized (ROW_MAJOR) output — verify matches tile output
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 4: untilize_output=True (ROW_MAJOR output)")
        torch.manual_seed(42)
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        # Run with TILE output (reference)
        weights_tile, indices_tile = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)
        # Run with ROW_MAJOR output
        weights_rm, indices_rm = run_fused_op(
            device, torch_input, torch_weight, torch_bias, B, K, N, k, untilize_output=True
        )

        logger.info(f"  TILE indices row 0: {indices_tile[0].tolist()}")
        logger.info(f"  RM   indices row 0: {indices_rm[0].tolist()}")
        logger.info(f"  TILE weights row 0: {[f'{w:.4f}' for w in weights_tile[0].tolist()]}")
        logger.info(f"  RM   weights row 0: {[f'{w:.4f}' for w in weights_rm[0].tolist()]}")

        rm_idx_match = (indices_rm == indices_tile).all().item()
        rm_weight_pcc = compute_pcc(weights_tile, weights_rm)
        logger.info(f"  Indices exact match (RM vs TILE): {rm_idx_match}")
        logger.info(f"  Weight PCC (RM vs TILE): {rm_weight_pcc:.6f}")

        # ================================================================
        # Test 5: TILE output → typecast + to_layout(RM) chain
        # Verifies the same conversion used in _fused_call throughput path:
        # slice [B,k] TILE bf16 → typecast uint32 → uint16 → to_layout(RM)
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 5: TILE output → typecast + RM conversion for throughput dispatch")
        torch.manual_seed(42)
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        # Run fused op with TILE output (default)
        weights_tile, indices_tile = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k)

        # Now run with the same TILE output but apply the typecast chain from _fused_call
        torch_bias_bcast = torch_bias.expand(B, N).contiguous()
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_output = ttnn.from_torch(
            torch.zeros(B, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        result, _, _, _ = ttnn.experimental.topk_router_gpt(
            tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            output_tensor=tt_output,
            k=k,
            num_experts=N,
        )
        # Slice [B, k] TILE → typecast → to_layout(RM) (same as _fused_call throughput path)
        expert_indices = ttnn.slice(result, [0, 32], [B, 32 + k])  # [B, k] TILE bf16
        expert_indices = ttnn.typecast(expert_indices, dtype=ttnn.uint32)
        expert_indices = ttnn.typecast(expert_indices, dtype=ttnn.uint16)
        expert_indices = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        expert_weights = ttnn.slice(result, [0, 0], [B, k])  # [B, k] TILE bf16
        expert_weights = ttnn.to_layout(expert_weights, ttnn.ROW_MAJOR_LAYOUT)

        # Convert to torch for comparison
        indices_torch = ttnn.to_torch(expert_indices)[:B, :k].long()
        weights_torch = ttnn.to_torch(expert_weights)[:B, :k].float()

        # Compare to the TILE reference (Test 2 same seed) to verify typecast preserves data
        tc_idx_match = (indices_torch == indices_tile).all().item()
        tc_weight_pcc = compute_pcc(weights_tile, weights_torch)
        logger.info(f"  Typecast indices row 0: {indices_torch[0].tolist()}")
        logger.info(f"  TILE ref  indices row 0: {indices_tile[0].tolist()}")
        logger.info(f"  Indices exact match (typecast vs TILE): {tc_idx_match}")
        logger.info(f"  Weight PCC (typecast vs TILE): {tc_weight_pcc:.6f}")

        # ================================================================
        # Test 6: produce_hidden_rm — verify dispatch outputs (indices uint16 RM, weights bf16 RM)
        # ================================================================
        logger.info("=" * 60)
        logger.info("TEST 6: produce_hidden_rm (dispatch outputs: indices + weights)")
        torch.manual_seed(42)
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

        torch_bias_bcast = torch_bias.expand(B, N).contiguous()
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_output = ttnn.from_torch(
            torch.zeros(B, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        result, _, indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
            tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            output_tensor=tt_output,
            k=k,
            num_experts=N,
            produce_hidden_rm=True,
        )

        # Verify dispatch outputs: indices_rm (uint16) and weights_rm (bf16)
        indices_rm_torch = ttnn.to_torch(indices_rm)[:B, :k].long()
        weights_rm_torch = ttnn.to_torch(weights_rm)[:B, :k].float()

        # Compare to the packed result (tile-based reference)
        tt_result = ttnn.to_torch(result)[:B, :64]
        ref_weights = tt_result[:, :k].float()
        ref_indices = tt_result[:, 32 : 32 + k].float().long()

        dispatch_idx_match = (indices_rm_torch == ref_indices).all().item()
        dispatch_wgt_pcc = compute_pcc(ref_weights, weights_rm_torch)
        logger.info(f"  indices_rm dtype: {indices_rm.dtype}")
        logger.info(f"  indices_rm row 0: {indices_rm_torch[0].tolist()}")
        logger.info(f"  ref       row 0: {ref_indices[0].tolist()}")
        logger.info(f"  Dispatch indices match: {dispatch_idx_match}")
        logger.info(f"  Dispatch weights PCC: {dispatch_wgt_pcc:.6f}")

        hrm_pass = dispatch_idx_match and dispatch_wgt_pcc >= 0.999

        # ================================================================
        # Summary
        # ================================================================
        logger.info("=" * 60)
        rm_pass = rm_idx_match and rm_weight_pcc >= 0.999
        tc_pass = tc_idx_match and tc_weight_pcc >= 0.95
        if weight_pcc >= 0.95 and idx_match_pct >= 90 and rm_pass and tc_pass and hrm_pass:
            logger.info("PASSED")
        else:
            logger.error(
                f"FAILED: weight_pcc={weight_pcc:.4f}, idx_match={idx_match_pct:.1f}%, "
                f"rm_idx_match={rm_idx_match}, rm_weight_pcc={rm_weight_pcc:.4f}, "
                f"tc_idx_match={tc_idx_match}, tc_weight_pcc={tc_weight_pcc:.4f}, "
                f"dispatch_idx_match={dispatch_idx_match}, dispatch_wgt_pcc={dispatch_wgt_pcc:.4f}"
            )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
