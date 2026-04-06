# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test deepseek_moe_post_combine_reduce with sparse weights (some experts zeroed out).

In real DeepSeek-V3, only top-k experts (e.g., 6 out of 256) are active per token.
This test simulates that by randomly zeroing out some expert weights.
"""

import torch
import ttnn
from loguru import logger


def ref_post_combine_reduce(combine_output, weights, expert_dim=2):
    weights_expanded = weights.expand(-1, -1, -1, combine_output.shape[-1])
    weighted = combine_output * weights_expanded
    return weighted.sum(dim=expert_dim)


def test_sparse_weights(device):
    """
    Test correctness and benchmark with sparse weights.
    Varies sparsity from 0% (all active) to 87.5% (1 of 8 active).
    """
    import time

    num_tokens = 3200
    num_experts = 8
    emb_dim = 7168
    expert_dim = 2
    num_iterations = 100

    torch.manual_seed(42)
    combine_output_torch = torch.randn(1, num_tokens, num_experts, emb_dim, dtype=torch.bfloat16)

    # Test different sparsity levels: k active experts out of 8
    for k_active in [8, 6, 4, 2, 1]:
        sparsity_pct = (1 - k_active / num_experts) * 100

        # Create sparse weights: for each token, randomly pick k_active experts
        weights_torch = torch.zeros(1, num_tokens, num_experts, 1, dtype=torch.bfloat16)
        for t in range(num_tokens):
            active_experts = torch.randperm(num_experts)[:k_active]
            weights_torch[0, t, active_experts, 0] = torch.randn(k_active, dtype=torch.bfloat16)

        # Reference
        ref_result = ref_post_combine_reduce(combine_output_torch, weights_torch, expert_dim)

        # TTNN
        combine_tt = ttnn.from_torch(
            combine_output_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        weights_tt = ttnn.from_torch(
            weights_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Correctness
        result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
            combine_tt, weights_tt, expert_dim=expert_dim, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        result_torch = ttnn.to_torch(result_tt)
        pcc = torch.corrcoef(torch.stack([result_torch.flatten().float(), ref_result.flatten().float()]))[0, 1]

        # Benchmark
        # Warmup
        for _ in range(3):
            _ = ttnn.experimental.deepseek_moe_post_combine_reduce(
                combine_tt, weights_tt, expert_dim=expert_dim, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        ttnn.synchronize_device(device)

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = ttnn.experimental.deepseek_moe_post_combine_reduce(
                combine_tt, weights_tt, expert_dim=expert_dim, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        ttnn.synchronize_device(device)
        duration = time.perf_counter() - start
        ms_per_iter = duration / num_iterations * 1000

        logger.info(
            f"  Active: {k_active}/{num_experts} ({sparsity_pct:.0f}% sparse) | "
            f"Latency: {ms_per_iter:.3f} ms | PCC: {pcc:.6f}"
        )

    logger.info("Done!")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_sparse_weights(device)
    finally:
        ttnn.close_device(device)
