# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import time
import ttnn
from loguru import logger


def ref_post_combine_reduce(combine_output: torch.Tensor, weights: torch.Tensor, expert_dim: int) -> torch.Tensor:
    """
    Reference implementation in PyTorch.
    combine_output: [1, seq_len, num_experts, emb_dim]
    weights: [1, seq_len, num_experts, 1]
    """
    weights_expanded = weights.expand(-1, -1, -1, combine_output.shape[-1])
    weighted = combine_output * weights_expanded
    return weighted.sum(dim=expert_dim)


def old_implementation(combine_output_tt, weights_tt, expert_dim):
    """
    Original implementation: to_layout(TILE) + mul + sum
    This is what tt_reduce.py does today.
    """
    # Convert combine_output from ROW_MAJOR to TILE (the to_layout we want to eliminate)
    combine_tiled = ttnn.to_layout(combine_output_tt, ttnn.TILE_LAYOUT)

    # Convert weights to TILE if needed, then unsqueeze for broadcast
    if weights_tt.layout != ttnn.TILE_LAYOUT:
        weights_tiled = ttnn.to_layout(weights_tt, ttnn.TILE_LAYOUT)
    else:
        weights_tiled = weights_tt

    # Broadcast multiply: [1, seq, experts, emb] * [1, seq, experts, 1]
    weighted = ttnn.mul(combine_tiled, weights_tiled)

    # Sum across expert dimension
    result = ttnn.sum(weighted, dim=expert_dim)
    return result


def new_implementation(combine_output_tt, weights_tt, expert_dim):
    """
    Fused implementation: deepseek_moe_post_combine_reduce
    Reads ROW_MAJOR directly, produces TILE output. No separate tilize needed.
    """
    return ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=expert_dim, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def benchmark(fn, combine_tt, weights_tt, expert_dim, num_iterations, device):
    """Run fn for num_iterations, return ms/iter."""
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fn(combine_tt, weights_tt, expert_dim)
    ttnn.synchronize_device(device)
    duration = time.perf_counter() - start
    return duration / num_iterations * 1000  # ms


@pytest.mark.parametrize(
    "batch_size,seq_len,num_experts,emb_dim,num_iterations",
    [
        (1, 3200, 8, 7168, 100),  # DeepSeek-V3 actual use case
    ],
)
def test_post_combine_reduce_performance(batch_size, seq_len, num_experts, emb_dim, num_iterations, device):
    """
    Performance comparison: old (to_layout + mul + sum) vs new (fused op).
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PERFORMANCE BENCHMARK: DeepSeek-V3 Post-Combine-Reduce")
    logger.info(f"   Shape: [{batch_size}, {seq_len}, {num_experts}, {emb_dim}]")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"{'='*80}")

    torch.manual_seed(42)
    combine_output_torch = torch.randn(batch_size, seq_len, num_experts, emb_dim, dtype=torch.bfloat16)
    weights_torch = torch.randn(batch_size, seq_len, num_experts, 1, dtype=torch.bfloat16)
    expert_dim = 2

    ref_result = ref_post_combine_reduce(combine_output_torch, weights_torch, expert_dim)

    # TTNN tensors - ROW_MAJOR for both (as they come from combine stage)
    combine_tt = ttnn.from_torch(
        combine_output_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights_tt = ttnn.from_torch(
        weights_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # --- Warm up both ---
    logger.info("Warming up old (to_layout + mul + sum)...")
    for _ in range(3):
        _ = old_implementation(combine_tt, weights_tt, expert_dim)
    ttnn.synchronize_device(device)

    logger.info("Warming up new (fused)...")
    for _ in range(3):
        _ = new_implementation(combine_tt, weights_tt, expert_dim)
    ttnn.synchronize_device(device)

    # --- Correctness check ---
    old_result = ttnn.to_torch(old_implementation(combine_tt, weights_tt, expert_dim))
    new_result = ttnn.to_torch(new_implementation(combine_tt, weights_tt, expert_dim))

    old_pcc = torch.corrcoef(torch.stack([old_result.flatten().float(), ref_result.flatten().float()]))[0, 1]
    new_pcc = torch.corrcoef(torch.stack([new_result.flatten().float(), ref_result.flatten().float()]))[0, 1]
    logger.info(f"Old PCC vs ref: {old_pcc:.6f}")
    logger.info(f"New PCC vs ref: {new_pcc:.6f}")

    # --- Benchmark old ---
    logger.info(f"Benchmarking old ({num_iterations} iters)...")
    old_ms = benchmark(old_implementation, combine_tt, weights_tt, expert_dim, num_iterations, device)

    # --- Benchmark new ---
    logger.info(f"Benchmarking new ({num_iterations} iters)...")
    new_ms = benchmark(new_implementation, combine_tt, weights_tt, expert_dim, num_iterations, device)

    # --- Results ---
    speedup = old_ms / new_ms
    input_mb = (batch_size * seq_len * num_experts * emb_dim * 2) / (1024 * 1024)
    output_mb = (batch_size * seq_len * emb_dim * 2) / (1024 * 1024)
    total_mb = input_mb + output_mb

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS:")
    logger.info(f"{'='*80}")
    logger.info(f"  Old (to_layout+mul+sum): {old_ms:.3f} ms/iter")
    logger.info(f"  New (fused):             {new_ms:.3f} ms/iter")
    logger.info(f"  Speedup:                 {speedup:.2f}x")
    logger.info(f"  Data moved:              {total_mb:.1f} MB")
    logger.info(f"  Old PCC: {old_pcc:.6f}  New PCC: {new_pcc:.6f}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_post_combine_reduce_performance(1, 3200, 8, 7168, 100, device)
    finally:
        ttnn.close_device(device)
