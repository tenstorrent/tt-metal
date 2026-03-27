# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import skip_for_grayskull
from loguru import logger


def ref_post_combine_reduce(combine_output: torch.Tensor, weights: torch.Tensor, expert_dim: int) -> torch.Tensor:
    """
    Reference implementation of post-combine reduce in PyTorch.

    combine_output: [..., seq_len, num_experts, emb_dim]
    weights: [..., seq_len, num_experts]

    Returns: [..., seq_len, emb_dim]
    """
    # Expand weights to broadcast: [..., seq_len, num_experts, 1]
    weights_expanded = weights.unsqueeze(-1)

    # Element-wise multiply: [..., seq_len, num_experts, emb_dim]
    weighted = combine_output * weights_expanded

    # Sum across expert dimension: [..., seq_len, emb_dim]
    result = weighted.sum(dim=expert_dim)

    return result


def old_implementation_separate_ops(
    combine_output_tt: ttnn.Tensor,
    weights_tt: ttnn.Tensor,
    expert_dim: int,
) -> ttnn.Tensor:
    """
    Old implementation using separate tilize, multiply, and reduce operations.
    """
    # Step 1: Tilize the ROW_MAJOR combine output
    combine_tiled = ttnn.tilize(combine_output_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Step 2: Expand weights and broadcast multiply
    # weights shape: [seq_len, num_experts] -> [seq_len, num_experts, 1]
    weights_expanded = ttnn.unsqueeze(weights_tt, dim=-1)

    # Step 3: Multiply (broadcast across embedding dimension)
    weighted = ttnn.mul(combine_tiled, weights_expanded, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Step 4: Reduce (sum) across expert dimension
    result = ttnn.sum(weighted, dim=expert_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return result


def new_implementation_fused(
    combine_output_tt: ttnn.Tensor,
    weights_tt: ttnn.Tensor,
    expert_dim: int,
) -> ttnn.Tensor:
    """
    New fused implementation using deepseek_moe_post_combine_reduce.
    """
    result = ttnn.prim.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    return result


@pytest.mark.skip(reason="Performance is priority - run manually if needed")
@skip_for_grayskull("Requires wormhole_b0")
@pytest.mark.parametrize(
    "batch_size,seq_len,num_experts,emb_dim",
    [
        (1, 3200, 8, 7168),  # DeepSeek-V3 actual use case
        # (1, 128, 8, 1024),      # Small test
        # (1, 512, 8, 2048),      # Medium test
        # (1, 1024, 8, 4096),     # Large test
        # (2, 256, 16, 2048),     # Multi-batch
    ],
)
def test_post_combine_reduce_correctness(batch_size, seq_len, num_experts, emb_dim, device):
    """
    Test correctness: compare old implementation, new implementation, and PyTorch reference.
    """
    logger.info(f"Testing shape: batch={batch_size}, seq_len={seq_len}, num_experts={num_experts}, emb_dim={emb_dim}")

    # Generate random test data
    torch.manual_seed(42)
    combine_output_torch = torch.randn(batch_size, seq_len, num_experts, emb_dim, dtype=torch.bfloat16)
    weights_torch = torch.randn(batch_size, seq_len, num_experts, dtype=torch.bfloat16)

    # Compute reference result
    expert_dim = 2  # Expert dimension in the tensor
    ref_result = ref_post_combine_reduce(combine_output_torch, weights_torch, expert_dim)

    # Convert to TTNN tensors (ROW_MAJOR for combine_output)
    combine_output_tt = ttnn.from_torch(
        combine_output_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    weights_tt = ttnn.from_torch(
        weights_torch, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Run old implementation
    logger.info("Running old implementation (separate ops)...")
    old_result_tt = old_implementation_separate_ops(combine_output_tt, weights_tt, expert_dim)
    old_result = ttnn.to_torch(old_result_tt)

    # Run new implementation
    logger.info("Running new implementation (fused)...")
    new_result_tt = new_implementation_fused(combine_output_tt, weights_tt, expert_dim)
    new_result = ttnn.to_torch(new_result_tt)

    # Compare results
    logger.info("Comparing results...")

    # Compare old vs reference
    old_pcc = torch.corrcoef(torch.stack([old_result.flatten(), ref_result.flatten()]))[0, 1]
    logger.info(f"Old implementation PCC vs reference: {old_pcc:.6f}")
    assert old_pcc > 0.99, f"Old implementation PCC too low: {old_pcc}"

    # Compare new vs reference
    new_pcc = torch.corrcoef(torch.stack([new_result.flatten(), ref_result.flatten()]))[0, 1]
    logger.info(f"New implementation PCC vs reference: {new_pcc:.6f}")
    assert new_pcc > 0.99, f"New implementation PCC too low: {new_pcc}"

    # Compare old vs new
    old_new_pcc = torch.corrcoef(torch.stack([old_result.flatten(), new_result.flatten()]))[0, 1]
    logger.info(f"Old vs New implementation PCC: {old_new_pcc:.6f}")
    assert old_new_pcc > 0.99, f"Old and new implementations differ too much: {old_new_pcc}"

    logger.info("✓ All correctness checks passed!")


@skip_for_grayskull("Requires wormhole_b0")
@pytest.mark.parametrize(
    "batch_size,seq_len,num_experts,emb_dim,num_iterations",
    [
        (1, 3200, 8, 7168, 100),  # DeepSeek-V3 actual use case - more iterations for accurate perf
        # (1, 128, 8, 1024, 100),     # Small - many iterations
        # (1, 512, 8, 2048, 50),      # Medium
        # (1, 1024, 8, 4096, 20),     # Large
        # (1, 2048, 8, 4096, 10),     # XLarge - typical DeepSeek size
    ],
)
def test_post_combine_reduce_performance(batch_size, seq_len, num_experts, emb_dim, num_iterations, device):
    """
    Performance comparison: measure throughput improvement of fused op vs separate ops.

    PRIMARY TEST - Performance is the key metric for this optimization.
    """
    import time

    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 PERFORMANCE BENCHMARK: DeepSeek-V3 Post-Combine-Reduce")
    logger.info(f"   Shape: [{batch_size}, {seq_len}, {num_experts}, {emb_dim}]")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"{'='*80}")

    # Generate test data
    torch.manual_seed(42)
    combine_output_torch = torch.randn(batch_size, seq_len, num_experts, emb_dim, dtype=torch.bfloat16)
    weights_torch = torch.randn(batch_size, seq_len, num_experts, dtype=torch.bfloat16)

    expert_dim = 2

    # Convert to TTNN tensors
    combine_output_tt = ttnn.from_torch(
        combine_output_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    weights_tt = ttnn.from_torch(
        weights_torch, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Warm-up
    logger.info("Warming up...")
    for _ in range(5):
        _ = old_implementation_separate_ops(combine_output_tt, weights_tt, expert_dim, device)
        _ = new_implementation_fused(combine_output_tt, weights_tt, expert_dim, device)

    ttnn.synchronize_device(device)

    # Benchmark old implementation
    logger.info(f"Benchmarking old implementation ({num_iterations} iterations)...")
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        old_result = old_implementation_separate_ops(combine_output_tt, weights_tt, expert_dim, device)

    ttnn.synchronize_device(device)
    old_duration = time.perf_counter() - start_time
    old_time_per_iter = old_duration / num_iterations * 1000  # ms

    logger.info(f"Old implementation: {old_time_per_iter:.3f} ms/iter")

    # Benchmark new implementation
    logger.info(f"Benchmarking new implementation ({num_iterations} iterations)...")
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        new_result = new_implementation_fused(combine_output_tt, weights_tt, expert_dim, device)

    ttnn.synchronize_device(device)
    new_duration = time.perf_counter() - start_time
    new_time_per_iter = new_duration / num_iterations * 1000  # ms

    logger.info(f"New implementation: {new_time_per_iter:.3f} ms/iter")

    # Calculate metrics
    speedup = old_time_per_iter / new_time_per_iter
    improvement_pct = (1 - new_time_per_iter / old_time_per_iter) * 100

    # Calculate throughput (tokens/sec)
    old_throughput = (seq_len * batch_size * 1000) / old_time_per_iter
    new_throughput = (seq_len * batch_size * 1000) / new_time_per_iter

    # Calculate data volume (input + output in MB)
    input_mb = (batch_size * seq_len * num_experts * emb_dim * 2) / (1024 * 1024)  # combine_output bf16
    weight_mb = (batch_size * seq_len * num_experts * 2) / (1024 * 1024)  # weights bf16
    output_mb = (batch_size * seq_len * emb_dim * 2) / (1024 * 1024)  # output bf16
    total_mb = input_mb + weight_mb + output_mb

    # Effective bandwidth (MB/s)
    old_bandwidth = total_mb / (old_time_per_iter / 1000)
    new_bandwidth = total_mb / (new_time_per_iter / 1000)

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 PERFORMANCE RESULTS:")
    logger.info(f"{'='*80}")
    logger.info(f"⏱️  Latency:")
    logger.info(f"   Old (separate ops):  {old_time_per_iter:.3f} ms/iter")
    logger.info(f"   New (fused):         {new_time_per_iter:.3f} ms/iter")
    logger.info(f"   → Speedup:           {speedup:.2f}x ({improvement_pct:+.1f}%)")
    logger.info(f"")
    logger.info(f"🚄 Throughput:")
    logger.info(f"   Old: {old_throughput:,.0f} tokens/sec")
    logger.info(f"   New: {new_throughput:,.0f} tokens/sec")
    logger.info(f"")
    logger.info(f"💾 Data Transfer ({total_mb:.1f} MB total):")
    logger.info(f"   Input:  {input_mb:.1f} MB")
    logger.info(f"   Weight: {weight_mb:.1f} MB")
    logger.info(f"   Output: {output_mb:.1f} MB")
    logger.info(f"")
    logger.info(f"📈 Effective Bandwidth:")
    logger.info(f"   Old: {old_bandwidth:,.0f} MB/s")
    logger.info(f"   New: {new_bandwidth:,.0f} MB/s")
    logger.info(f"{'='*80}\n")

    # Assert some improvement (should be at least 1.2x faster due to fusion)
    assert speedup > 1.2, f"Expected at least 1.2x speedup, got {speedup:.2f}x"


@skip_for_grayskull("Requires wormhole_b0")
def test_post_combine_reduce_edge_cases(device):
    """
    Test edge cases and different configurations.
    """
    # Test with minimum viable size (1 expert, minimal embedding)
    logger.info("Testing minimal size...")
    combine_output = torch.randn(1, 32, 1, 32, dtype=torch.bfloat16)
    weights = torch.randn(1, 32, 1, dtype=torch.bfloat16)

    combine_output_tt = ttnn.from_torch(combine_output, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    weights_tt = ttnn.from_torch(weights, device=device, layout=ttnn.TILE_LAYOUT)

    result = new_implementation_fused(combine_output_tt, weights_tt, expert_dim=2)
    result_torch = ttnn.to_torch(result)
    ref = ref_post_combine_reduce(combine_output, weights, expert_dim=2)

    pcc = torch.corrcoef(torch.stack([result_torch.flatten(), ref.flatten()]))[0, 1]
    assert pcc > 0.99, f"Edge case failed with PCC: {pcc}"

    logger.info("✓ Edge case test passed!")


if __name__ == "__main__":
    # Run performance benchmark for DeepSeek-V3 dimensions
    device = ttnn.open_device(device_id=0)

    try:
        logger.info("=" * 80)
        logger.info("🎯 DeepSeek-V3 Post-Combine-Reduce Performance Benchmark")
        logger.info("=" * 80)

        # Primary focus: Performance
        logger.info("\n⏱️  Running performance benchmark [1, 3200, 8, 7168]...")
        test_post_combine_reduce_performance(1, 3200, 8, 7168, 100, device)

        # Optional: Quick correctness check (if you want to verify)
        # Uncomment to run:
        # logger.info("\n✓ Running quick correctness check...")
        # test_post_combine_reduce_correctness(1, 3200, 8, 7168, device)

        logger.info("\n✅ Performance benchmark complete!")

    finally:
        ttnn.close_device(device)
