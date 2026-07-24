#!/usr/bin/env python3
"""
Generate GPT-OSS synthetic weights with router weights designed to avoid ties.
This ensures exact reproducibility between PyTorch and TTNN implementations.
"""

from typing import Dict, Optional

import numpy as np
import torch


def generate_gpt_oss_no_ties_weights(
    layer_idx: int = 0,
    num_experts: int = 128,
    hidden_size: int = 2880,
    intermediate_size: int = 2880,
    dtype: torch.dtype = torch.bfloat16,
    seed: Optional[int] = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic weights for GPT-OSS MoE layer with tie-free router.

    The router weights are designed to produce well-separated logits that avoid ties
    in topk selection, ensuring reproducible expert selection across implementations.

    Args:
        layer_idx: Layer index (affects weight initialization)
        num_experts: Number of experts (default 128 for GPT-OSS)
        hidden_size: Hidden dimension size (default 2880)
        intermediate_size: Expert intermediate size (default 2880 for GPT-OSS)
        dtype: Data type for weights (default bfloat16)
        seed: Random seed for reproducibility

    Returns:
        Dictionary of weight tensors matching GPT-OSS structure
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    weights = {}

    # ========================================
    # Router weights - DESIGNED TO AVOID TIES
    # ========================================
    # Key insight: With bfloat16, we need sufficient magnitude differences to avoid
    # quantization causing ties. We'll create diverse router weights so different
    # inputs naturally select different experts.

    # Start with random Gaussian weights (similar to real training)
    router_weight = torch.randn(num_experts, hidden_size, dtype=torch.float32) * 0.02

    # Add structured variation to each expert to ensure uniqueness
    for i in range(num_experts):
        # Each expert gets a unique "specialty" - stronger response to certain input patterns
        # Create frequency-based patterns (like Fourier basis)
        freq = (i + 1) * np.pi / num_experts
        phase = i * np.pi / (2 * num_experts)

        # Add sinusoidal pattern to break symmetry
        for j in range(hidden_size):
            router_weight[i, j] += 0.005 * np.sin(freq * j / hidden_size + phase)

        # Add sparse "expert signature" - each expert is particularly sensitive to specific dims
        expert_special_dims = torch.randperm(hidden_size)[:20]  # 20 special dimensions per expert
        router_weight[i, expert_special_dims] += torch.randn(20) * 0.01

    # Ensure sufficient magnitude for bfloat16 precision
    # Scale up to avoid quantization ties
    router_weight = router_weight * 2.0

    # Convert to target dtype
    router_weight = router_weight.to(dtype)

    # Router bias - smaller, diverse values
    # Don't let bias dominate - we want the input to determine expert selection
    router_bias = torch.randn(num_experts, dtype=torch.float32) * 0.1
    # Add small systematic variation to ensure no exact duplicates
    for i in range(num_experts):
        router_bias[i] += i * 0.001  # Small linear trend

    router_bias = router_bias.to(dtype)

    weights[f"model.layers.{layer_idx}.mlp.router.weight"] = router_weight
    weights[f"model.layers.{layer_idx}.mlp.router.bias"] = router_bias

    # Log router statistics
    print(f"\nNo-Ties Router Weight Statistics:")
    print(f"  Router weight shape: {router_weight.shape}")
    print(f"  Router weight range: [{router_weight.min():.6f}, {router_weight.max():.6f}]")
    print(f"  Router weight mean: {router_weight.mean():.6f}, std: {router_weight.std():.6f}")
    print(f"  Router bias range: [{router_bias.min():.6f}, {router_bias.max():.6f}]")
    print(f"  Router bias std: {router_bias.std():.6f}")

    # ========================================
    # Expert weights (same as original)
    # ========================================
    # Use same statistics as real GPT-OSS for expert weights
    gate_std = 0.020
    up_std = 0.020
    down_std = 0.020

    for expert_idx in range(num_experts):
        # Gate projection
        gate_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype) * gate_std
        weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = gate_weight

        # Up projection
        up_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype) * up_std
        weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = up_weight

        # Down projection
        down_weight = torch.randn(hidden_size, intermediate_size, dtype=dtype) * down_std
        weights[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = down_weight

    # Log sample expert statistics
    sample_key = f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
    if sample_key in weights:
        sample = weights[sample_key]
        print(f"\nExpert Weight Statistics (expert 0 gate_proj):")
        print(f"  Shape: {sample.shape}")
        print(f"  Mean: {sample.mean():.6f}")
        print(f"  Std: {sample.std():.6f}")
        print(f"  Range: [{sample.min():.4f}, {sample.max():.4f}]")

    return weights


def test_no_ties():
    """Test that the generated weights produce no ties in topk."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 80)
    print("Testing No-Ties Weight Generator")
    print("=" * 80)

    # Generate weights
    weights = generate_gpt_oss_no_ties_weights(seed=42)

    # Create test input
    torch.manual_seed(123)
    batch_size = 1
    seq_len = 128
    hidden_size = 2880
    num_experts = 128
    num_experts_per_tok = 4

    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16) * 0.1

    # Get router weights
    router_weight = weights["model.layers.0.mlp.router.weight"]
    router_bias = weights["model.layers.0.mlp.router.bias"]

    # Compute logits
    input_flat = input_tensor.view(-1, hidden_size)
    router_logits = input_flat @ router_weight.T + router_bias

    # Convert to float32 for analysis
    router_logits = router_logits.float()

    # Check for ties in top-k selection
    print("\nChecking for ties in router logits...")

    num_exact_ties = 0
    num_close_ties = 0
    tie_threshold = 0.001  # Consider values within this range as potential ties

    for token_idx in range(input_flat.shape[0]):
        token_logits = router_logits[token_idx]

        # Get top-k+2 to check for ties around the cutoff
        topk_values, topk_indices = torch.topk(token_logits, k=min(num_experts_per_tok + 2, num_experts))

        # Check for exact ties in top-k+2
        has_exact_tie = False
        has_close_tie = False

        for i in range(len(topk_values) - 1):
            diff = abs(topk_values[i] - topk_values[i + 1])
            if diff == 0:
                has_exact_tie = True
            elif diff < tie_threshold:
                has_close_tie = True

        if has_exact_tie:
            num_exact_ties += 1
        if has_close_tie:
            num_close_ties += 1

    print(
        f"  Tokens with exact ties in top-{num_experts_per_tok+2}: {num_exact_ties}/{seq_len} ({100*num_exact_ties/seq_len:.1f}%)"
    )
    print(
        f"  Tokens with close ties (<{tie_threshold}): {num_close_ties}/{seq_len} ({100*num_close_ties/seq_len:.1f}%)"
    )

    # Check diversity of expert selection
    topk_values, topk_indices = torch.topk(router_logits, k=num_experts_per_tok, dim=-1)

    # Count unique expert selections
    unique_selections = set()
    for i in range(topk_indices.shape[0]):
        selection = tuple(sorted(topk_indices[i].tolist()))
        unique_selections.add(selection)

    print(f"\nExpert selection diversity:")
    print(f"  Unique expert combinations: {len(unique_selections)}/{seq_len}")
    print(f"  Expert selection diversity: {100*len(unique_selections)/seq_len:.1f}%")

    # Calculate minimum separation in top-k
    min_separation = float("inf")
    avg_separation = 0
    count = 0

    for token_idx in range(topk_values.shape[0]):
        token_topk = topk_values[token_idx]
        for i in range(len(token_topk) - 1):
            sep = float(token_topk[i] - token_topk[i + 1])
            min_separation = min(min_separation, sep)
            avg_separation += sep
            count += 1

    avg_separation /= count if count > 0 else 1

    print(f"\nTop-K separation analysis:")
    print(f"  Minimum separation: {min_separation:.6f}")
    print(f"  Average separation: {avg_separation:.6f}")
    print(f"  Average top-k spread: {(topk_values[:, 0] - topk_values[:, -1]).mean():.4f}")

    # Show distribution of selected experts
    expert_counts = torch.zeros(num_experts)
    for i in range(topk_indices.shape[0]):
        for j in range(topk_indices.shape[1]):
            expert_counts[topk_indices[i, j]] += 1

    most_selected = expert_counts.topk(10)
    print(f"\nMost frequently selected experts (top 10):")
    for idx, count in zip(most_selected.indices, most_selected.values):
        print(f"  Expert {idx:3d}: {count:3.0f} times")

    # Show a few examples
    print(f"\nExample top-k values for first 3 tokens:")
    for i in range(min(3, seq_len)):
        vals = topk_values[i]
        indices = topk_indices[i]
        print(f"  Token {i}: experts {indices.tolist()}")
        print(f"    Values: {[f'{v:.4f}' for v in vals.tolist()]}")
        print(f"    Gaps: {[f'{(vals[j]-vals[j+1]):.4f}' for j in range(len(vals)-1)]}")

    if num_exact_ties == 0:
        print("\n✅ SUCCESS: No exact ties detected!")
        if num_close_ties < seq_len * 0.05:  # Less than 5% have close ties
            print("✅ Very few close ties - router weights should work well for deterministic topk.")
        else:
            print(f"⚠️  {num_close_ties} tokens have values within {tie_threshold} - may still cause issues.")
    else:
        print(f"\n⚠️  WARNING: {num_exact_ties} exact ties detected. Need to adjust parameters.")

    return weights


if __name__ == "__main__":
    test_no_ties()
