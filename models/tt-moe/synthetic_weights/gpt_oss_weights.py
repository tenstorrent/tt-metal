"""
Synthetic weight generator for GPT-OSS model.

This module generates synthetic weights with distributions matching
the real GPT-OSS-120B model weights.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


# Statistics derived from analyzing real GPT-OSS weights
# These values were obtained by analyzing the actual model at:
# /data/MLPerf/huggingface/hub/models--openai--gpt-oss-120b/
GPT_OSS_WEIGHT_STATS = {
    "router": {
        "weight_std": 0.00722,  # Standard deviation for router weights
        "bias_mean": 0.0,  # Router biases are typically zero
    },
    "expert_projections": {
        "gate_proj_std": 0.020,  # Standard deviation for gate projections
        "up_proj_std": 0.020,  # Standard deviation for up projections
        "down_proj_std": 0.020,  # Standard deviation for down projections
        # Note: Real GPT-OSS uses same std for all projections
    },
    "initialization": {
        "method": "normal",  # Normal distribution
        "mean": 0.0,  # Zero mean
    },
}


def generate_gpt_oss_synthetic_weights(
    layer_idx: int = 0,
    num_experts: int = 128,
    hidden_size: int = 2880,
    intermediate_size: int = 2880,
    dtype: torch.dtype = torch.bfloat16,
    stats_override: Optional[Dict] = None,
    seed: Optional[int] = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic weights for GPT-OSS MoE layer with realistic distributions.

    Args:
        layer_idx: Layer index for key naming
        num_experts: Number of experts in the MoE layer (default 128 for GPT-OSS)
        hidden_size: Hidden dimension size (default 2880 for GPT-OSS)
        intermediate_size: Intermediate/FFN dimension (default 2880 for GPT-OSS)
        dtype: Data type for weights (default bfloat16)
        stats_override: Optional dict to override default statistics
        seed: Random seed for reproducibility

    Returns:
        Dictionary of synthetic weight tensors with keys matching GPT-OSS format
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Use provided stats or defaults
    stats = stats_override if stats_override else GPT_OSS_WEIGHT_STATS

    state_dict = {}

    # Generate router weights
    router_std = stats["router"]["weight_std"]
    router_bias_mean = stats["router"]["bias_mean"]

    # Router weight matrix: [num_experts, hidden_size]
    state_dict[f"model.layers.{layer_idx}.mlp.router.weight"] = (
        torch.randn(num_experts, hidden_size, dtype=dtype) * router_std
    )

    # Router bias: typically zeros
    state_dict[f"model.layers.{layer_idx}.mlp.router.bias"] = torch.full((num_experts,), router_bias_mean, dtype=dtype)

    # Generate expert weights
    gate_std = stats["expert_projections"]["gate_proj_std"]
    up_std = stats["expert_projections"]["up_proj_std"]
    down_std = stats["expert_projections"]["down_proj_std"]

    for expert_idx in range(num_experts):
        # Gate projection: [intermediate_size, hidden_size]
        state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = (
            torch.randn(intermediate_size, hidden_size, dtype=dtype) * gate_std
        )

        # Up projection: [intermediate_size, hidden_size]
        state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = (
            torch.randn(intermediate_size, hidden_size, dtype=dtype) * up_std
        )

        # Down projection: [hidden_size, intermediate_size]
        state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = (
            torch.randn(hidden_size, intermediate_size, dtype=dtype) * down_std
        )

        # Note: GPT-OSS doesn't use biases in expert projections

    # Log statistics if logger is configured
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Generated GPT-OSS synthetic weights for layer {layer_idx}:")
        logger.info(f"  Router weight std: {router_std:.6f}")
        logger.info(f"  Num experts: {num_experts}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Intermediate size: {intermediate_size}")
        logger.info(f"  Expert projection stds: gate={gate_std:.4f}, up={up_std:.4f}, down={down_std:.4f}")

        # Sample verification
        sample_key = f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if sample_key in state_dict:
            sample_tensor = state_dict[sample_key]
            logger.info(f"  Sample tensor (expert 0 gate_proj):")
            logger.info(f"    Shape: {sample_tensor.shape}")
            logger.info(f"    Mean: {sample_tensor.mean():.6f}")
            logger.info(f"    Std: {sample_tensor.std():.6f}")
            logger.info(f"    Range: [{sample_tensor.min():.4f}, {sample_tensor.max():.4f}]")

    return state_dict


def load_weight_statistics(stats_file: Optional[Path] = None) -> Dict:
    """
    Load weight statistics from a JSON file.

    Args:
        stats_file: Path to JSON file with weight statistics.
                   If None, uses default GPT_OSS_WEIGHT_STATS.

    Returns:
        Dictionary of weight statistics
    """
    if stats_file and stats_file.exists():
        with open(stats_file, "r") as f:
            custom_stats = json.load(f)

        # Merge with defaults
        stats = GPT_OSS_WEIGHT_STATS.copy()

        # Update router stats if provided
        if "router" in custom_stats:
            if "mean_of_stds" in custom_stats["router"]:
                stats["router"]["weight_std"] = custom_stats["router"]["mean_of_stds"]

        # Update expert projection stats if provided
        for proj_type in ["gate_proj", "up_proj", "down_proj"]:
            if proj_type in custom_stats and "std" in custom_stats[proj_type]:
                stats["expert_projections"][f"{proj_type}_std"] = custom_stats[proj_type]["std"]

        return stats

    return GPT_OSS_WEIGHT_STATS


# Export the main function for use by MoE block
__all__ = ["generate_gpt_oss_synthetic_weights", "load_weight_statistics", "GPT_OSS_WEIGHT_STATS"]


if __name__ == "__main__":
    # Test the weight generator
    logging.basicConfig(level=logging.INFO)

    # Load custom stats if available
    stats_file = Path("/tmp/gpt_oss_weight_stats.json")
    if stats_file.exists():
        logger.info(f"Loading weight statistics from {stats_file}")
        stats = load_weight_statistics(stats_file)
    else:
        logger.info("Using default weight statistics")
        stats = GPT_OSS_WEIGHT_STATS

    # Generate test weights
    weights = generate_gpt_oss_synthetic_weights(
        layer_idx=0,
        num_experts=4,  # Small test
        stats_override=stats,
    )

    logger.info(f"\nGenerated {len(weights)} weight tensors")
    total_params = sum(w.numel() for w in weights.values())
    logger.info(f"Total parameters: {total_params:,}")

    # Save test weights
    torch.save(weights, "/tmp/gpt_oss_test_weights.pt")
    logger.info(f"Saved test weights to /tmp/gpt_oss_test_weights.pt")
