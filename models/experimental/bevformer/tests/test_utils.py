# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test utilities for comparing ttnn and PyTorch tensor outputs.
Provides detailed statistical analysis and comparison functions with sparsity awareness.
"""

import torch
import ttnn
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Module-level constants to replace magic numbers
ZERO_THRESHOLD_DEFAULT = 1e-8
DIVISION_SAFETY_THRESHOLD = 1e-12
SPARSITY_DENSE_THRESHOLD = 0.1
SPARSITY_MODERATE_THRESHOLD = 0.5
SPARSITY_SPARSE_THRESHOLD = 0.9


def _categorize_sparsity(sparsity_ratio: float, include_icon: bool = True) -> Tuple[str, str]:
    """
    Central sparsity categorization logic.

    Args:
        sparsity_ratio: Sparsity ratio (0.0 to 1.0)
        include_icon: Whether to include emoji icons

    Returns:
        (category_name, icon) tuple
    """
    if sparsity_ratio < SPARSITY_DENSE_THRESHOLD:
        category = "Dense"
        icon = "🟦" if include_icon else ""
    elif sparsity_ratio < SPARSITY_MODERATE_THRESHOLD:
        category = "Moderately Sparse"
        icon = "🟨" if include_icon else ""
    elif sparsity_ratio < SPARSITY_SPARSE_THRESHOLD:
        category = "Sparse"
        icon = "🟧" if include_icon else ""
    else:
        category = "Extremely Sparse"
        icon = "🟥" if include_icon else ""

    return category, icon


def _preprocess_tensor_pair(
    torch_tensor: torch.Tensor, ttnn_tensor, tensor_name: str
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Standard tensor preprocessing: convert, validate shapes, flatten.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        tensor_name: Name for logging

    Returns:
        (torch_flat, ttnn_flat, shapes_match) tuple
    """
    # Convert ttnn tensor to torch
    ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)

    # Check shapes match
    torch_shape = torch_tensor.shape
    ttnn_shape = ttnn_as_torch.shape
    shapes_match = torch_shape == ttnn_shape

    if not shapes_match:
        logger.info(f"{tensor_name} shape comparison:")
        logger.info(f"  PyTorch shape: {torch_shape}")
        logger.info(f"  ttnn shape:    {ttnn_shape}")
        logger.info(f"  Match: {shapes_match}")

    # Flatten tensors for comparison
    torch_flat = torch_tensor.float().flatten()
    ttnn_flat = ttnn_as_torch.float().flatten()

    return torch_flat, ttnn_flat, shapes_match


def _compute_error_tensors(torch_flat: torch.Tensor, ttnn_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute absolute and relative error tensors with consistent logic.

    Args:
        torch_flat: Flattened reference tensor
        ttnn_flat: Flattened comparison tensor

    Returns:
        Dictionary with error tensors: {"abs_error", "rel_error", "torch_abs"}
    """
    # Absolute error
    abs_error = torch.abs(torch_flat - ttnn_flat)

    # Relative error (avoid division by zero)
    torch_abs = torch.abs(torch_flat)
    rel_error = torch.where(
        torch_abs > DIVISION_SAFETY_THRESHOLD,  # Avoid division by very small numbers
        abs_error / torch_abs,
        abs_error,  # Use absolute error when reference is near zero
    )

    return {"abs_error": abs_error, "rel_error": rel_error, "torch_abs": torch_abs}


def convert_ttnn_to_torch(ttnn_tensor) -> torch.Tensor:
    """Convert ttnn tensor to PyTorch tensor for comparison."""
    if isinstance(ttnn_tensor, torch.Tensor):
        return ttnn_tensor
    return ttnn.to_torch(ttnn_tensor)


def analyze_sparsity(
    tensor: torch.Tensor, zero_threshold: float = 1e-8, block_size: Optional[int] = None, tensor_name: str = "tensor"
) -> Dict[str, Any]:
    """
    Comprehensive sparsity analysis of a tensor.

    Args:
        tensor: Input tensor to analyze
        zero_threshold: Values below this threshold are considered zero
        block_size: Size for block sparsity analysis (e.g., 4 for 4x4 blocks)
        tensor_name: Name for logging

    Returns:
        Dictionary containing sparsity statistics
    """
    if tensor.numel() == 0:
        return {"error": "Empty tensor"}

    tensor_flat = tensor.float().flatten()
    total_elements = tensor_flat.numel()

    # Basic sparsity
    near_zero_mask = torch.abs(tensor_flat) <= zero_threshold
    num_zeros = near_zero_mask.sum().item()
    num_nonzeros = total_elements - num_zeros
    sparsity_ratio = num_zeros / total_elements

    # Value distribution analysis
    nonzero_values = tensor_flat[~near_zero_mask]

    stats = {
        "tensor_name": tensor_name,
        "shape": tuple(tensor.shape),
        "total_elements": total_elements,
        "zero_threshold": zero_threshold,
        # Basic sparsity metrics
        "num_zeros": num_zeros,
        "num_nonzeros": num_nonzeros,
        "sparsity_ratio": sparsity_ratio,
        "density_ratio": 1.0 - sparsity_ratio,
        # Nonzero value statistics
        "nonzero_min": nonzero_values.min().item() if num_nonzeros > 0 else 0.0,
        "nonzero_max": nonzero_values.max().item() if num_nonzeros > 0 else 0.0,
        "nonzero_mean": nonzero_values.mean().item() if num_nonzeros > 0 else 0.0,
        "nonzero_std": nonzero_values.std().item() if num_nonzeros > 0 else 0.0,
        "nonzero_median": nonzero_values.median().item() if num_nonzeros > 0 else 0.0,
    }

    # Effective rank analysis (for 2D tensors)
    if tensor.dim() == 2 and num_nonzeros > 0:
        try:
            # Compute SVD to get effective rank
            U, S, V = torch.svd(tensor)
            # Effective rank: number of singular values above threshold
            significant_sv = (S > zero_threshold).sum().item()
            total_sv = S.numel()

            # Spectral metrics
            spectral_norm = S.max().item()
            frobenius_norm = torch.norm(tensor, "fro").item()
            nuclear_norm = S.sum().item()

            # Rank-based sparsity
            rank_sparsity = 1.0 - (significant_sv / total_sv)

            stats.update(
                {
                    "effective_rank": significant_sv,
                    "max_rank": total_sv,
                    "rank_sparsity": rank_sparsity,
                    "spectral_norm": spectral_norm,
                    "frobenius_norm": frobenius_norm,
                    "nuclear_norm": nuclear_norm,
                    "condition_number": (S.max() / S[S > zero_threshold].min()).item() if significant_sv > 1 else 1.0,
                }
            )
        except Exception as e:
            logger.warning(f"SVD analysis failed for {tensor_name}: {e}")
            stats.update({"effective_rank": None, "svd_error": str(e)})

    # Block sparsity analysis
    if block_size is not None and tensor.dim() >= 2:
        try:
            h, w = tensor.shape[-2], tensor.shape[-1]
            if h >= int(block_size) and w >= int(block_size):
                # Reshape into blocks
                blocks_h = h // int(block_size)
                blocks_w = w // int(block_size)

                # Extract blocks
                tensor_2d = tensor.view(-1, h, w)  # Flatten batch dimensions
                blocks = tensor_2d[:, : blocks_h * int(block_size), : blocks_w * int(block_size)]
                blocks = blocks.view(tensor_2d.shape[0], blocks_h, int(block_size), blocks_w, int(block_size))
                blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()
                blocks = blocks.view(-1, int(block_size), int(block_size))

                # Analyze block sparsity
                block_norms = torch.norm(blocks.view(blocks.shape[0], -1), dim=1)
                zero_blocks = (block_norms <= zero_threshold).sum().item()
                total_blocks = blocks.shape[0]
                block_sparsity = zero_blocks / total_blocks

                stats.update(
                    {
                        "block_size": block_size,
                        "total_blocks": total_blocks,
                        "zero_blocks": zero_blocks,
                        "block_sparsity": block_sparsity,
                        "structured_sparsity_ratio": block_sparsity,
                    }
                )
        except Exception as e:
            logger.warning(f"Block sparsity analysis failed for {tensor_name}: {e}")
            stats.update({"block_sparsity_error": str(e)})

    # Pattern analysis for attention-like tensors
    if tensor.dim() >= 3:  # Likely attention tensor [batch, heads, seq, seq] or similar
        try:
            # Analyze sparsity across different dimensions
            batch_sparsity = []
            head_sparsity = []

            if tensor.dim() == 4:  # [batch, heads, seq, seq]
                for b in range(tensor.shape[0]):
                    batch_mask = torch.abs(tensor[b]) <= zero_threshold
                    batch_sparsity.append(batch_mask.float().mean().item())

                for h in range(tensor.shape[1]):
                    head_mask = torch.abs(tensor[:, h]) <= zero_threshold
                    head_sparsity.append(head_mask.float().mean().item())

                stats.update(
                    {
                        "batch_sparsity_mean": np.mean(batch_sparsity),
                        "batch_sparsity_std": np.std(batch_sparsity),
                        "head_sparsity_mean": np.mean(head_sparsity),
                        "head_sparsity_std": np.std(head_sparsity),
                        "attention_pattern_analysis": True,
                    }
                )
        except Exception as e:
            logger.warning(f"Attention pattern analysis failed for {tensor_name}: {e}")

    return stats


def analyze_attention_sparsity(
    attention_weights: torch.Tensor, zero_threshold: float = 1e-8, tensor_name: str = "attention_weights"
) -> Dict[str, Any]:
    """
    Specialized sparsity analysis for attention weight tensors.

    Args:
        attention_weights: Attention weight tensor [batch, heads, seq_len, seq_len]
        zero_threshold: Threshold for considering values as zero
        tensor_name: Name for logging

    Returns:
        Dictionary with attention-specific sparsity metrics
    """
    base_stats = analyze_sparsity(attention_weights, zero_threshold, tensor_name=tensor_name)

    if attention_weights.dim() < 3:
        return base_stats

    # Attention-specific analysis
    attention_stats = {}

    try:
        # Assume last two dimensions are [seq_len, seq_len]
        seq_len = attention_weights.shape[-1]

        # Diagonal attention analysis
        if attention_weights.shape[-2] == attention_weights.shape[-1]:
            # Extract diagonal elements
            diag_elements = torch.diagonal(attention_weights, dim1=-2, dim2=-1)
            diag_mask = torch.abs(diag_elements) <= zero_threshold
            diagonal_sparsity = diag_mask.float().mean().item()

            # Extract off-diagonal elements
            mask = ~torch.eye(seq_len, dtype=torch.bool, device=attention_weights.device)
            off_diag_elements = attention_weights[..., mask].view(*attention_weights.shape[:-2], -1)
            off_diag_mask = torch.abs(off_diag_elements) <= zero_threshold
            off_diagonal_sparsity = off_diag_mask.float().mean().item()

            attention_stats.update(
                {
                    "diagonal_sparsity": diagonal_sparsity,
                    "off_diagonal_sparsity": off_diagonal_sparsity,
                    "diagonal_dominance": diagonal_sparsity < off_diagonal_sparsity,
                }
            )

        # Local vs global attention analysis
        if seq_len > 8:  # Only for reasonably sized sequences
            # Local attention (within distance 2)
            local_mask = torch.zeros_like(attention_weights, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - 2)
                end = min(seq_len, i + 3)
                local_mask[..., i, start:end] = True

            local_elements = attention_weights[local_mask]
            local_sparsity_mask = torch.abs(local_elements) <= zero_threshold
            local_sparsity = local_sparsity_mask.float().mean().item()

            # Global attention (rest)
            global_elements = attention_weights[~local_mask]
            global_sparsity_mask = torch.abs(global_elements) <= zero_threshold
            global_sparsity = global_sparsity_mask.float().mean().item()

            attention_stats.update(
                {
                    "local_attention_sparsity": local_sparsity,
                    "global_attention_sparsity": global_sparsity,
                    "locality_preference": local_sparsity < global_sparsity,
                }
            )

        # Entropy-based attention pattern analysis
        if attention_weights.min() >= 0:  # Valid probability distribution
            # Compute entropy for each attention head
            eps = 1e-12
            log_weights = torch.log(attention_weights + eps)
            entropy = -(attention_weights * log_weights).sum(dim=-1)  # Entropy per query

            attention_stats.update(
                {
                    "attention_entropy_mean": entropy.mean().item(),
                    "attention_entropy_std": entropy.std().item(),
                    "attention_entropy_min": entropy.min().item(),
                    "attention_entropy_max": entropy.max().item(),
                }
            )

    except Exception as e:
        logger.warning(f"Attention-specific analysis failed: {e}")
        attention_stats["attention_analysis_error"] = str(e)

    # Merge with base stats
    base_stats.update(attention_stats)
    return base_stats


def compare_sparsity_patterns(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    zero_threshold: float = 1e-8,
    tensor_names: Tuple[str, str] = ("tensor1", "tensor2"),
) -> Dict[str, Any]:
    """
    Compare sparsity patterns between two tensors.

    Args:
        tensor1: First tensor (reference)
        tensor2: Second tensor (comparison)
        zero_threshold: Threshold for zero detection
        tensor_names: Names for the tensors

    Returns:
        Dictionary with sparsity comparison metrics
    """
    if tensor1.shape != tensor2.shape:
        return {"error": f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"}

    # Individual sparsity analysis
    stats1 = analyze_sparsity(tensor1, zero_threshold, tensor_name=tensor_names[0])
    stats2 = analyze_sparsity(tensor2, zero_threshold, tensor_name=tensor_names[1])

    # Pattern comparison
    flat1 = tensor1.float().flatten()
    flat2 = tensor2.float().flatten()

    mask1 = torch.abs(flat1) <= zero_threshold
    mask2 = torch.abs(flat2) <= zero_threshold

    # Sparsity pattern overlap
    both_zero = (mask1 & mask2).sum().item()
    either_zero = (mask1 | mask2).sum().item()
    only_tensor1_zero = (mask1 & ~mask2).sum().item()
    only_tensor2_zero = (~mask1 & mask2).sum().item()

    pattern_overlap = both_zero / either_zero if either_zero > 0 else 1.0

    # Jaccard similarity of sparsity patterns (intersection over union)
    jaccard_similarity = both_zero / either_zero if either_zero > 0 else 1.0

    # Alternative similarity: intersection over smaller set
    smaller_zero_count = min(mask1.sum().item(), mask2.sum().item())
    intersection_similarity = both_zero / smaller_zero_count if smaller_zero_count > 0 else 1.0

    # Sparsity difference
    sparsity_diff = abs(stats1["sparsity_ratio"] - stats2["sparsity_ratio"])

    comparison_stats = {
        "tensor_names": tensor_names,
        "individual_stats": {tensor_names[0]: stats1, tensor_names[1]: stats2},
        # Pattern comparison
        "pattern_overlap": pattern_overlap,
        "jaccard_similarity": jaccard_similarity,
        "intersection_similarity": intersection_similarity,
        "sparsity_difference": sparsity_diff,
        "both_zero_elements": both_zero,
        "either_zero_elements": either_zero,
        "only_tensor1_zero": only_tensor1_zero,
        "only_tensor2_zero": only_tensor2_zero,
        "total_elements": flat1.numel(),
        # Relative sparsity metrics
        "sparsity_ratio_1": stats1["sparsity_ratio"],
        "sparsity_ratio_2": stats2["sparsity_ratio"],
        "sparsity_correlation": "higher" if stats1["sparsity_ratio"] < stats2["sparsity_ratio"] else "lower",
    }

    return comparison_stats


def print_sparsity_analysis(
    tensor: torch.Tensor,
    zero_threshold: float = 1e-8,
    block_size: Optional[int] = None,
    tensor_name: str = "tensor",
    show_detailed: bool = True,
) -> None:
    """
    Print comprehensive sparsity analysis.

    Args:
        tensor: Tensor to analyze
        zero_threshold: Zero threshold
        block_size: Block size for structured sparsity analysis
        tensor_name: Name for display
        show_detailed: Whether to show detailed statistics
    """
    print(f"\n🕳️  SPARSITY ANALYSIS: {tensor_name}")
    print("=" * 60)

    stats = analyze_sparsity(tensor, zero_threshold, block_size, tensor_name)

    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return

    # Basic sparsity info
    print(f"📊 BASIC SPARSITY:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Total elements: {stats['total_elements']:,}")
    print(f"  Zero threshold: {stats['zero_threshold']:.2e}")
    print(f"  Zero elements: {stats['num_zeros']:,} ({stats['sparsity_ratio']:.1%})")
    print(f"  Non-zero elements: {stats['num_nonzeros']:,} ({stats['density_ratio']:.1%})")

    # Sparsity categorization
    sparsity_level = stats["sparsity_ratio"]
    sparsity_category, icon = _categorize_sparsity(sparsity_level)

    print(f"  Category: {icon} {sparsity_category}")

    # Non-zero value statistics
    if stats["num_nonzeros"] > 0:
        print(f"\n📈 NON-ZERO VALUE STATISTICS:")
        print(f"  Range: [{stats['nonzero_min']:.6f}, {stats['nonzero_max']:.6f}]")
        print(f"  Mean: {stats['nonzero_mean']:.6f}")
        print(f"  Median: {stats['nonzero_median']:.6f}")
        print(f"  Std: {stats['nonzero_std']:.6f}")

    # Rank analysis for 2D tensors
    if "effective_rank" in stats and stats["effective_rank"] is not None:
        print(f"\n🔢 RANK ANALYSIS (2D tensors):")
        print(f"  Effective rank: {stats['effective_rank']}/{stats['max_rank']}")
        print(f"  Rank utilization: {(stats['effective_rank']/stats['max_rank']):.1%}")
        print(f"  Condition number: {stats['condition_number']:.2f}")
        if show_detailed:
            print(f"  Spectral norm: {stats['spectral_norm']:.6f}")
            print(f"  Frobenius norm: {stats['frobenius_norm']:.6f}")
            print(f"  Nuclear norm: {stats['nuclear_norm']:.6f}")

    # Block sparsity
    if "block_sparsity" in stats:
        print(f"\n🧱 BLOCK SPARSITY ({stats['block_size']}x{stats['block_size']} blocks):")
        print(f"  Total blocks: {stats['total_blocks']:,}")
        print(f"  Zero blocks: {stats['zero_blocks']:,} ({stats['block_sparsity']:.1%})")
        print(f"  Structured sparsity: {stats['structured_sparsity_ratio']:.1%}")

    # Attention pattern analysis
    if "attention_pattern_analysis" in stats:
        print(f"\n🎯 ATTENTION PATTERN ANALYSIS:")
        print(f"  Batch sparsity: {stats['batch_sparsity_mean']:.1%} ± {stats['batch_sparsity_std']:.1%}")
        print(f"  Head sparsity: {stats['head_sparsity_mean']:.1%} ± {stats['head_sparsity_std']:.1%}")

    # Attention-specific metrics
    if "diagonal_sparsity" in stats:
        print(f"\n🔍 ATTENTION-SPECIFIC METRICS:")
        print(f"  Diagonal sparsity: {stats['diagonal_sparsity']:.1%}")
        print(f"  Off-diagonal sparsity: {stats['off_diagonal_sparsity']:.1%}")
        print(f"  Diagonal dominance: {'✅' if stats['diagonal_dominance'] else '❌'}")

        if "local_attention_sparsity" in stats:
            print(f"  Local attention sparsity: {stats['local_attention_sparsity']:.1%}")
            print(f"  Global attention sparsity: {stats['global_attention_sparsity']:.1%}")
            print(f"  Locality preference: {'✅' if stats['locality_preference'] else '❌'}")

        if "attention_entropy_mean" in stats:
            print(f"  Attention entropy: {stats['attention_entropy_mean']:.3f} ± {stats['attention_entropy_std']:.3f}")

    print("=" * 60)


def compare_tensor_shapes(torch_tensor: torch.Tensor, ttnn_tensor, tensor_name: str = "tensor") -> bool:
    """
    Compare shapes of torch and ttnn tensors.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor or converted torch tensor
        tensor_name: Name for logging

    Returns:
        True if shapes match, False otherwise
    """
    ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)

    torch_shape = torch_tensor.shape
    ttnn_shape = ttnn_as_torch.shape

    shapes_match = torch_shape == ttnn_shape

    logger.info(f"{tensor_name} shape comparison:")
    logger.info(f"  PyTorch shape: {torch_shape}")
    logger.info(f"  ttnn shape:    {ttnn_shape}")
    logger.info(f"  Match: {shapes_match}")

    return shapes_match


def compute_error_statistics(torch_tensor: torch.Tensor, ttnn_tensor, tensor_name: str = "tensor") -> Dict[str, float]:
    """
    Compute comprehensive error statistics between torch and ttnn tensors.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        tensor_name: Name for logging

    Returns:
        Dictionary containing error statistics
    """
    # Convert ttnn tensor (needed for stats later)
    ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)

    # Preprocess tensors (validate shapes, flatten)
    torch_flat, ttnn_flat, shapes_match = _preprocess_tensor_pair(torch_tensor, ttnn_tensor, tensor_name)

    # Ensure tensors have same shape (preserve original exception behavior)
    if not shapes_match:
        raise ValueError(f"Shape mismatch: torch={torch_tensor.shape}, ttnn={ttnn_as_torch.shape}")

    # Compute error tensors
    error_data = _compute_error_tensors(torch_flat, ttnn_flat)
    abs_error = error_data["abs_error"]
    rel_error = error_data["rel_error"]

    # Pearson Correlation Coefficient
    pcc = torch.corrcoef(torch.stack([torch_flat, ttnn_flat]))[0, 1].item()
    if torch.isnan(torch.tensor(pcc)):
        pcc = 0.0  # Handle case where one tensor is constant

    # Compute statistics
    stats = {
        # Basic tensor info
        "num_elements": torch_flat.numel(),
        "torch_dtype": str(torch_tensor.dtype),
        "ttnn_dtype": str(ttnn_as_torch.dtype),
        # Absolute error statistics
        "abs_error_min": abs_error.min().item(),
        "abs_error_max": abs_error.max().item(),
        "abs_error_mean": abs_error.mean().item(),
        "abs_error_std": abs_error.std().item(),
        "abs_error_median": abs_error.median().item(),
        # Relative error statistics
        "rel_error_min": rel_error.min().item(),
        "rel_error_max": rel_error.max().item(),
        "rel_error_mean": rel_error.mean().item(),
        "rel_error_std": rel_error.std().item(),
        "rel_error_median": rel_error.median().item(),
        # Correlation
        "pcc": pcc,
        # Value range statistics
        "torch_min": torch_flat.min().item(),
        "torch_max": torch_flat.max().item(),
        "torch_mean": torch_flat.mean().item(),
        "torch_std": torch_flat.std().item(),
        "ttnn_min": ttnn_flat.min().item(),
        "ttnn_max": ttnn_flat.max().item(),
        "ttnn_mean": ttnn_flat.mean().item(),
        "ttnn_std": ttnn_flat.std().item(),
        # Percentile-based error analysis
        "abs_error_p95": torch.quantile(abs_error, 0.95).item(),
        "abs_error_p99": torch.quantile(abs_error, 0.99).item(),
        "rel_error_p95": torch.quantile(rel_error, 0.95).item(),
        "rel_error_p99": torch.quantile(rel_error, 0.99).item(),
    }

    # Add sparsity analysis
    try:
        torch_sparsity = analyze_sparsity(torch_tensor, tensor_name=f"{tensor_name}_torch")
        ttnn_sparsity = analyze_sparsity(ttnn_as_torch, tensor_name=f"{tensor_name}_ttnn")
        sparsity_comparison = compare_sparsity_patterns(
            torch_tensor, ttnn_as_torch, tensor_names=(f"{tensor_name}_torch", f"{tensor_name}_ttnn")
        )

        # Add key sparsity metrics to main stats
        stats.update(
            {
                "torch_sparsity_ratio": torch_sparsity["sparsity_ratio"],
                "ttnn_sparsity_ratio": ttnn_sparsity["sparsity_ratio"],
                "sparsity_difference": sparsity_comparison["sparsity_difference"],
                "sparsity_pattern_similarity": sparsity_comparison["jaccard_similarity"],
                "torch_density_ratio": torch_sparsity["density_ratio"],
                "ttnn_density_ratio": ttnn_sparsity["density_ratio"],
            }
        )

        # Store detailed sparsity stats separately
        stats["detailed_sparsity"] = {
            "torch_sparsity": torch_sparsity,
            "ttnn_sparsity": ttnn_sparsity,
            "comparison": sparsity_comparison,
        }

    except Exception as e:
        logger.warning(f"Sparsity analysis failed for {tensor_name}: {e}")
        stats["sparsity_analysis_error"] = str(e)

    return stats


def print_detailed_comparison(
    torch_tensor: torch.Tensor,
    ttnn_tensor,
    tensor_name: str = "tensor",
    show_histograms: bool = False,
    show_sparsity: bool = True,
) -> None:
    """
    Print detailed comparison statistics between torch and ttnn tensors.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        tensor_name: Name for logging
        show_histograms: Whether to show error distribution histograms
        show_sparsity: Whether to show sparsity analysis
    """
    print(f"\n{'='*60}")
    print(f"DETAILED COMPARISON: {tensor_name}")
    print(f"{'='*60}")

    # Shape comparison
    if not compare_tensor_shapes(torch_tensor, ttnn_tensor, tensor_name):
        print("❌ Shape mismatch - cannot proceed with detailed comparison")
        return

    # Error statistics
    try:
        stats = compute_error_statistics(torch_tensor, ttnn_tensor, tensor_name)

        print(f"\n📊 TENSOR INFO:")
        print(f"  Shape: {torch_tensor.shape}")
        print(f"  Elements: {stats['num_elements']:,}")
        print(f"  PyTorch dtype: {stats['torch_dtype']}")
        print(f"  ttnn dtype: {stats['ttnn_dtype']}")

        print(f"\n📈 VALUE RANGES:")
        print(
            f"  PyTorch: [{stats['torch_min']:.6f}, {stats['torch_max']:.6f}] (mean: {stats['torch_mean']:.6f}, std: {stats['torch_std']:.6f})"
        )
        print(
            f"  ttnn:    [{stats['ttnn_min']:.6f}, {stats['ttnn_max']:.6f}] (mean: {stats['ttnn_mean']:.6f}, std: {stats['ttnn_std']:.6f})"
        )

        # Sparsity summary
        if show_sparsity and "torch_sparsity_ratio" in stats:
            print(f"\n🕳️  SPARSITY SUMMARY:")
            print(f"  PyTorch sparsity: {stats['torch_sparsity_ratio']:.1%}")
            print(f"  ttnn sparsity:    {stats['ttnn_sparsity_ratio']:.1%}")
            print(f"  Sparsity difference: {stats['sparsity_difference']:.1%}")
            print(f"  Pattern similarity: {stats['sparsity_pattern_similarity']:.3f}")

            # Categorize sparsity level
            avg_sparsity = (stats["torch_sparsity_ratio"] + stats["ttnn_sparsity_ratio"]) / 2
            category_name, icon = _categorize_sparsity(avg_sparsity)
            sparsity_category = f"{icon} {category_name}"
            print(f"  Overall category: {sparsity_category}")

        print(f"\n🎯 CORRELATION:")
        print(f"  Pearson CC: {stats['pcc']:.8f}")

        print(f"\n📏 ABSOLUTE ERROR:")
        print(f"  Min:    {stats['abs_error_min']:.2e}")
        print(f"  Max:    {stats['abs_error_max']:.2e}")
        print(f"  Mean:   {stats['abs_error_mean']:.2e}")
        print(f"  Median: {stats['abs_error_median']:.2e}")
        print(f"  Std:    {stats['abs_error_std']:.2e}")
        print(f"  95th percentile: {stats['abs_error_p95']:.2e}")
        print(f"  99th percentile: {stats['abs_error_p99']:.2e}")

        print(f"\n📊 RELATIVE ERROR (as fraction, e.g., 0.01 = 1%):")
        print(f"  Min:    {stats['rel_error_min']:.2e} ({stats['rel_error_min']*100:.2e}%)")
        print(f"  Max:    {stats['rel_error_max']:.2e} ({stats['rel_error_max']*100:.2e}%)")
        print(f"  Mean:   {stats['rel_error_mean']:.2e} ({stats['rel_error_mean']*100:.2e}%)")
        print(f"  Median: {stats['rel_error_median']:.2e} ({stats['rel_error_median']*100:.2e}%)")
        print(f"  Std:    {stats['rel_error_std']:.2e}")
        print(f"  95th percentile: {stats['rel_error_p95']:.2e} ({stats['rel_error_p95']*100:.2e}%)")
        print(f"  99th percentile: {stats['rel_error_p99']:.2e} ({stats['rel_error_p99']*100:.2e}%)")

        if show_histograms:
            print_error_histograms(torch_tensor, ttnn_tensor, tensor_name)

        # Detailed sparsity analysis if available and requested
        if show_sparsity and "detailed_sparsity" in stats:
            detailed_sparsity = stats["detailed_sparsity"]

            # Quick sparsity comparison insights
            print(f"\n🔍 SPARSITY INSIGHTS:")

            # Check if sparsity patterns are preserved
            similarity = detailed_sparsity["comparison"]["jaccard_similarity"]
            if similarity > 0.95:
                pattern_status = "✅ Excellent pattern preservation"
            elif similarity > 0.8:
                pattern_status = "✅ Good pattern preservation"
            elif similarity > 0.5:
                pattern_status = "⚠️ Moderate pattern preservation"
            else:
                pattern_status = "❌ Poor pattern preservation"
            print(f"  {pattern_status} ({similarity:.3f})")

            # Check for rank preservation in 2D tensors
            torch_sparsity = detailed_sparsity["torch_sparsity"]
            ttnn_sparsity = detailed_sparsity["ttnn_sparsity"]

            if "effective_rank" in torch_sparsity and torch_sparsity["effective_rank"] is not None:
                torch_rank = torch_sparsity["effective_rank"]
                if "effective_rank" in ttnn_sparsity and ttnn_sparsity["effective_rank"] is not None:
                    ttnn_rank = ttnn_sparsity["effective_rank"]
                    rank_diff = abs(torch_rank - ttnn_rank)
                    if rank_diff <= 1:
                        rank_status = "✅ Rank preserved"
                    elif rank_diff <= 3:
                        rank_status = "⚠️ Rank slightly changed"
                    else:
                        rank_status = "❌ Rank significantly changed"
                    print(f"  {rank_status} (torch: {torch_rank}, ttnn: {ttnn_rank})")

    except Exception as e:
        print(f"❌ Error computing statistics: {e}")

    print(f"{'='*60}")


def print_error_histograms(
    torch_tensor: torch.Tensor, ttnn_tensor, tensor_name: str = "tensor", bins: int = 10
) -> None:
    """
    Print simple ASCII histograms of error distributions.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        tensor_name: Name for display
        bins: Number of histogram bins
    """
    try:
        torch_flat, ttnn_flat, shapes_match = _preprocess_tensor_pair(torch_tensor, ttnn_tensor, tensor_name)

        if not shapes_match:
            print(f"❌ Error creating histogram: Shape mismatch")
            return

        abs_error = torch.abs(torch_flat - ttnn_flat).numpy()

        # Create histogram
        counts, bin_edges = np.histogram(abs_error, bins=bins)

        print(f"\n📊 ABSOLUTE ERROR HISTOGRAM ({tensor_name}):")
        max_count = max(counts) if max(counts) > 0 else 1

        for i in range(len(counts)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            count = counts[i]
            percentage = (count / len(abs_error)) * 100

            # Create bar representation
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length

            print(f"  [{bin_start:.2e}, {bin_end:.2e}): {bar} {count:>6} ({percentage:5.1f}%)")

    except Exception as e:
        print(f"❌ Error creating histogram: {e}")


def check_with_tolerances(
    torch_tensor: torch.Tensor,
    ttnn_tensor,
    pcc_threshold: float = 0.99,
    abs_error_threshold: float = 1e-2,
    rel_error_threshold: float = 0.05,
    max_error_ratio: float = 0.01,
    tensor_name: str = "tensor",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive tolerance-based checking with multiple criteria.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        pcc_threshold: Minimum acceptable Pearson correlation coefficient
        abs_error_threshold: Maximum acceptable absolute error (mean)
        rel_error_threshold: Maximum acceptable relative error (mean)
        max_error_ratio: Maximum acceptable ratio of elements with high error
        tensor_name: Name for logging

    Returns:
        (passed, results_dict) - passed is True if all criteria met
    """
    results = {
        "tensor_name": tensor_name,
        "pcc_threshold": pcc_threshold,
        "abs_error_threshold": abs_error_threshold,
        "rel_error_threshold": rel_error_threshold,
        "max_error_ratio": max_error_ratio,
        "passed": False,
        "individual_checks": {},
        "stats": {},
    }

    try:
        # Shape check
        shapes_match = compare_tensor_shapes(torch_tensor, ttnn_tensor, tensor_name)
        results["individual_checks"]["shapes_match"] = shapes_match

        if not shapes_match:
            return False, results

        # Compute statistics
        stats = compute_error_statistics(torch_tensor, ttnn_tensor, tensor_name)
        results["stats"] = stats

        # Individual tolerance checks
        pcc_pass = stats["pcc"] >= pcc_threshold
        abs_error_pass = stats["abs_error_mean"] <= abs_error_threshold
        rel_error_pass = stats["rel_error_mean"] <= rel_error_threshold

        # High error element ratio check
        torch_flat, ttnn_flat, _ = _preprocess_tensor_pair(torch_tensor, ttnn_tensor, tensor_name)
        error_data = _compute_error_tensors(torch_flat, ttnn_flat)
        abs_error = error_data["abs_error"]
        high_error_elements = (abs_error > abs_error_threshold).sum().item()
        high_error_ratio = high_error_elements / torch_flat.numel()
        error_ratio_pass = high_error_ratio <= max_error_ratio

        results["individual_checks"].update(
            {
                "pcc_pass": pcc_pass,
                "abs_error_pass": abs_error_pass,
                "rel_error_pass": rel_error_pass,
                "error_ratio_pass": error_ratio_pass,
                "high_error_ratio": high_error_ratio,
            }
        )

        # Overall pass/fail
        all_pass = all([shapes_match, pcc_pass, abs_error_pass, rel_error_pass, error_ratio_pass])

        results["passed"] = all_pass

        # Helper function to get status indicator with warnings for overly conservative thresholds
        def get_status(passed, actual, threshold, higher_is_better=True, warn_margin=0.05):
            """Get status indicator with warning for overly conservative thresholds."""
            if not passed:
                return "❌"

            if higher_is_better:
                # For PCC: warn if actual is significantly higher than threshold
                if actual > threshold * (1 + warn_margin):
                    return "⚠️"  # Warning: threshold might be too low
            else:
                # For errors: warn if actual is significantly lower than threshold
                if actual < threshold * (1 - warn_margin):
                    return "⚠️"  # Warning: threshold might be too high

            return "✅"

        # Log results with warning indicators
        print(f"\n🧪 TOLERANCE CHECK RESULTS ({tensor_name}):")
        print(f"  Shapes match: {'✅' if shapes_match else '❌'}")

        pcc_indicator = get_status(pcc_pass, stats["pcc"], pcc_threshold, higher_is_better=True)
        print(f"  PCC ≥ {pcc_threshold}: {pcc_indicator} (actual: {stats['pcc']:.6f})")

        abs_error_indicator = get_status(
            abs_error_pass, stats["abs_error_mean"], abs_error_threshold, higher_is_better=False
        )
        print(f"  Abs error ≤ {abs_error_threshold}: {abs_error_indicator} (actual: {stats['abs_error_mean']:.6f})")

        rel_error_indicator = get_status(
            rel_error_pass, stats["rel_error_mean"], rel_error_threshold, higher_is_better=False
        )
        print(f"  Rel error ≤ {rel_error_threshold}: {rel_error_indicator} (actual: {stats['rel_error_mean']:.6f})")

        error_ratio_indicator = get_status(error_ratio_pass, high_error_ratio, max_error_ratio, higher_is_better=False)
        print(f"  High error ratio ≤ {max_error_ratio}: {error_ratio_indicator} (actual: {high_error_ratio:.6f})")

        print(f"  OVERALL: {'✅ PASSED' if all_pass else '❌ FAILED'}")

        # Add explanation for warnings
        warnings_shown = any(
            [
                pcc_indicator == "⚠️",
                abs_error_indicator == "⚠️",
                rel_error_indicator == "⚠️",
                error_ratio_indicator == "⚠️",
            ]
        )
        if warnings_shown:
            print(f"  ⚠️  = Warning: Actual performance significantly better than threshold (consider tightening)")

        return all_pass, results

    except Exception as e:
        logger.error(f"Error in tolerance checking: {e}")
        results["error"] = str(e)
        return False, results


def check_with_pcc(torch_tensor: torch.Tensor, ttnn_tensor, pcc: float = 0.99) -> Tuple[bool, str]:
    """
    Legacy PCC checking function for backward compatibility.

    Args:
        torch_tensor: Reference tensor
        ttnn_tensor: Tensor to compare
        pcc: PCC threshold

    Returns:
        (passed, message)
    """
    try:
        torch_flat, ttnn_flat, shapes_match = _preprocess_tensor_pair(torch_tensor, ttnn_tensor, "pcc_check")

        if not shapes_match:
            ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)  # For error message
            return False, f"Shape mismatch: {torch_tensor.shape} vs {ttnn_as_torch.shape}"

        stats = compute_error_statistics(torch_tensor, ttnn_tensor)
        actual_pcc = stats["pcc"]

        passed = actual_pcc >= pcc
        message = f"PCC: {actual_pcc:.6f} (threshold: {pcc})"

        return passed, message

    except Exception as e:
        return False, f"Error computing PCC: {e}"


def print_sparsity_summary(tensor: torch.Tensor, tensor_name: str = "tensor", zero_threshold: float = 1e-8) -> None:
    """
    Print a concise sparsity summary for a single tensor.

    Args:
        tensor: Tensor to analyze
        tensor_name: Name for display
        zero_threshold: Threshold for zero detection
    """
    stats = analyze_sparsity(tensor, zero_threshold, tensor_name=tensor_name)

    if "error" in stats:
        print(f"❌ Sparsity analysis error for {tensor_name}: {stats['error']}")
        return

    # Categorize sparsity
    sparsity_ratio = stats["sparsity_ratio"]
    category_name, icon = _categorize_sparsity(sparsity_ratio)
    category = f"{icon} {category_name}"

    print(
        f"🕳️  {tensor_name}: {category} ({sparsity_ratio:.1%} sparse, {stats['num_zeros']:,}/{stats['total_elements']:,} zeros)"
    )

    # Add rank info if available
    if "effective_rank" in stats and stats["effective_rank"] is not None:
        print(
            f"   Rank: {stats['effective_rank']}/{stats['max_rank']} (utilization: {(stats['effective_rank']/stats['max_rank']):.1%})"
        )

    # Add attention-specific info if available
    if "diagonal_sparsity" in stats:
        print(
            f"   Attention patterns: diag {stats['diagonal_sparsity']:.1%}, off-diag {stats['off_diagonal_sparsity']:.1%}"
        )


def analyze_model_sparsity(model_outputs: Dict[str, torch.Tensor], zero_threshold: float = 1e-8) -> Dict[str, Any]:
    """
    Analyze sparsity patterns across multiple model outputs (e.g., attention weights, features).

    Args:
        model_outputs: Dictionary of tensor_name -> tensor
        zero_threshold: Threshold for zero detection

    Returns:
        Dictionary with aggregated sparsity statistics
    """
    results = {
        "zero_threshold": zero_threshold,
        "tensor_count": len(model_outputs),
        "individual_stats": {},
        "aggregate_stats": {},
    }

    sparsity_ratios = []
    total_elements = 0
    total_zeros = 0

    print(f"\n🔍 MODEL SPARSITY ANALYSIS")
    print("=" * 50)

    for tensor_name, tensor in model_outputs.items():
        stats = analyze_sparsity(tensor, zero_threshold, tensor_name)
        results["individual_stats"][tensor_name] = stats

        if "error" not in stats:
            sparsity_ratios.append(stats["sparsity_ratio"])
            total_elements += stats["total_elements"]
            total_zeros += stats["num_zeros"]

            # Print concise summary
            print_sparsity_summary(tensor, tensor_name, zero_threshold)

    # Aggregate statistics
    if sparsity_ratios:
        results["aggregate_stats"] = {
            "mean_sparsity": np.mean(sparsity_ratios),
            "std_sparsity": np.std(sparsity_ratios),
            "min_sparsity": np.min(sparsity_ratios),
            "max_sparsity": np.max(sparsity_ratios),
            "median_sparsity": np.median(sparsity_ratios),
            "overall_sparsity": total_zeros / total_elements,
            "total_elements": total_elements,
            "total_zeros": total_zeros,
        }

        print(f"\n📊 AGGREGATE STATISTICS:")
        print(f"  Overall sparsity: {results['aggregate_stats']['overall_sparsity']:.1%}")
        print(
            f"  Mean tensor sparsity: {results['aggregate_stats']['mean_sparsity']:.1%} ± {results['aggregate_stats']['std_sparsity']:.1%}"
        )
        print(
            f"  Range: [{results['aggregate_stats']['min_sparsity']:.1%}, {results['aggregate_stats']['max_sparsity']:.1%}]"
        )
        print(f"  Total elements analyzed: {total_elements:,}")

    print("=" * 50)
    return results
