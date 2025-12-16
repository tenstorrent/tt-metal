"""
Comprehensive test utilities for comparing ttnn and PyTorch tensor outputs.
Provides detailed statistical analysis and comparison functions.
"""

import torch
import ttnn
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def convert_ttnn_to_torch(ttnn_tensor) -> torch.Tensor:
    """Convert ttnn tensor to PyTorch tensor for comparison."""
    if isinstance(ttnn_tensor, torch.Tensor):
        return ttnn_tensor
    return ttnn.to_torch(ttnn_tensor)


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
    ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)

    # Ensure tensors have same shape
    if torch_tensor.shape != ttnn_as_torch.shape:
        raise ValueError(f"Shape mismatch: torch={torch_tensor.shape}, ttnn={ttnn_as_torch.shape}")

    # Convert to float32 for precision in calculations
    torch_flat = torch_tensor.float().flatten()
    ttnn_flat = ttnn_as_torch.float().flatten()

    # Absolute error
    abs_error = torch.abs(torch_flat - ttnn_flat)

    # Relative error (avoid division by zero)
    torch_abs = torch.abs(torch_flat)
    rel_error = torch.where(
        torch_abs > 1e-12,  # Avoid division by very small numbers
        abs_error / torch_abs,
        abs_error,  # Use absolute error when reference is near zero
    )

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

    return stats


def print_detailed_comparison(
    torch_tensor: torch.Tensor, ttnn_tensor, tensor_name: str = "tensor", show_histograms: bool = False
) -> None:
    """
    Print detailed comparison statistics between torch and ttnn tensors.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        tensor_name: Name for logging
        show_histograms: Whether to show error distribution histograms
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
        ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)

        torch_flat = torch_tensor.float().flatten()
        ttnn_flat = ttnn_as_torch.float().flatten()
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
        ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)
        torch_flat = torch_tensor.float().flatten()
        ttnn_flat = ttnn_as_torch.float().flatten()
        abs_error = torch.abs(torch_flat - ttnn_flat)
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

        # Log results
        print(f"\n🧪 TOLERANCE CHECK RESULTS ({tensor_name}):")
        print(f"  Shapes match: {'✅' if shapes_match else '❌'}")
        print(f"  PCC ≥ {pcc_threshold}: {'✅' if pcc_pass else '❌'} (actual: {stats['pcc']:.6f})")
        print(
            f"  Abs error ≤ {abs_error_threshold}: {'✅' if abs_error_pass else '❌'} (actual: {stats['abs_error_mean']:.6f})"
        )
        print(
            f"  Rel error ≤ {rel_error_threshold}: {'✅' if rel_error_pass else '❌'} (actual: {stats['rel_error_mean']:.6f})"
        )
        print(
            f"  High error ratio ≤ {max_error_ratio}: {'✅' if error_ratio_pass else '❌'} (actual: {high_error_ratio:.6f})"
        )
        print(f"  OVERALL: {'✅ PASSED' if all_pass else '❌ FAILED'}")

        return all_pass, results

    except Exception as e:
        logger.error(f"Error in tolerance checking: {e}")
        results["error"] = str(e)
        return False, results


def save_comparison_report(
    torch_tensor: torch.Tensor, ttnn_tensor, tensor_name: str, output_path: str, **kwargs
) -> None:
    """
    Save detailed comparison report to file.

    Args:
        torch_tensor: Reference PyTorch tensor
        ttnn_tensor: ttnn tensor to compare
        tensor_name: Name for the tensor
        output_path: Path to save the report
        **kwargs: Additional parameters for tolerance checking
    """
    try:
        with open(output_path, "w") as f:
            # Redirect print output to file
            import sys

            original_stdout = sys.stdout
            sys.stdout = f

            print(f"COMPARISON REPORT: {tensor_name}")
            print(f"Generated at: {torch.utils.data.get_worker_info()}")  # Timestamp placeholder
            print("=" * 80)

            # Detailed comparison
            print_detailed_comparison(torch_tensor, ttnn_tensor, tensor_name, show_histograms=True)

            # Tolerance checking
            passed, results = check_with_tolerances(torch_tensor, ttnn_tensor, tensor_name=tensor_name, **kwargs)

            # Restore stdout
            sys.stdout = original_stdout

        logger.info(f"Comparison report saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error saving comparison report: {e}")


# Legacy compatibility functions
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
        ttnn_as_torch = convert_ttnn_to_torch(ttnn_tensor)

        if torch_tensor.shape != ttnn_as_torch.shape:
            return False, f"Shape mismatch: {torch_tensor.shape} vs {ttnn_as_torch.shape}"

        stats = compute_error_statistics(torch_tensor, ttnn_tensor)
        actual_pcc = stats["pcc"]

        passed = actual_pcc >= pcc
        message = f"PCC: {actual_pcc:.6f} (threshold: {pcc})"

        return passed, message

    except Exception as e:
        return False, f"Error computing PCC: {e}"


# Example usage function
def example_usage():
    """Example of how to use the comparison utilities."""
    # Create example tensors
    torch_ref = torch.randn(2, 256, 64)
    torch_test = torch_ref + 0.01 * torch.randn_like(torch_ref)  # Add small noise

    print("Example usage of comparison utilities:")

    # Basic detailed comparison
    print_detailed_comparison(torch_ref, torch_test, "example_tensor")

    # Tolerance checking
    passed, results = check_with_tolerances(
        torch_ref,
        torch_test,
        pcc_threshold=0.95,
        abs_error_threshold=0.02,
        rel_error_threshold=0.1,
        tensor_name="example_tensor",
    )

    # Legacy PCC check
    pcc_passed, pcc_msg = check_with_pcc(torch_ref, torch_test, pcc=0.95)
    print(f"\nLegacy PCC check: {pcc_passed}, {pcc_msg}")


if __name__ == "__main__":
    example_usage()
