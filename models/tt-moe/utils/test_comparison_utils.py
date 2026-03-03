# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test utilities for TT-MoE component testing.

This module provides common utilities for comparing TTNN and PyTorch implementations,
including PCC calculation, weight extraction, and debug output formatting.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import comp_pcc


class TestComparisonConfig:
    """Configuration for test comparison thresholds and settings."""

    # Default PCC thresholds for different components
    ROUTER_WEIGHTS_PCC = 0.97
    ROUTER_INDICES_MISMATCH_TOLERANCE = 0.30
    EXPERTS_OUTPUT_PCC = 0.95
    INTEGRATION_PCC = 0.99
    INTERMEDIATE_PCC = 0.98

    # Memory configurations
    DEFAULT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG

    # Debug settings
    VERBOSE_DEBUG = False
    SAVE_TENSORS = False
    TENSOR_SAVE_PATH = "/tmp/tt_moe_debug_tensors"


def compare_tensors_pcc(
    reference: torch.Tensor,
    ttnn_output: torch.Tensor,
    threshold: float,
    component_name: str = "Component",
) -> Tuple[bool, str, float]:
    """Compare two tensors using Pearson Correlation Coefficient.

    Args:
        reference: Reference PyTorch tensor
        ttnn_output: TTNN output tensor
        threshold: PCC threshold for passing
        component_name: Name of component being tested

    Returns:
        Tuple of (passing, message, actual_pcc)
    """
    passing, pcc_message = comp_pcc(reference, ttnn_output, threshold)

    # Extract actual PCC value from message
    try:
        actual_pcc = float(pcc_message.split("=")[-1].strip())
    except:
        actual_pcc = 0.0

    detailed_message = f"{component_name} PCC: {pcc_message}"

    if not passing:
        # Add debug information
        detailed_message += (
            f"\n  Reference stats: min={reference.min():.6f}, max={reference.max():.6f}, mean={reference.mean():.6f}"
        )
        detailed_message += (
            f"\n  TTNN stats: min={ttnn_output.min():.6f}, max={ttnn_output.max():.6f}, mean={ttnn_output.mean():.6f}"
        )

        # Check for NaN or Inf
        if torch.isnan(reference).any():
            detailed_message += "\n  WARNING: Reference contains NaN values"
        if torch.isnan(ttnn_output).any():
            detailed_message += "\n  WARNING: TTNN output contains NaN values"
        if torch.isinf(reference).any():
            detailed_message += "\n  WARNING: Reference contains Inf values"
        if torch.isinf(ttnn_output).any():
            detailed_message += "\n  WARNING: TTNN output contains Inf values"

    return passing, detailed_message, actual_pcc


def compare_indices(
    reference: torch.Tensor,
    ttnn_output: torch.Tensor,
    mismatch_tolerance: float = 0.30,
) -> Tuple[bool, str, float]:
    """Compare index tensors allowing for tie-breaking differences.

    Args:
        reference: Reference index tensor
        ttnn_output: TTNN index tensor
        mismatch_tolerance: Maximum allowed mismatch ratio

    Returns:
        Tuple of (passing, message, mismatch_ratio)
    """
    # Sort indices for comparison (handles tie-breaking differences)
    ref_sorted = torch.sort(reference.to(torch.int32), dim=-1, stable=True)[0]
    ttnn_sorted = torch.sort(ttnn_output.to(torch.int32), dim=-1, stable=True)[0]

    indices_match = torch.equal(ref_sorted, ttnn_sorted)

    if indices_match:
        return True, "Indices match perfectly after sorting", 0.0

    # Calculate mismatch ratio
    num_mismatches = (ref_sorted != ttnn_sorted).sum().item()
    total_indices = ref_sorted.numel()
    mismatch_ratio = num_mismatches / total_indices if total_indices > 0 else 1.0

    message = f"Indices mismatch: {num_mismatches}/{total_indices} ({mismatch_ratio:.1%})"

    if mismatch_ratio <= mismatch_tolerance:
        message += f" - within tolerance ({mismatch_tolerance:.0%})"
        passing = True
    else:
        message += f" - exceeds tolerance ({mismatch_tolerance:.0%})"
        passing = False

    return passing, message, mismatch_ratio


def extract_weights_for_component(
    reference_model: torch.nn.Module,
    component_type: str,
    num_experts: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Extract weights from reference model for specific component.

    Args:
        reference_model: Reference PyTorch model
        component_type: Type of component ("router", "experts")
        num_experts: Number of experts (for expert extraction)

    Returns:
        Dictionary of extracted weights
    """
    state_dict = {}

    if component_type == "router":
        # Extract router/gate weights
        if hasattr(reference_model, "weight"):
            state_dict["weight"] = reference_model.weight
        if hasattr(reference_model, "e_score_correction_bias"):
            state_dict["e_score_correction_bias"] = reference_model.e_score_correction_bias

    elif component_type == "experts" and num_experts:
        # Extract expert weights
        for expert_id in range(num_experts):
            if hasattr(reference_model, "w1_weight"):
                state_dict[f"experts.{expert_id}.gate_proj.weight"] = reference_model.w1_weight[expert_id]
            if hasattr(reference_model, "w2_weight"):
                state_dict[f"experts.{expert_id}.down_proj.weight"] = reference_model.w2_weight[expert_id]
            if hasattr(reference_model, "w3_weight"):
                state_dict[f"experts.{expert_id}.up_proj.weight"] = reference_model.w3_weight[expert_id]

            # Add quantization scales if needed
            if hasattr(reference_model, "w1_weight"):
                state_dict[f"experts.{expert_id}.gate_proj.weight_scale_inv"] = torch.ones_like(
                    reference_model.w1_weight[expert_id][:, 0:1]
                )
            if hasattr(reference_model, "w2_weight"):
                state_dict[f"experts.{expert_id}.down_proj.weight_scale_inv"] = torch.ones_like(
                    reference_model.w2_weight[expert_id][:, 0:1]
                )
            if hasattr(reference_model, "w3_weight"):
                state_dict[f"experts.{expert_id}.up_proj.weight_scale_inv"] = torch.ones_like(
                    reference_model.w3_weight[expert_id][:, 0:1]
                )

    return state_dict


def format_tensor_debug_info(
    tensor: torch.Tensor,
    name: str,
    sample_size: int = 10,
) -> str:
    """Format tensor information for debug output.

    Args:
        tensor: Tensor to format
        name: Name of the tensor
        sample_size: Number of values to show

    Returns:
        Formatted debug string
    """
    info = [
        f"{name}:",
        f"  Shape: {tensor.shape}",
        f"  Dtype: {tensor.dtype}",
        f"  Device: {tensor.device}",
        f"  Min/Max: {tensor.min():.6f} / {tensor.max():.6f}",
        f"  Mean/Std: {tensor.mean():.6f} / {tensor.std():.6f}",
    ]

    # Add sample values
    flat = tensor.flatten()
    if len(flat) > sample_size:
        sample = flat[:sample_size]
        info.append(f"  First {sample_size} values: {sample.tolist()}")

    # Check for special values
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    zero_count = (tensor == 0).sum().item()

    if nan_count > 0:
        info.append(f"  ⚠️  Contains {nan_count} NaN values")
    if inf_count > 0:
        info.append(f"  ⚠️  Contains {inf_count} Inf values")
    if zero_count == tensor.numel():
        info.append(f"  ⚠️  All values are zero")

    return "\n".join(info)


def save_debug_tensors(
    tensors: Dict[str, torch.Tensor],
    test_name: str,
    iteration: Optional[int] = None,
) -> str:
    """Save tensors for debugging purposes.

    Args:
        tensors: Dictionary of tensors to save
        test_name: Name of the test
        iteration: Optional iteration number

    Returns:
        Path where tensors were saved
    """
    save_dir = Path(TestComparisonConfig.TENSOR_SAVE_PATH) / test_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if iteration is not None:
        save_path = save_dir / f"iter_{iteration}.pt"
    else:
        save_path = save_dir / "tensors.pt"

    torch.save(tensors, save_path)
    logger.info(f"Saved debug tensors to: {save_path}")

    # Also save metadata
    metadata = {
        "test_name": test_name,
        "iteration": iteration,
        "tensor_shapes": {name: list(t.shape) for name, t in tensors.items()},
        "tensor_dtypes": {name: str(t.dtype) for name, t in tensors.items()},
    }

    metadata_path = save_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(save_path)


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute MD5 hash of a tensor for exact comparison.

    Args:
        tensor: Tensor to hash

    Returns:
        MD5 hash string
    """
    # Convert to numpy for consistent hashing
    numpy_array = tensor.detach().cpu().numpy()

    # Create hash
    hasher = hashlib.md5()
    hasher.update(numpy_array.tobytes())
    hasher.update(str(numpy_array.shape).encode())
    hasher.update(str(numpy_array.dtype).encode())

    return hasher.hexdigest()


def verify_memory_config(
    tensor: ttnn.Tensor,
    expected_config: Union[ttnn.MemoryConfig, str],
) -> Tuple[bool, str]:
    """Verify that a tensor has the expected memory configuration.

    Args:
        tensor: TTNN tensor to check
        expected_config: Expected memory configuration

    Returns:
        Tuple of (matches, message)
    """
    actual_config = tensor.memory_config()

    if isinstance(expected_config, str):
        # Convert string to memory config
        if expected_config.upper() == "L1":
            expected_config = ttnn.L1_MEMORY_CONFIG
        elif expected_config.upper() == "DRAM":
            expected_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            return False, f"Unknown memory config string: {expected_config}"

    matches = actual_config == expected_config
    message = f"Memory config: expected={expected_config}, actual={actual_config}"

    return matches, message


def create_test_report(
    test_name: str,
    results: List[Dict],
    save_path: Optional[str] = None,
) -> str:
    """Create a test report summarizing results.

    Args:
        test_name: Name of the test
        results: List of result dictionaries
        save_path: Optional path to save report

    Returns:
        Formatted report string
    """
    report = [
        "=" * 80,
        f"Test Report: {test_name}",
        "=" * 80,
        "",
    ]

    # Summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("passed", False))
    failed_tests = total_tests - passed_tests

    report.append(f"Total Tests: {total_tests}")
    report.append(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    report.append(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    report.append("")

    # Detailed results
    report.append("Detailed Results:")
    report.append("-" * 40)

    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
        report.append(f"\n{i}. {result.get('test_case', 'Unknown')} - {status}")

        if "pcc" in result:
            report.append(f"   PCC: {result['pcc']:.6f}")

        if "message" in result:
            report.append(f"   {result['message']}")

        if not result.get("passed", False) and "error" in result:
            report.append(f"   Error: {result['error']}")

    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)
        logger.info(f"Test report saved to: {save_path}")

    return report_text


class TestTimer:
    """Context manager for timing test execution."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        logger.info(f"Starting test: {self.test_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            logger.info(f"Test completed: {self.test_name} ({duration:.2f}s)")
        else:
            logger.error(f"Test failed: {self.test_name} ({duration:.2f}s)")

    @property
    def duration(self) -> float:
        """Get test duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def get_pcc_threshold(component: str) -> float:
    """Get appropriate PCC threshold for a component.

    Args:
        component: Component name

    Returns:
        PCC threshold value
    """
    thresholds = {
        "router": TestComparisonConfig.ROUTER_WEIGHTS_PCC,
        "experts": TestComparisonConfig.EXPERTS_OUTPUT_PCC,
        "integration": TestComparisonConfig.INTEGRATION_PCC,
        "intermediate": TestComparisonConfig.INTERMEDIATE_PCC,
    }

    return thresholds.get(component.lower(), 0.95)  # Default to 0.95


def validate_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate tensor properties against expectations.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape
        expected_dtype: Expected dtype
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether Inf values are allowed

    Returns:
        Tuple of (valid, list of error messages)
    """
    errors = []

    # Check shape
    if expected_shape and tensor.shape != expected_shape:
        errors.append(f"Shape mismatch: expected {expected_shape}, got {tensor.shape}")

    # Check dtype
    if expected_dtype and tensor.dtype != expected_dtype:
        errors.append(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")

    # Check for NaN
    if not allow_nan and torch.isnan(tensor).any():
        errors.append("Tensor contains NaN values")

    # Check for Inf
    if not allow_inf and torch.isinf(tensor).any():
        errors.append("Tensor contains Inf values")

    valid = len(errors) == 0
    return valid, errors
