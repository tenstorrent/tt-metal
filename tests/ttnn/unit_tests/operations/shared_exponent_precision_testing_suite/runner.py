# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from tqdm import tqdm
import ttnn

from generators import generate_distributions, generate_test_patterns
from constants import ShapeType, OperationType, MatmulTTConfig, ResultKeys


# Helper functions
def _compute_ulp_error(reference: torch.Tensor, test: torch.Tensor) -> dict:
    """Compute Units in Last Place error"""
    # Get the epsilon for the reference values
    eps = torch.finfo(reference.dtype).eps

    # Compute ULP for each element
    ulp_per_element = torch.abs(reference - test) / (eps * torch.abs(reference))

    return {
        "mean": ulp_per_element.mean().item(),
        "max": ulp_per_element.max().item(),
        "percentiles": {
            "50": torch.quantile(ulp_per_element, 0.5).item(),
            "90": torch.quantile(ulp_per_element, 0.9).item(),
            "99": torch.quantile(ulp_per_element, 0.99).item(),
        },
    }


def _compute_metrics(reference: torch.Tensor, test: torch.Tensor) -> dict:
    """Compute all comparison metrics"""
    # Flatten for correlation
    ref_flat = reference.flatten()
    test_flat = test.flatten()

    # Compute ULP error
    ulp_errors = _compute_ulp_error(reference, test)

    # Relative error
    rel_error = torch.abs(reference - test) / (torch.abs(reference) + 1e-10)

    metrics = {
        ResultKeys.PCC_KEY: torch.corrcoef(torch.stack([ref_flat, test_flat]))[0, 1].item(),
        ResultKeys.ALLCLOSE_1E_2_KEY: torch.allclose(reference, test, rtol=1e-2),
        ResultKeys.ALLCLOSE_1E_3_KEY: torch.allclose(reference, test, rtol=1e-3),
        ResultKeys.MAX_ABS_ERROR_KEY: torch.max(torch.abs(reference - test)).item(),
        ResultKeys.MEAN_ABS_ERROR_KEY: torch.mean(torch.abs(reference - test)).item(),
        ResultKeys.MAX_REL_ERROR_KEY: torch.max(rel_error).item(),
        ResultKeys.MEAN_REL_ERROR_KEY: torch.mean(rel_error).item(),
        ResultKeys.ULP_MEAN_KEY: ulp_errors["mean"],
        ResultKeys.ULP_MAX_KEY: ulp_errors["max"],
        ResultKeys.ULP_PERCENTILES_KEY: ulp_errors["percentiles"],
    }

    return metrics


def _perform_torch_operation(
    tensor: torch.Tensor, operation: str = "sum", axis: int = 0, optional_second_tensor: torch.Tensor = None
) -> torch.Tensor:
    """Perform the specified operation on the tensor"""
    if operation == OperationType.SUM_KEY:
        return torch.sum(tensor, dim=axis)
    elif operation == OperationType.MEAN_KEY:
        return torch.mean(tensor, dim=axis)
    elif operation == OperationType.MAX_KEY:
        return torch.max(tensor, dim=axis)[0]
    elif operation in [OperationType.MATMUL_KEY, OperationType.MATMUL_TT_KEY]:
        return torch.matmul(tensor, optional_second_tensor)
    elif operation == OperationType.SOFTMAX_KEY:
        return torch.softmax(tensor, dim=axis)
    elif operation == OperationType.CUMSUM_KEY:
        return torch.cumsum(tensor, dim=axis)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def _perform_ttnn_operation(
    tensor: ttnn.Tensor, device, operation: str, axis: int = 0, optional_second_tensor: ttnn.Tensor = None
) -> ttnn.Tensor:
    """Perform the specified operation using ttnn"""
    if operation == OperationType.SUM_KEY:
        return ttnn.sum(tensor, dim=axis)
    elif operation == OperationType.MEAN_KEY:
        return ttnn.mean(tensor, dim=axis)
    elif operation == OperationType.MAX_KEY:
        return ttnn.max(tensor, dim=axis)
    elif operation in [OperationType.MATMUL_KEY, OperationType.MATMUL_TT_KEY]:
        return ttnn.matmul(tensor, optional_second_tensor)
    elif operation == OperationType.SOFTMAX_KEY:
        return ttnn.softmax(tensor, dim=axis)
    elif operation == OperationType.CUMSUM_KEY:
        return ttnn.cumsum(tensor, dim=axis)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def _run_precision_test(
    input_tensor: torch.Tensor, device, operation: str = "sum", axis: int = 0, matmul_config: dict = None
) -> dict:
    """Run a single precision test comparing bfloat16 vs bfloat8_b results"""

    # Store original for reference
    torch_optional_second_tensor = None
    ttnn_optional_second_tensor = None
    torch_tensor_bf16 = input_tensor.bfloat16()
    if operation in [OperationType.MATMUL_KEY, OperationType.MATMUL_TT_KEY]:
        # For matmul, create a second tensor with compatible shape
        second_shape = (input_tensor.shape[1], input_tensor.shape[0])
        torch_optional_second_tensor = torch.randn(second_shape).bfloat16()

    # Convert to ttnn tensors
    ttnn_tensor_bf8_b = ttnn.from_torch(torch_tensor_bf16, device=device, layout=ttnn.Layout.TILE, dtype=ttnn.bfloat8_b)

    if operation == OperationType.MATMUL_KEY:
        ttnn_optional_second_tensor = ttnn.from_torch(
            torch_optional_second_tensor,
            device=device,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.bfloat8_b,
        )
    elif operation == OperationType.MATMUL_TT_KEY:
        ttnn_optional_second_tensor = ttnn.from_torch(
            torch_optional_second_tensor,
            device=device,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.bfloat8_b,
            tile=ttnn.Tile(
                (32, matmul_config[MatmulTTConfig.TILE_W_KEY]),
                transpose_tile=matmul_config[MatmulTTConfig.TRANSPOSE_KEY],
            ),
        )

    # Perform operation in both precisions
    result_torch_tensor_bf16 = _perform_torch_operation(
        torch_tensor_bf16, operation, axis, torch_optional_second_tensor
    )
    result_ttnn_tensor_bf8_b = _perform_ttnn_operation(
        ttnn_tensor_bf8_b, device, operation, axis, ttnn_optional_second_tensor
    )

    # Compute metrics
    result_ttnn_tensor_bf8_b_converted = ttnn.to_torch(result_ttnn_tensor_bf8_b, dtype=torch.float32)
    result_torch_tensor_bf16 = result_torch_tensor_bf16.float()
    metrics = _compute_metrics(result_torch_tensor_bf16, result_ttnn_tensor_bf8_b_converted)

    # Also store input statistics for analysis
    metrics["input_stats"] = {
        "min": input_tensor.min().item(),
        "max": input_tensor.max().item(),
        "mean": input_tensor.mean().item(),
        "std": input_tensor.std().item(),
        "range": input_tensor.max().item() - input_tensor.min().item(),
    }

    return metrics


def _run_shape_experiments(shape, operations: list, axes: int, device) -> dict:
    """Run all experiments for a specific shape"""

    results = {}

    # Get pattern generators
    pattern_generators = generate_test_patterns(shape)

    # Get distribution generators
    distributions = generate_distributions(shape)

    # Run all combinations
    for pattern_name, pattern_gen in tqdm(pattern_generators.items(), desc="Patterns", leave=False):
        logger.info(f"  Testing pattern: {pattern_name}")
        results[pattern_name] = {}

        for dist_name, dist_tensor in tqdm(distributions.items(), desc="Distributions", leave=False):
            # Generate test input by applying pattern to distribution
            test_input = pattern_gen() * dist_tensor

            results[pattern_name][dist_name] = {}

            # Test each operation
            for operation in tqdm(operations, desc="Operations", leave=False):
                # Some operations don't use axis
                if operation == OperationType.MATMUL_KEY:
                    results[pattern_name][dist_name][operation] = _run_precision_test(test_input, device, operation)
                elif operation == OperationType.MATMUL_TT_KEY:
                    for tile_w in [16, 32]:
                        for transpose in [False, True]:
                            config = {
                                MatmulTTConfig.TILE_W_KEY: tile_w,
                                MatmulTTConfig.TRANSPOSE_KEY: transpose,
                            }
                            if operation not in results[pattern_name][dist_name]:
                                results[pattern_name][dist_name][operation] = {}
                            if tile_w not in results[pattern_name][dist_name][operation]:
                                results[pattern_name][dist_name][operation][tile_w] = {}
                            if transpose not in results[pattern_name][dist_name][operation][tile_w]:
                                results[pattern_name][dist_name][operation][tile_w][transpose] = {}
                            results[pattern_name][dist_name][operation][tile_w][transpose] = _run_precision_test(
                                test_input, device, operation, 0, matmul_config=config
                            )
                else:
                    # Test along each axis
                    for axis in axes:
                        if operation not in results[pattern_name][dist_name]:
                            results[pattern_name][dist_name][operation] = {}
                        results[pattern_name][dist_name][operation][axis] = _run_precision_test(
                            test_input, device, operation, axis
                        )

    return results


# Main module function
def run_experiments() -> dict:
    """Run all experiments with different patterns, distributions, and operations"""

    # Open device
    device = ttnn.open_device(device_id=0)

    results = {}

    # Configuration
    operations = [
        OperationType.SUM_KEY,
        OperationType.MEAN_KEY,
        OperationType.MAX_KEY,
        OperationType.SOFTMAX_KEY,
        OperationType.MATMUL_KEY,
        OperationType.MATMUL_TT_KEY,
    ]
    axes = [0, 1]  # Test both row-wise and column-wise operations

    # Test 1: Single tile (32x32)
    logger.info("=== Running single tile experiments ===")
    single_tile_shape = (32, 32)
    results[ShapeType.SINGLE_TILE_KEY] = _run_shape_experiments(single_tile_shape, operations, axes, device)

    # Test 2: Multiple tiles (16x16 tiles of 32x32 each = 512x512)
    logger.info("=== Running multi-tile experiments ===")
    multi_tile_shape = (512, 512)
    results[ShapeType.MULTI_TILE_KEY] = _run_shape_experiments(multi_tile_shape, operations, axes, device)

    # Test 3: Rectangular shapes (to test non-square behavior)
    logger.info("=== Running rectangular experiments ===")
    rect_shapes = [(32, 128), (128, 32), (64, 256)]
    for shape in rect_shapes:
        key = ShapeType.RECTANGULAR_KEY + "-" + str(shape)
        results[key] = _run_shape_experiments(shape, operations, axes, device)

    ttnn.close_device(device)

    return results
