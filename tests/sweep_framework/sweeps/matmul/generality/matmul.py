# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import pytest
import random
import torch
import ttnn
from loguru import logger

from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Possible tensor dimensions for matmul
DIM_SIZES = [32, 1, 0]

# Create parameter combinations for different test scenarios
parameters = {
    # Matrix-matrix multiplication: (m, k) x (k, n) -> (m, n)
    "matrix_matrix": {
        "shapes": [((m, k), (k, n)) for m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # Batched matrix multiplication: (b, m, k) x (1, k, n) -> (b, m, n)
    "batched_matrix": {
        "shapes": [
            ((b, m, k), (1, k, n)) for b, m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)
        ],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # Vector-matrix: (k) x (k, n) -> (n)
    "vector_matrix": {
        "shapes": [((k,), (k, n)) for k, n in itertools.product(DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # Matrix-vector: (m, k) x (k) -> (m)
    "matrix_vector": {
        "shapes": [((m, k), (k,)) for m, k in itertools.product(DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # Vector-vector: (k) x (k) -> scalar
    "vector_vector": {
        "shapes": [((k,), (k,)) for k in DIM_SIZES],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 1D x 3D: (k) x (b, k, n) -> (b, n)
    "vector_3d": {
        "shapes": [((k,), (1, k, n)) for k, n in itertools.product(DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 3D x 1D: (b, m, k) x (k) -> (b, m)
    "3d_vector": {
        "shapes": [((b, m, k), (k,)) for b, m, k in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 1D x 4D: (k) x (a, b, k, n) -> (a, b, n)
    "vector_4d": {
        "shapes": [((k,), (1, 1, k, n)) for k, n in itertools.product(DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 4D x 1D: (a, b, m, k) x (k) -> (a, b, m)
    "4d_vector": {
        "shapes": [
            ((a, b, m, k), (k,)) for a, b, m, k in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)
        ],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 2D matrix x 3D matrix: (m, k) x (1, k, n) -> (m, n)
    "matrix_3d": {
        "shapes": [((m, k), (1, k, n)) for m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 3D matrix x 2D matrix: (b, m, k) x (k, n) -> (b, m, n)
    "3d_matrix": {
        "shapes": [((b, m, k), (k, n)) for b, m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 2D matrix x 4D matrix: (m, k) x (1, 1, k, n) -> (m, n)
    "matrix_4d": {
        "shapes": [((m, k), (1, 1, k, n)) for m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES)],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 4D matrix x 2D matrix: (a, b, m, k) x (k, n) -> (a, b, m, n)
    "4d_matrix": {
        "shapes": [
            ((a, b, m, k), (k, n))
            for a, b, m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)
        ],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 3D matrix x 4D matrix: (1, m, k) x (a, b, k, n) -> (a, b, m, n)
    "3d_4d": {
        "shapes": [
            ((a, m, k), (1, 1, k, n)) for a, m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)
        ],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 4D matrix x 3D matrix: (a, b, m, k) x (1, k, n) -> (a, b, m, n)
    "4d_3d": {
        "shapes": [
            ((a, b, m, k), (1, k, n))
            for a, b, m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)
        ],
        "transpose_a": [False],
        "transpose_b": [False],
    },
    # 4D matrix x 4D matrix: (a, b, m, k) x (c, d, k, n) -> (a, b, c, d, m, n)
    "4d_4d": {
        "shapes": [
            ((a, b, m, k), (1, 1, k, n))
            for a, b, m, k, n in itertools.product(DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES, DIM_SIZES)
        ],
        "transpose_a": [False],
        "transpose_b": [False],
    },
}


def get_matmul_dimensions(shape_a, shape_b, transpose_a, transpose_b):
    """
    Calculate the dimensions that will be multiplied in matmul operation
    and the expected output shape after accounting for transpositions.
    """
    # Handle transpose for shapes with rank >= 2
    effective_shape_a = list(shape_a)
    effective_shape_b = list(shape_b)

    if transpose_a and len(shape_a) >= 2:
        effective_shape_a[-1], effective_shape_a[-2] = effective_shape_a[-2], effective_shape_a[-1]

    if transpose_b and len(shape_b) >= 2:
        effective_shape_b[-1], effective_shape_b[-2] = effective_shape_b[-2], effective_shape_b[-1]

    # For matmul to be valid, the inner dimensions must match
    # For 1D tensors, special handling applies
    if len(effective_shape_a) == 1 and len(effective_shape_b) == 1:
        # Vector-vector: inner dimension must match
        inner_dim_a = effective_shape_a[0]
        inner_dim_b = effective_shape_b[0]
        output_shape = (1,)  # Scalar output (represented as 1D tensor with 1 element)
    elif len(effective_shape_a) == 1:
        # Vector-matrix or vector-ND
        inner_dim_a = effective_shape_a[0]
        inner_dim_b = effective_shape_b[-2]
        output_shape = effective_shape_b[:-2] + (effective_shape_b[-1],)
    elif len(effective_shape_b) == 1:
        # Matrix-vector or ND-vector
        inner_dim_a = effective_shape_a[-1]
        inner_dim_b = effective_shape_b[0]
        output_shape = effective_shape_a[:-1]
    else:
        # Matrix-matrix or ND-ND
        inner_dim_a = effective_shape_a[-1]
        inner_dim_b = effective_shape_b[-2]
        output_shape = effective_shape_a[:-1] + (effective_shape_b[-1],)

    return inner_dim_a, inner_dim_b, output_shape


def run_matmul(device, shapes, transpose_a, transpose_b) -> tuple:
    """
    Test the compatibility of the torch and ttnn matmul for the given operation and different
    tensor shapes, transpose options.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    """
    shape_a, shape_b = shapes

    # Create random tensors with appropriate dimensions
    torch_a = torch.randn(*shape_a, dtype=torch.float32)
    torch_b = torch.randn(*shape_b, dtype=torch.float32)

    # Create ttnn tensors
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply transpose if needed
    if transpose_a:
        torch_a = torch.transpose(torch_a, -1, -2) if len(shape_a) > 1 else torch_a
    if transpose_b:
        torch_b = torch.transpose(torch_b, -1, -2) if len(shape_b) > 1 else torch_b

    # Handle exceptions in torch
    torch_errored = False
    torch_error_msg = ""
    try:
        torch_result = torch.matmul(torch_a, torch_b)
    except Exception as e:
        torch_errored = True
        torch_error_msg = str(e)

    # Run ttnn matmul with the same operations
    ttnn_errored = False
    ttnn_error_msg = ""
    start_time = start_measuring_time()
    try:
        op_output_tensor = ttnn.matmul(ttnn_a, ttnn_b, transpose_a=transpose_a, transpose_b=transpose_b)
        output_tensor = ttnn.to_torch(op_output_tensor)
    except Exception as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)
    e2e_perf = stop_measuring_time(start_time)

    # Compare error behavior
    if torch_errored != ttnn_errored:
        return [
            (
                False,
                f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})",
            ),
            e2e_perf,
        ]

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return [(True, ""), e2e_perf]

    # Check shape compatibility
    if output_tensor.shape != torch_result.shape:
        return [
            (
                False,
                f"shape mismatch: torch: {torch_result.shape}, ttnn: {output_tensor.shape}",
            ),
            e2e_perf,
        ]

    # Allow some tolerance for numeric differences
    atol = rtol = 0.1
    allclose = (torch.allclose(torch_result, output_tensor, atol=atol, rtol=rtol, equal_nan=True),)
    if not allclose:
        return [(False, f"mismatch in allclose: torch: {torch_result}, ttnn: {output_tensor}"), e2e_perf]

    expected_pcc = 0.99
    tensors = [ttnn_a, ttnn_b, op_output_tensor]

    flop_counts = list(shape_a) + [2, shape_b[-1]]  # shape_a: all batch dimensions, m, k; shape_b[-1]: n
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf, flop_counts)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_matmul(
    device,
    shapes,
    transpose_a,
    transpose_b,
):
    result, error_msg = run_matmul(
        device,
        shapes,
        transpose_a,
        transpose_b,
    )
    assert result, error_msg


def run(
    shapes,
    transpose_a,
    transpose_b,
    *,
    device,
) -> tuple:
    return run_matmul(
        device,
        shapes,
        transpose_a,
        transpose_b,
    )
