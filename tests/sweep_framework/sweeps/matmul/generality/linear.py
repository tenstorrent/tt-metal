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

# Possible tensor dimensions for linear
DIM_SIZES = [32, 1, 0]


def get_bias_shapes(shape_a, shape_b):
    """
    Generate different valid shapes for bias to test broadcasting.
    Works as an iterator, yielding one bias shape at a time.
    """
    rank_a, rank_b = len(shape_a), len(shape_b)
    # Generate all valid bias shapes based on input ranks
    if rank_a == 2 and rank_b == 2:  # Matrix-matrix: (m, k) x (k, n) -> (m, n)
        m, k = shape_a
        n = shape_b[-1]
        yield (n,)  # Standard case
        yield (1, n)  # Broadcast first dimension
        yield (m, n)  # Full shape
        yield (1,)  # Broadcast all dimensions
    elif rank_a == 1 and rank_b == 2:  # Vector-matrix: (k) x (k, n) -> (n)
        n = shape_b[-1]
        yield (n,)  # Standard case
        yield (1,)  # Broadcast all dimensions
    elif rank_a == 1 and rank_b == 1:  # Vector-vector: (k) x (k) -> scalar
        yield tuple()  # Broadcast all dimensions
        pass  # No bias for vector-vector
    elif rank_a == 3 and rank_b == 2:  # Rank 3: (b, m, k) x (k, n) -> (b, m, n)
        b, m, k = shape_a
        n = shape_b[-1]
        # For rank 3, bias must be broadcastable to (b, m, n)
        yield (n,)  # Standard case
        yield (1, n)  # Broadcast first dimension
        yield (1,)  # Broadcast all dimensions
    elif rank_a == 4 and rank_b == 2:  # Rank 4: (b, c, m, k) x (k, n) -> (b, c, m, n)
        b, c, m, k = shape_a
        n = shape_b[-1]
        yield (n,)  # Standard case
        yield (1, n)  # Broadcast first dimension
        yield (1, 1, n)  # Broadcast batch and channel dimensions
        yield (1, 1, 1, n)  # Broadcast all dimensions
        yield (1, 1, m, n)  # Broadcast batch and channel dimensions
        yield (1, c, 1, n)  # Broadcast batch and middle dimensions
        yield (b, 1, 1, n)  # Broadcast channel and middle dimensions
        yield (b, c, m, n)  # Full shape
        yield (1,)  # Broadcast all dimensions


def get_linear_shapes(rank_a, rank_b):
    """
    Generate different valid shapes for linear operation.
    For each tensor rank, return a tuple of (shape_a, shape_b, bias_shape).
    """
    for shape_a in itertools.product(DIM_SIZES, repeat=rank_a):
        # For shape_b, we need to ensure the inner dimension matches shape_a's last dimension
        # This is because for linear operation, the inner dimension must match
        inner_dim = shape_a[-1]
        # Create shape_b with matching inner dimension
        outer_dims = itertools.product(DIM_SIZES, repeat=rank_b - 1)
        if rank_b == 1:
            shape_bs = [(inner_dim,)]
        else:
            shape_bs = [(inner_dim,) + outer_dim for outer_dim in outer_dims]
        for shape_b in shape_bs:
            # First yield without bias
            yield (shape_a, shape_b, None)

            # TODO: Enable after GH issue 16599 is resolved
            # Then yield with bias
            # for bias_shape in get_bias_shapes(shape_a, shape_b):
            #     yield (shape_a, shape_b, bias_shape)


general = {
    "transpose_a": [False],
    "transpose_b": [False],
}

# Create parameter combinations for different test scenarios
parameters = {
    # Matrix-matrix multiplication: (m, k) x (k, n) -> (m, n)
    "matrix_matrix": {
        "shapes": get_linear_shapes(2, 2),
    },
    # Vector-matrix: (k) x (k, n) -> (n)
    "vector_matrix": {
        "shapes": get_linear_shapes(1, 2),
    },
    # Vector-vector: (k) x (k) -> scalar
    "vector_vector": {
        "shapes": get_linear_shapes(1, 1),
    },
    # Rank 3 tensor: (b, m, k) x (k, n) -> (b, m, n)
    "rank3_matrix": {
        "shapes": get_linear_shapes(3, 2),
    },
    # Rank 4 tensor: (b, c, m, k) x (k, n) -> (b, c, m, n)
    "rank4_matrix": {
        "shapes": get_linear_shapes(4, 2),
    },
}

for p in parameters.values():
    p.update(general)


def run_linear(device, shapes, transpose_a, transpose_b) -> tuple:
    """
    Test the compatibility of the torch and ttnn linear operations for the given
    tensor shapes, transpose options, and bias shape.
    """
    shape_a, shape_b, shape_bias = shapes

    # Create random tensors with appropriate dimensions
    torch_a = torch.randn(*shape_a, dtype=torch.float32)
    torch_b = torch.randn(*shape_b, dtype=torch.float32)

    # Apply transpose to b for torch.linear (weight matrix)
    # In torch.linear, weight matrix is expected to be (out_features, in_features)
    # but internally it's transposed during the operation
    torch_weight = torch_b
    if len(shape_b) >= 2:
        torch_weight = torch.transpose(torch_weight, -1, -2)

    # Create bias tensor if needed
    torch_bias = None
    ttnn_bias = None
    if shape_bias is not None:
        torch_bias = torch.randn(*shape_bias, dtype=torch.float32) if shape_bias != tuple() else torch.randn(())
        ttnn_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    # Create ttnn tensors
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Handle exceptions in torch
    torch_errored = False
    torch_error_msg = ""
    try:
        # For torch.linear, input is a (first), weight matrix is b (transposed), and bias is optional
        torch_result = torch.nn.functional.linear(torch_a, torch_weight, torch_bias)
    except Exception as e:
        torch_errored = True
        torch_error_msg = str(e)

    # Run ttnn linear with the same operations
    ttnn_errored = False
    ttnn_error_msg = ""
    start_time = start_measuring_time()
    try:
        op_output_tensor = ttnn.linear(ttnn_a, ttnn_b, bias=ttnn_bias, transpose_a=transpose_a, transpose_b=transpose_b)
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
    flop_counts = list(shape_a) + [
        2,
        shape_b[-1],
    ]  # shape_a: all batch dimensions, m, k; shape_b[-1]: n, disregards addition
    tensors = [ttnn_a, ttnn_b, op_output_tensor]
    if ttnn_bias:
        tensors.append(ttnn_bias)
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf, flop_counts)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_linear(
    device,
    shapes,
    transpose_a,
    transpose_b,
):
    result, error_msg = run_linear(
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
    return run_linear(
        device,
        shapes,
        transpose_a,
        transpose_b,
    )
