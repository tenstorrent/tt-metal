# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("matrix_size", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_addmm_square_matrices(device, dtype, matrix_size):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((matrix_size, matrix_size), dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn((matrix_size, matrix_size), dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn((matrix_size, matrix_size), dtype=torch.bfloat16)

    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    target_pcc = 0.9999

    if dtype == ttnn.bfloat8_b:
        target_pcc = 0.999

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=target_pcc)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("matrix_size", [4, 8, 16, 32])
@pytest.mark.parametrize("alpha", [-0.5, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("beta", [-0.5, 0.0, 0.5, 1.0, 1.5])
def test_addmm_with_alpha_beta(device, dtype, matrix_size, alpha, beta):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16)

    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor, alpha=alpha, beta=beta)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha=alpha, beta=beta)
    output_tensor = ttnn.to_torch(output_tensor)

    target_pcc = 0.9999

    if dtype == ttnn.bfloat8_b:
        target_pcc = 0.999

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=target_pcc)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "matrix_dims",
    [
        (2, 3, 4),
        (4, 6, 8),
        (8, 12, 16),
        (3, 5, 7),
        (16, 8, 32),
        (32, 16, 8),
        (128, 32, 64),
        (32, 128, 64),
    ],
)
def test_addmm_rectangular_matrices(device, dtype, matrix_dims):
    torch.manual_seed(0)

    n, m, p = matrix_dims

    # mat1: (n, m), mat2: (m, p), input: (n, p) -> result: (n, p)
    torch_input_tensor = torch.randn(n, p, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(n, m, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(m, p, dtype=torch.bfloat16)

    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = 0.9999

    if dtype == ttnn.bfloat8_b:
        pcc = 0.999

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("size", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("case_type", ["matrix_vector"])  # TODO "vector_matrix" not working
def test_vector_matrix_multiplication(device, dtype, size, case_type):
    """
    Test vector-matrix and matrix-vector multiplication cases:
    - vector_matrix: n=1, testing (1, m) @ (m, p) + (1, p)
    - matrix_vector: p=1, testing (n, m) @ (m, 1) + (n, 1)
    """
    torch.manual_seed(0)

    if case_type == "vector_matrix":
        # Vector-Matrix case: mat1=(1, size), mat2=(size, size), input=(1, size)
        n, m, p = 1, size, size
    else:  # matrix_vector
        # Matrix-Vector case: mat1=(size, size), mat2=(size, 1), input=(size, 1)
        n, m, p = size, size, 1

    # Create torch tensors with the determined shapes
    torch_input_tensor = torch.randn(n, p, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(n, m, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(m, p, dtype=torch.bfloat16)

    # Compute expected output using torch
    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor)

    # Convert to ttnn tensors
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )

    # Perform the operation using ttnn
    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    target_pcc = 0.9999

    if dtype == ttnn.bfloat8_b:
        target_pcc = 0.999

    # Assert the results match
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=target_pcc)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "shape",
    [
        # (n, m, p) where:
        # n = input rows/mat1 rows
        # m = mat1 columns/mat2 rows
        # p = input columns/mat2 columns
        (37, 55, 41),  # All dimensions not multiples of 32
        (64, 37, 41),  # Only first dimension is multiple of 32
        (37, 64, 41),  # Only middle dimension is multiple of 32
        (37, 55, 64),  # Only last dimension is multiple of 3
        (31, 33, 65),  # Just off from multiples of 32
        (95, 127, 63),  # Larger dimensions not multiple of 32
    ],
)
def test_addmm_non_tile_multiple_dimensions(device, dtype, shape):
    torch.manual_seed(0)

    n, m, p = shape

    torch_input_tensor = torch.randn(n, p, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(n, m, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(m, p, dtype=torch.bfloat16)

    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor)
    output_tensor_torch = ttnn.to_torch(output_tensor)

    target_pcc = 0.9999

    if dtype == ttnn.bfloat8_b:
        target_pcc = 0.999

    assert_with_pcc(torch_output_tensor, output_tensor_torch, pcc=target_pcc)


def test_alpha_zero_should_throw_error(device):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(4, 4, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )

    try:
        ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha=0.0)
    except Exception as e:
        if not "alpha parameter cannot be 0" in str(e):
            pytest.fail("Expected error message not found.")
    else:
        pytest.fail("Calling ttnn.addmm with alpha=0 should throw an error.")


def test_input_tensor_with_invalid_shape_should_throw_error(device):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(8, 8, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(4, 4, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )

    try:
        ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor)
    except Exception as e:
        if not "input_tensor must have shape matching one of result of mat1_tensor @ mat2_tensor" in str(e):
            pytest.fail("Expected error message not found.")
    else:
        pytest.fail("Calling ttnn.addmm with incompatible shapes should throw an error.")


def test_input_tensor_with_invalid_shape_should_be_ignored_if_beta_is_0(device):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(8, 8, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(4, 4, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )

    ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor, beta=0.0)


def test_cast_to_another_dtype(device):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(4, 4, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor, dtype=ttnn.float32)
    assert output_tensor.dtype == ttnn.float32, "Output tensor must be float32"


def test_unsupported_dtype_should_throw_error(device):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn(4, 4, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        device=device,
    )

    try:
        ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor)
    except Exception as e:
        if not "only ttnn.bfloat16, ttnn.float32 and ttnn.bfloat8_b types are supported" in str(e):
            pytest.fail("Expected error message not found.")
    else:
        pytest.fail("Calling ttnn.addmm with invalid dtype of input tensors should throw an error.")


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_addmm_with_output_tensor_inplace_op(device, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((32, 32), dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn((32, 32), dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn((32, 32), dtype=torch.bfloat16)
    torch_out_tensor = torch.zeros((32, 32), dtype=torch.bfloat16)

    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )
    out_tensor = ttnn.from_torch(
        torch_out_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor, optional_output_tensor=out_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    out_tensor = ttnn.to_torch(out_tensor)

    target_pcc = 0.9999

    if dtype == ttnn.bfloat8_b:
        target_pcc = 0.999

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=target_pcc)
    assert_with_pcc(torch_output_tensor, out_tensor, pcc=target_pcc)


def test_addmm_with_output_tensor_inplace_op_with_different_dtype(device):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((32, 32), dtype=torch.bfloat16)
    torch_mat1_tensor = torch.randn((32, 32), dtype=torch.bfloat16)
    torch_mat2_tensor = torch.randn((32, 32), dtype=torch.bfloat16)
    torch_out_tensor = torch.zeros((32, 32), dtype=torch.float32)

    torch_output_tensor = torch.addmm(torch_input_tensor, torch_mat1_tensor, torch_mat2_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat1_tensor = ttnn.from_torch(
        torch_mat1_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    mat2_tensor = ttnn.from_torch(
        torch_mat2_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    out_tensor = ttnn.from_torch(
        torch_out_tensor,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
    )

    output_tensor = ttnn.addmm(input_tensor, mat1_tensor, mat2_tensor, optional_output_tensor=out_tensor)

    assert out_tensor.dtype == ttnn.float32, "out_tensor must be float32"
    assert output_tensor.dtype == ttnn.float32, "output_tensor must be float32"

    output_tensor = ttnn.to_torch(output_tensor)
    out_tensor = ttnn.to_torch(out_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9999)
    assert_with_pcc(torch_output_tensor, out_tensor, pcc=0.9999)
