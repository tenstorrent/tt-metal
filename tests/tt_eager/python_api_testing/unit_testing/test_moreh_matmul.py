# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensor_shape(input_a_shape, input_b_shape, transpose_b):
    a_b1 = input_a_shape[0]
    a_b2 = input_a_shape[1]
    a_m = input_a_shape[2]
    a_k = input_a_shape[3]

    b_b1 = input_b_shape[0]
    b_b2 = input_b_shape[1]

    b_k = input_b_shape[3] if transpose_b else input_b_shape[2]
    b_n = input_b_shape[2] if transpose_b else input_b_shape[3]
    return a_b1, a_b2, a_m, a_k, b_b1, b_b2, b_k, b_n


def get_tensors(input_a_shape, input_b_shape, transpose_b, device):
    torch.manual_seed(2023)
    dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    # create input tensors using torch
    a = torch.randn(input_a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(input_b_shape, dtype=torch.bfloat16).float()

    # TT matmul
    # set different padded value for tt_a and tt_b.
    tt_a = (
        ttl.tensor.Tensor(a.reshape(-1).tolist(), input_a_shape, dtype, cpu_layout)
        .pad_to_tile(1)
        .to(npu_layout)
        .to(device)
    )

    tt_b = (
        ttl.tensor.Tensor(b.reshape(-1).tolist(), input_b_shape, dtype, cpu_layout)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )

    torch_a = a.bfloat16()
    torch_b = torch.transpose(b.bfloat16(), 2, 3) if transpose_b else b.bfloat16()

    return tt_a, tt_b, torch_a, torch_b


def compare(tt_out, torch_out, atol=0.2, rtop=0.2):
    # TODO: better way to compare results
    allclose_result = torch.allclose(tt_out, torch_out, atol=0.2, rtol=0.2)
    isclose_sum = torch.isclose(tt_out, torch_out, atol=0.2, rtol=0.2).sum()
    isclose_true_ratio = float(isclose_sum) / torch.numel(torch_out)
    return allclose_result, isclose_true_ratio


@pytest.mark.parametrize(
    "input_a_shape",
    (
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 11],
        [1, 1, TILE_HEIGHT * 2 - 1, TILE_WIDTH * 2],
        [1, 1, TILE_HEIGHT * 9 - 7, TILE_WIDTH * 3 - 10],
        [1, 1, TILE_HEIGHT * 18 - 17, TILE_WIDTH * 2 - 10],
    ),
)

# input_b_shape[2] is dummy
@pytest.mark.parametrize(
    "input_b_shape",
    (
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [1, 1, TILE_HEIGHT, TILE_WIDTH - 1],
        [1, 1, TILE_HEIGHT, TILE_WIDTH * 2 - 1],
        [1, 1, TILE_HEIGHT, TILE_WIDTH * 12 - 10],
        [1, 1, TILE_HEIGHT, TILE_WIDTH * 24 - 20],
    ),
)
def test_moreh_matmul(input_a_shape, input_b_shape, device):
    # check matmul shape
    transpose_b = False
    input_b_shape[2] = input_a_shape[3]
    a_b1, a_b2, a_m, a_k, b_b1, b_b2, b_k, b_n = get_tensor_shape(input_a_shape, input_b_shape, transpose_b)
    output_shape = [a_b1 if a_b1 >= b_b1 else b_b1, a_b2 if a_b2 >= b_b2 else b_b2, a_m, b_n]

    if a_k != b_k:
        pytest.skip(f"k dim {a_k} and {b_k} is not the same")

    if not ((a_b2 == b_b2) or (a_b2 == 1) or (b_b2 == 1)):
        pytest.skip(f"The size of tensor a {a_b2} must match the size of tensor b {b_b2} at non-singleton dimension 1")

    if not ((a_b1 == b_b1) or (a_b1 == 1) or (b_b1 == 1)):
        pytest.skip(f"The size of tensor a {a_b1} must match the size of tensor b {b_b1} at non-singleton dimension 0")

    # get tensors
    tt_a, tt_b, torch_a, torch_b = get_tensors(input_a_shape, input_b_shape, transpose_b, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = (
        ttl.operations.primary.moreh_matmul(tt_a, tt_b, transpose_b=transpose_b)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    torch_out = torch.matmul(torch_a, torch_b)

    # compare results
    allclose_result, isclose_true_ratio = compare(tt_out, torch_out)
    assert allclose_result or isclose_true_ratio > 0.95


@pytest.mark.parametrize(
    "input_a_shape",
    (
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [1, 2, TILE_HEIGHT * 5 - 1, TILE_WIDTH],
        [2, 1, TILE_HEIGHT * 5 - 1, TILE_WIDTH - 11],
        [2, 2, TILE_HEIGHT * 5 - 7, TILE_WIDTH - 17],
        [1, 5, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 3],
        [5, 1, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 3 - 11],
        [5, 5, TILE_HEIGHT * 20 - 7, TILE_WIDTH * 3 - 1],
    ),
)

# input_b_shape[2] is dummy
@pytest.mark.parametrize(
    "input_b_shape",
    (
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [1, 2, TILE_HEIGHT, TILE_WIDTH * 5 - 1],
        [2, 1, TILE_HEIGHT, TILE_WIDTH * 5 - 13],
        [2, 2, TILE_HEIGHT, TILE_WIDTH * 5 - 31],
        [1, 5, TILE_HEIGHT, TILE_WIDTH * 10 - 1],
        [5, 1, TILE_HEIGHT, TILE_WIDTH * 10 - 13],
        [5, 5, TILE_HEIGHT, TILE_WIDTH * 20 - 1],
    ),
)
def test_batched_moreh_matmul(input_a_shape, input_b_shape, device):
    transpose_b = False
    input_b_shape[2] = input_a_shape[3]

    # check matmul shape
    a_b1, a_b2, a_m, a_k, b_b1, b_b2, b_k, b_n = get_tensor_shape(input_a_shape, input_b_shape, transpose_b)
    output_shape = [a_b1 if a_b1 >= b_b1 else b_b1, a_b2 if a_b2 >= b_b2 else b_b2, a_m, b_n]

    if a_k != b_k:
        pytest.skip(f"k dim {a_k} and {b_k} is not the same")

    if not ((a_b2 == b_b2) or (a_b2 == 1) or (b_b2 == 1)):
        pytest.skip(f"The size of tensor a {a_b2} must match the size of tensor b {b_b2} at non-singleton dimension 1")

    if not ((a_b1 == b_b1) or (a_b1 == 1) or (b_b1 == 1)):
        pytest.skip(f"The size of tensor a {a_b1} must match the size of tensor b {b_b1} at non-singleton dimension 0")

    # get tensors
    tt_a, tt_b, torch_a, torch_b = get_tensors(input_a_shape, input_b_shape, transpose_b, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = (
        ttl.operations.primary.moreh_matmul(tt_a, tt_b, transpose_b=transpose_b)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    torch_out = torch.matmul(torch_a, torch_b)

    # compare results
    allclose_result, isclose_true_ratio = compare(tt_out, torch_out)
    assert allclose_result or isclose_true_ratio > 0.95


@pytest.mark.parametrize(
    "input_a_shape",
    (
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 11],
        [1, 1, TILE_HEIGHT * 2 - 1, TILE_WIDTH * 2],
        [1, 1, TILE_HEIGHT * 9 - 7, TILE_WIDTH * 3 - 10],
        [1, 1, TILE_HEIGHT * 18 - 17, TILE_WIDTH * 2 - 10],
    ),
)

# input_b_shape[3] is dummy
@pytest.mark.parametrize(
    "input_b_shape",
    (
        [1, 1, TILE_WIDTH, TILE_HEIGHT],
        [1, 1, TILE_WIDTH - 1, TILE_HEIGHT],
        [1, 1, TILE_WIDTH * 2 - 1, TILE_HEIGHT],
        [1, 1, TILE_WIDTH * 12 - 10, TILE_HEIGHT],
        [1, 1, TILE_WIDTH * 24 - 20, TILE_HEIGHT],
    ),
)
def test_moreh_matmul_transpose_b(input_a_shape, input_b_shape, device):
    transpose_b = True
    input_b_shape[3] = input_a_shape[3]

    # check matmul shape
    a_b1, a_b2, a_m, a_k, b_b1, b_b2, b_k, b_n = get_tensor_shape(input_a_shape, input_b_shape, transpose_b)
    output_shape = [a_b1 if a_b1 >= b_b1 else b_b1, a_b2 if a_b2 >= b_b2 else b_b2, a_m, b_n]

    if a_k != b_k:
        pytest.skip(f"k dim {a_k} and {b_k} is not the same")

    if not ((a_b2 == b_b2) or (a_b2 == 1) or (b_b2 == 1)):
        pytest.skip(f"The size of tensor a {a_b2} must match the size of tensor b {b_b2} at non-singleton dimension 1")

    if not ((a_b1 == b_b1) or (a_b1 == 1) or (b_b1 == 1)):
        pytest.skip(f"The size of tensor a {a_b1} must match the size of tensor b {b_b1} at non-singleton dimension 0")

    # get tensors
    tt_a, tt_b, torch_a, torch_b = get_tensors(input_a_shape, input_b_shape, transpose_b, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = (
        ttl.operations.primary.moreh_matmul(tt_a, tt_b, transpose_b=transpose_b)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    torch_out = torch.matmul(torch_a, torch_b)

    # compare results
    allclose_result, isclose_true_ratio = compare(tt_out, torch_out)
    assert allclose_result or isclose_true_ratio > 0.95


@pytest.mark.parametrize(
    "input_a_shape",
    (
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [1, 2, TILE_HEIGHT * 5 - 1, TILE_WIDTH],
        [2, 1, TILE_HEIGHT * 5 - 1, TILE_WIDTH - 11],
        [2, 2, TILE_HEIGHT * 5 - 7, TILE_WIDTH - 17],
        [1, 5, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 3],
        [5, 1, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 3 - 11],
        [5, 5, TILE_HEIGHT * 20 - 7, TILE_WIDTH * 3 - 1],
    ),
)

# input_b_shape[3] is dummy
@pytest.mark.parametrize(
    "input_b_shape",
    (
        [1, 1, TILE_WIDTH, TILE_HEIGHT],
        [1, 2, TILE_WIDTH * 5 - 1, TILE_HEIGHT],
        [2, 1, TILE_WIDTH * 5 - 13, TILE_HEIGHT],
        [2, 2, TILE_WIDTH * 5 - 31, TILE_HEIGHT],
        [1, 5, TILE_WIDTH * 10 - 1, TILE_HEIGHT],
        [5, 1, TILE_WIDTH * 10 - 13, TILE_HEIGHT],
        [5, 5, TILE_WIDTH * 20 - 1, TILE_HEIGHT],
    ),
)
def test_batched_moreh_matmul_transpose_b(input_a_shape, input_b_shape, device):
    transpose_b = True
    input_b_shape[3] = input_a_shape[3]

    # check matmul shape
    a_b1, a_b2, a_m, a_k, b_b1, b_b2, b_k, b_n = get_tensor_shape(input_a_shape, input_b_shape, transpose_b)
    output_shape = [a_b1 if a_b1 >= b_b1 else b_b1, a_b2 if a_b2 >= b_b2 else b_b2, a_m, b_n]

    if a_k != b_k:
        pytest.skip(f"k dim {a_k} and {b_k} is not the same")

    if not ((a_b2 == b_b2) or (a_b2 == 1) or (b_b2 == 1)):
        pytest.skip(f"The size of tensor a {a_b2} must match the size of tensor b {b_b2} at non-singleton dimension 1")

    if not ((a_b1 == b_b1) or (a_b1 == 1) or (b_b1 == 1)):
        pytest.skip(f"The size of tensor a {a_b1} must match the size of tensor b {b_b1} at non-singleton dimension 0")

    # get tensors
    tt_a, tt_b, torch_a, torch_b = get_tensors(input_a_shape, input_b_shape, transpose_b, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = (
        ttl.operations.primary.moreh_matmul(tt_a, tt_b, transpose_b=transpose_b)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    torch_out = torch.matmul(torch_a, torch_b)

    # compare results
    allclose_result, isclose_true_ratio = compare(tt_out, torch_out)
    assert allclose_result or isclose_true_ratio > 0.95


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, TILE_WIDTH],  # test single tile
        [1, 1, 1, TILE_WIDTH * 20],  # test multiple tiles
        [1, 1, 1, TILE_WIDTH * 20 - 17],  # test multiple tiles, not a multiple of 32
    ),
)
def test_moreh_matmul_1d(input_shape, device):
    transpose_b = False

    if input_shape[0] != 1 or input_shape[1] != 1 or input_shape[2] != 1:
        pytest.skip(f"dim 0, 1, 2 should be 1")

    output_shape = [1, 1, 1, TILE_WIDTH]
    # get tensors
    tt_a, tt_b, torch_a, torch_b = get_tensors(input_shape, input_shape, transpose_b, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = (
        ttl.operations.primary.moreh_matmul(tt_a, tt_b, transpose_b=transpose_b)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    torch_a = torch.reshape(torch_a, (torch_a.shape[-1],))
    torch_b = torch.reshape(torch_b, (torch_b.shape[-1],))
    torch_out = torch.matmul(torch_a, torch_b)

    # compare results
    tt_out = tt_out[0][0][0][0]
    allclose_result, isclose_true_ratio = compare(tt_out, torch_out)
    assert allclose_result or isclose_true_ratio > 0.95
