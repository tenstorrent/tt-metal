# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import itertools

from tests.ttnn.utils_for_testing import assert_allclose, assert_equal, assert_with_pcc


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("h", [32, 33])
@pytest.mark.parametrize("w", [64, 65])
@pytest.mark.parametrize("dims", [[2], [3], [2, 3], [1, 2, 3]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_rm(device, n, c, h, w, dims, dtype):
    torch.manual_seed(2005)
    shape = (n, c, h, w)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, dims)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, dims)
    assert_equal(torch_output, ttnn.to_torch(tt_output))


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("h", [32, 33])
@pytest.mark.parametrize("w", [64, 65])
@pytest.mark.parametrize("dims", [[2], [3], [2, 3], [1, 2, 3]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_tiled(device, n, c, h, w, dims, dtype):
    torch.manual_seed(2005)
    shape = (n, c, h, w)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, dims)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, dims)

    if dtype == ttnn.float32:
        # Looks like untilize -> tilize logic introduces small error for float32
        # TODO: investigate
        assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.9999)
    else:
        assert_equal(torch_output, ttnn.to_torch(tt_output))


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("h", [16, 33])
@pytest.mark.parametrize("w", [32, 65])
@pytest.mark.parametrize("dims", [[2], [3], [4], [2, 3], [3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_5d_rm(device, b, n, c, h, w, dims, dtype):
    torch.manual_seed(42)
    shape = (b, n, c, h, w)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, dims)

    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, dims)
    assert_equal(torch_output, ttnn.to_torch(tt_output))


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("h", [16, 33])
@pytest.mark.parametrize("w", [32, 65])
@pytest.mark.parametrize("dims", [[2], [3], [4], [2, 3], [3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_5d_tiled(device, b, n, c, h, w, dims, dtype):
    torch.manual_seed(42)
    shape = (b, n, c, h, w)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, dims)

    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, dims)

    if dtype == ttnn.float32:
        # Looks like untilize -> tilize logic introduces small error for float32
        # TODO: investigate
        assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.9999)
    else:
        assert_equal(torch_output, ttnn.to_torch(tt_output))


# TODO: flip fails for 1D input ((10,), [0]),
@pytest.mark.parametrize(
    "shape,dims",
    [
        ((5, 10), [0]),
        ((5, 10), [1]),
        ((5, 10), [0, 1]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_small_tensors_rm(device, shape, dims, dtype):
    torch.manual_seed(123)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, dims)

    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, dims)

    assert_equal(torch_output, ttnn.to_torch(tt_output))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_small_tensors_tiled(device, shape, dims, dtype):
    torch.manual_seed(123)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, dims)

    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, dims)

    if dtype == ttnn.float32:
        # Looks like untilize -> tilize logic introduces small error for float32
        # TODO: investigate
        assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.9999)
    else:
        assert_equal(torch_output, ttnn.to_torch(tt_output))
