# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "shape",
    [
        [3, 4, 5, 32],  # multiple of 32
        [41, 21, 33, 34],  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        2.5,
        1.72,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
def test_index_fill_float(shape, dim, value, dtype, device):
    torch.manual_seed(2024)

    torch_input = torch.rand(shape, dtype=dtype)
    torch_index = torch.tensor([0, 2])
    torch_output = torch.index_fill(torch_input, dim, torch_index, value)

    input = ttnn.from_torch(torch_input, device=device)
    input = ttnn.to_device(input, device)

    index = ttnn.from_torch(torch_index, device=device)
    index = ttnn.to_device(index, device)

    ttnn_output = ttnn.index_fill(input, dim, index, value)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert assert_equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "shape",
    [
        [6, 3, 7, 32],  # multiple of 32
        [13, 15, 22, 13],  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        15,
        12,
    ],
)
def test_index_fill_int(shape, dim, value, device):
    torch.manual_seed(2024)

    torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    torch_index = torch.tensor([0, 2])
    torch_output = torch.index_fill(torch_input, dim, torch_index, value)

    input = ttnn.from_torch(torch_input, device=device)
    input = ttnn.to_device(input, device)

    index = ttnn.from_torch(torch_index, device=device)
    index = ttnn.to_device(index, device)

    ttnn_output = ttnn.index_fill(input, dim, index, value)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert assert_equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "shape",
    [
        [4, 3, 7, 32],  # multiple of 32
        [13, 15, 22, 13],  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        0,
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        2002,
    ],
)
def test_index_fill_callback(shape, dim, value, device, use_program_cache):
    torch.manual_seed(2024)
    for i in range(2):
        torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
        torch_index = torch.tensor([0, 2])
        torch_output = torch.index_fill(torch_input, dim, torch_index, value)

        input = ttnn.from_torch(torch_input, device=device)
        input = ttnn.to_device(input, device)

        index = ttnn.from_torch(torch_index, device=device)
        index = ttnn.to_device(index, device)

        ttnn_output = ttnn.index_fill(input, dim, index, value)
        ttnn_output = ttnn.from_device(ttnn_output)
        ttnn_output = ttnn.to_torch(ttnn_output)

        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)

    assert assert_equal(ttnn_output, torch_output)
