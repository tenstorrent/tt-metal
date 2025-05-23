# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_equal


def run_index_fill_test(shape, dim, value, dtype, device):
    if len(shape) - 1 < dim:
        pytest.skip("Given dim is higher than tensor rank")

    if dtype == torch.int32:
        torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(shape, dtype=dtype)
    torch_index = torch.tensor([0, 2])
    torch_output = torch.index_fill(torch_input, dim, torch_index, value)

    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_index = ttnn.from_torch(torch_index, device=device)

    ttnn_output = ttnn.index_fill(tt_input, dim, tt_index, value)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert assert_equal(ttnn_output, torch_output)


@pytest.mark.skip("Test case failing assert_equal() - see #22482")
@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # multiple of 32
        [12, 24],  # not multiple of 32
        [23, 41, 32],  # multiple of 32
        [9, 5, 38],  # not multiple of 32
        [3, 4, 5, 32],  # multiple of 32
        [41, 21, 33, 34],  # not multiple of 32,
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

    run_index_fill_test(shape, dim, value, dtype, device)


@pytest.mark.skip("Test case failing assert_equal() - see #22482")
@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # multiple of 32
        [12, 23],  # not multiple of 32
        [27, 12, 32],  # multiple of 32
        [61, 3, 6],  # not multiple of 32
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

    run_index_fill_test(shape, dim, value, torch.int32, device)


@pytest.mark.skip("Test case failing assert_equal() - see #22482")
@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # multiple of 32
        [12, 23],  # not multiple of 32
        [27, 12, 32],  # multiple of 32
        [61, 3, 6],  # not multiple of 32
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
        run_index_fill_test(shape, dim, value, torch.int32, device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)
