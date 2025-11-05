# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_equal


def run_index_fill_test(shape, dim, indices, value, dtype, device):
    torch.manual_seed(2025)

    if dim > len(shape) - 1:
        pytest.skip("Given dim is higher than tensor rank")

    if dtype == torch.int32:
        torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(shape, dtype=dtype)
    torch_index = torch.tensor(indices)
    torch_output = torch.index_fill(torch_input, dim, torch_index, value)

    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_index = ttnn.from_torch(torch_index, device=device)
    tt_output = ttnn.index_fill(tt_input, dim, tt_index, value)
    tt_output = ttnn.to_torch(tt_output)

    assert_equal(tt_output, torch_output)


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # 2D, multiple of 32
        [12, 24],  # 2D, not multiple of 32
        [23, 41, 32],  # 3D, multiple of 32
        [9, 5, 38],  # 3D, not multiple of 32
        [3, 4, 5, 32],  # 4D, multiple of 32
        [41, 21, 33, 34],  # 4D, not multiple of 32
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("indices", [[0, 2]])
@pytest.mark.parametrize("value", [2.5, 1.72])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_index_fill_float(shape, dim, indices, value, dtype, device):
    run_index_fill_test(shape, dim, indices, value, dtype, device)


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # 2D, multiple of 32
        [12, 23],  # 2D, not multiple of 32
        [27, 12, 32],  # 3D, multiple of 32
        [61, 3, 6],  # 3D, not multiple of 32
        [6, 3, 7, 32],  # 4D, multiple of 32
        [13, 15, 22, 13],  # 4D, not multiple of 32
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("indices", [[0, 2]])
@pytest.mark.parametrize("value", [15, 12])
@pytest.mark.parametrize("dtype", [torch.int32])
def test_index_fill_int(shape, dim, indices, value, dtype, device):
    run_index_fill_test(shape, dim, indices, value, dtype, device)


@pytest.mark.parametrize(
    "shape, dim, indices",
    [
        ([1], 0, [0]),  # 1 element tensor
        ([1, 1], 0, [0]),  # 1 element tensor
        ([4, 32], 1, list(range(14))),  # large index tensor
    ],
)
@pytest.mark.parametrize("value", [15])
@pytest.mark.parametrize("dtype", [torch.int32])
def test_index_fill_cornercases(shape, dim, indices, value, dtype, device):
    run_index_fill_test(shape, dim, indices, value, dtype, device)


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # 2D, multiple of 32
        [12, 23],  # 2D, not multiple of 32
        [27, 12, 32],  # 3D, multiple of 32
        [61, 3, 6],  # 3D, not multiple of 32
        [4, 3, 7, 32],  # 4D, multiple of 32
        [13, 15, 22, 13],  # 4D, not multiple of 32
    ],
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("indices", [[0, 2]])
@pytest.mark.parametrize("value", [2002])
def test_index_fill_callback(shape, dim, indices, value, device):
    for i in range(2):
        run_index_fill_test(shape, dim, indices, value, torch.int32, device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
