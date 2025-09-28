# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose


def select_torch_dtype(ttnn_dtype):
    if ttnn_dtype is ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype is ttnn.float32:
        return torch.float32
    if ttnn_dtype is ttnn.uint8:
        return torch.uint8
    if ttnn_dtype is ttnn.uint16:
        return torch.int64
    if ttnn_dtype is ttnn.int32:
        return torch.int64
    if ttnn_dtype is ttnn.uint32:
        return (
            torch.int64
        )  # !!! there is a strict requirement for the index tensor in Torch to be int64, and there is no int64 in ttnn


@pytest.mark.parametrize(
    "elements, test_elements, dtype, layout, invert",
    [
        # ([i for i in range(100)], [2, 3, 1, 500], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False),
        # ([i for i in range(200)], [-100, 200, 300], ttnn.int32, ttnn.TILE_LAYOUT, True),
        # (
        #     [i for i in range(10, 200)],
        #     [11, 2, 3, 24, 20, 10, 200, 199],
        #     ttnn.uint16,
        #     ttnn.ROW_MAJOR_LAYOUT,
        #     False,
        # ),
        # (
        #     [[i for i in range(10, 20)] for _ in range(5)],
        #     [11, 2, 3, 24, 20, 10, 200, 199],
        #     ttnn.uint32,
        #     ttnn.TILE_LAYOUT,
        #     True,
        # ),
        # (
        #     [[[i ^ j ^ k for i in range(0, 10)] for j in range(0, 10)] for k in range(0, 10)],
        #     [28 * i for i in range(0, 20)],
        #     ttnn.int32,
        #     ttnn.TILE_LAYOUT,
        #     False,
        # ),
        # (
        #     [
        #         [[[i ^ j ^ k ^ l for i in range(0, 10)] for j in range(0, 10)] for k in range(0, 10)]
        #         for l in range(0, 10)
        #     ],
        #     [28 * i for i in range(0, 20)],
        #     ttnn.int32,
        #     ttnn.TILE_LAYOUT,
        #     True,
        # ),
    ],
)
def test_isin_typical_predefined_data(elements, test_elements, dtype, layout, invert, device):
    torch_dtype = select_torch_dtype(dtype)
    elements_torch = torch.tensor(elements, dtype=torch_dtype)
    test_elements_torch = torch.tensor(test_elements, dtype=torch_dtype)

    elements_ttnn = ttnn.from_torch(elements_torch, device=device, layout=layout)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, layout=layout)

    torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
    ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)


@pytest.mark.parametrize(
    "elements_shape, test_elements_shape, invert",
    [
        # ([10], [20], False),
        # ([20], [10], True),
        # ([10, 10], [20, 20], False),
        # ([20, 10], [10, 20], True),
        # ([32], [32], False),
        # ([5, 10, 50], [4, 10], True),
        # ([2, 2, 2, 2, 2], [1, 2, 2, 1], False),
        # ([3, 2, 3, 2, 3, 2, 3], [1, 1, 10, 2, 1], True),
        ([100, 1000], [20, 10, 5, 2, 5], False),
        # ([1, 1, 80000], [10], True),
        # ([1, 1, 20000, 1, 1], [100], False),
        # ([1, 10000, 1], [100], True),
        # ([20000, 2, 1], [1, 100, 1], False),
        # ([100000, 1, 1], [100], True),
        # ([10, 10, 10, 10, 10], [20, 2], False),
        # ([10, 10, 2, 10, 2], [1, 20, 20, 1, 20, 20], True),
        # ([10, 10, 2, 5, 50], [20, 10, 20, 1, 10, 1], False),
        # ([5, 10, 5, 1, 1, 1, 1, 1, 1, 5], [20], True),
    ],
)
def test_isin_random_data(elements_shape, test_elements_shape, invert, device):
    torch.manual_seed(0)

    elements_torch = torch.randint(0, 10000, elements_shape, dtype=torch.int64)
    test_elements_torch = torch.randint(0, 10000, test_elements_shape, dtype=torch.int64)

    elements_ttnn = ttnn.from_torch(elements_torch, device=device, dtype=ttnn.int32)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, dtype=ttnn.int32)
    ttnn.set_printoptions(profile="full")

    torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
    ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)


@pytest.mark.parametrize(
    "elements_shape, test_elements_shape, invert, expected_num_program_cache_entries",
    [
        # ([10], [20], False, 1),
        # ([20], [10], True, 1),
        # ([10, 10], [20, 20], False, 4),
        # ([5, 10, 5, 1, 1, 1, 1, 1, 1, 5], [20], True, 3),
    ],
)
def test_isin_program_cache_and_random_data(
    elements_shape, test_elements_shape, invert, expected_num_program_cache_entries, device
):
    torch.manual_seed(0)

    elements_torch = torch.randint(0, 10000, elements_shape, dtype=torch.int64)
    test_elements_torch = torch.randint(0, 10000, test_elements_shape, dtype=torch.int64)

    elements_ttnn = ttnn.from_torch(elements_torch, device=device, dtype=ttnn.int32)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, dtype=ttnn.int32)

    for _ in range(2):
        torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
        ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)
    assert device.num_program_cache_entries() == expected_num_program_cache_entries
