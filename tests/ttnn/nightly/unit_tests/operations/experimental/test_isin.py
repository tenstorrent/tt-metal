# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def select_torch_dtype(ttnn_dtype):
    """
    Convert a ttnn dtype to the corresponding torch dtype.

    :param ttnn_dtype: ttnn dtype to be converted
    :return: Corresponding torch dtype
    """
    if ttnn_dtype == ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype == ttnn.float32:
        return torch.float32
    if ttnn_dtype == ttnn.uint8:
        return torch.uint8
    if ttnn_dtype == ttnn.uint16:
        return torch.int64
    if ttnn_dtype == ttnn.int32:
        return torch.int64
    if ttnn_dtype == ttnn.uint32:
        return torch.int64  # PyTorch requires int64 for index tensors
    raise TypeError(f"Unsupported ttnn dtype: {ttnn_dtype}")


@pytest.mark.parametrize(
    "elements, test_elements, dtype, layout, invert",
    [
        ([i for i in range(100)], [2, 3, 1, 500], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False),
        ([i for i in range(200)], [-100, 200, 300], ttnn.int32, ttnn.TILE_LAYOUT, True),
        (
            [i for i in range(10, 200)],
            [11, 2, 3, 24, 20, 10, 200, 199],
            ttnn.uint16,
            ttnn.ROW_MAJOR_LAYOUT,
            True,
        ),
        (
            [[[i ^ j ^ k for i in range(0, 10)] for j in range(0, 10)] for k in range(0, 10)],
            [28 * i for i in range(0, 20)],
            ttnn.int32,
            ttnn.TILE_LAYOUT,
            False,
        ),
    ],
)
def test_isin_typical_predefined_data(elements, test_elements, dtype, layout, invert, device):
    # Arrange - Prepare data
    torch_dtype = select_torch_dtype(dtype)
    elements_torch = torch.tensor(elements, dtype=torch_dtype)
    test_elements_torch = torch.tensor(test_elements, dtype=torch_dtype)

    # Convert to ttnn tensors
    elements_ttnn = ttnn.from_torch(elements_torch, device=device, layout=layout)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, layout=layout)

    # Act - Compute results
    torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
    ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    # Assert - Compare results
    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)


@pytest.mark.parametrize(
    "elements_shape, test_elements_shape, invert",
    [
        ([10, 10], [20, 20], False),
        ([32], [32], False),
        ([5, 10, 50], [4, 10], True),
        ([2, 2, 2, 2, 2], [1, 2, 2, 1], False),
        ([3, 2, 3, 2, 3, 2, 3], [1, 1, 10, 2, 1], True),
        ([1, 1, 80000], [10], False),
        ([5, 10, 5, 1, 1, 1, 1, 1, 1, 5], [20], True),
    ],
)
def test_isin_random_data(elements_shape, test_elements_shape, invert, device):
    torch.manual_seed(0)

    # Arrange - Prepare data
    elements_torch = torch.randint(0, 10000, elements_shape, dtype=torch.int64)
    test_elements_torch = torch.randint(0, 10000, test_elements_shape, dtype=torch.int64)

    elements_ttnn = ttnn.from_torch(elements_torch, device=device, dtype=ttnn.int32)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, dtype=ttnn.int32)

    # Act - Compute results
    torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
    ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    # Assert - Compare results
    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)


@pytest.mark.parametrize(
    "elements_shape, test_elements_shape, invert, expected_num_program_cache_entries",
    [
        ([10], [20], False, 1),
        ([20], [10], True, 1),
        ([10, 10], [20, 20], False, 4),
        ([5, 10, 5, 1, 1, 1, 1, 1, 1, 5], [20], True, 3),
    ],
)
def test_isin_program_cache_and_random_data(
    elements_shape, test_elements_shape, invert, expected_num_program_cache_entries, device
):
    torch.manual_seed(0)

    # Arrange - Prepare data
    elements_torch = torch.randint(0, 10000, elements_shape, dtype=torch.int64)
    test_elements_torch = torch.randint(0, 10000, test_elements_shape, dtype=torch.int64)

    # Convert to ttnn tensors
    elements_ttnn = ttnn.from_torch(elements_torch, device=device, dtype=ttnn.int32)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, dtype=ttnn.int32)

    # Act - Compute results multiple times to test program cache
    for _ in range(2):
        torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
        ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    # Assert - Compare results
    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)
    assert device.num_program_cache_entries() == expected_num_program_cache_entries
