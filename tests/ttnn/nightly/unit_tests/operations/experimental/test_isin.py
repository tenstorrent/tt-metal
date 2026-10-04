# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

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
    elements_ttnn = ttnn.from_torch(elements_torch, device=device, layout=layout, dtype=dtype)
    test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, layout=layout, dtype=dtype)

    # Act - Compute results
    torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
    ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)

    # Assert - Compare results
    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result).to(torch_isin_result.dtype)
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
    torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result).to(torch_isin_result.dtype)
    assert torch_isin_result.shape == torch_result_from_ttnn.shape
    assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
    assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)


def cache_hit_isin_inputs(elements_shape, test_elements_shape):
    """Return two same-spec calls whose membership masks expose a stale binding.

    The signal is the first three elements. Every other element is 0, and test-element
    padding is 99, so neither value is in the other tensor's signal set. Membership of
    those three positions is:

        call 0 elements [10, 11, 12] vs {10, 20, 12} -> True, False, True
        call 1 elements [20, 21, 22] vs {11, 21, 22} -> False, True, True
        stale elements (call-0 elements, call-1 test) -> False, True, False
        stale test elements (call-1 elements, call-0 test) -> True, False, False

    Both buffers stale reproduces call 0. invert flips every mask and keeps them distinct.
    """
    assert math.prod(elements_shape) >= 3
    assert math.prod(test_elements_shape) >= 3

    elements_0 = torch.zeros(elements_shape, dtype=torch.int64)
    elements_1 = torch.zeros(elements_shape, dtype=torch.int64)
    test_elements_0 = torch.full(test_elements_shape, 99, dtype=torch.int64)
    test_elements_1 = torch.full(test_elements_shape, 99, dtype=torch.int64)
    elements_0.view(-1)[:3] = torch.tensor([10, 11, 12], dtype=torch.int64)
    elements_1.view(-1)[:3] = torch.tensor([20, 21, 22], dtype=torch.int64)
    test_elements_0.view(-1)[:3] = torch.tensor([10, 20, 12], dtype=torch.int64)
    test_elements_1.view(-1)[:3] = torch.tensor([11, 21, 22], dtype=torch.int64)
    return elements_0, test_elements_0, elements_1, test_elements_1


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
    elements_0, test_elements_0, elements_1, test_elements_1 = cache_hit_isin_inputs(
        elements_shape, test_elements_shape
    )
    # A stale elements buffer, a stale test-elements buffer, and both stale together
    # each disagree with the second call. The device comparison below then fails
    # unless both reader bindings were refreshed.
    outcome_masks = (
        torch.isin(elements_0, test_elements_0, invert=invert),
        torch.isin(elements_1, test_elements_1, invert=invert),
        torch.isin(elements_0, test_elements_1, invert=invert),
        torch.isin(elements_1, test_elements_0, invert=invert),
    )
    for left_index, left_mask in enumerate(outcome_masks):
        for right_mask in outcome_masks[left_index + 1 :]:
            assert not torch.equal(left_mask, right_mask)

    # Keep both iterations' tensors alive so the second call gets a genuinely
    # different buffer address instead of reusing a freed one.
    kept_alive_tensors = []
    for elements_torch, test_elements_torch in (
        (elements_0, test_elements_0),
        (elements_1, test_elements_1),
    ):
        elements_ttnn = ttnn.from_torch(elements_torch, device=device, dtype=ttnn.int32)
        test_elements_ttnn = ttnn.from_torch(test_elements_torch, device=device, dtype=ttnn.int32)
        torch_isin_result = torch.isin(elements_torch, test_elements_torch, invert=invert)
        ttnn_isin_result = ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert)
        kept_alive_tensors.append((elements_ttnn, test_elements_ttnn, ttnn_isin_result))

        torch_result_from_ttnn = ttnn.to_torch(ttnn_isin_result).to(torch_isin_result.dtype)
        assert torch_isin_result.shape == torch_result_from_ttnn.shape
        assert torch_isin_result.count_nonzero() == torch_result_from_ttnn.count_nonzero()
        assert torch.equal(torch_isin_result != 0, torch_result_from_ttnn != 0)
    assert len(kept_alive_tensors) == 2
    assert device.num_program_cache_entries() == expected_num_program_cache_entries
