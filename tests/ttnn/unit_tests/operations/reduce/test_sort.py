# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE_WIDTH = 32


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, False),
        ([32, 128], -1, False),
        ([1, 1, 32, 64], -1, True),
        ([32, 128], 1, True),
        ([1], 0, True),
        ([], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
        ([1, 1, 512, 64], -1, False),
        ([1, 1, 2112, 64], -1, False),
    ],
)
def test_sort_standard(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_with_pcc(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, False),
        ([32, 128], -1, False),
        ([1, 1, 32, 64], -1, True),
        ([32, 128], 1, True),
        ([1], 0, True),
        ([], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
        ([1, 1, 512, 64], -1, False),
        ([1, 1, 2112, 64], -1, False),
    ],
)
def test_sort_prealocated_output(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    ttnn_sort_values = ttnn.zeros_like(ttnn_input)
    ttnn_sort_indices = ttnn.zeros_like(ttnn_input)
    ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending, out=(ttnn_sort_values, ttnn_sort_indices))

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_with_pcc(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([1, 1, 32, 96 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
        ([1, 4748 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_long_tensor(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_with_pcc(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 96], -1, True),
        ([1, 1, 32, 96 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_l1_memory_tensor(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        input,
        ttnn.bfloat16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_with_pcc(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([32, 128], -1, True),
        ([1, 1, 32, 128 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_program_cache(shape, dim, descending, device):
    torch.manual_seed(0)
    device.enable_program_cache()

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    test_iterations = 3
    for _ in range(test_iterations):
        # Run the sort operation multiple times to fill the program cache
        ttnn_sort_values, ttnn_sort_indices = ttnn.experimental.sort(ttnn_input, dim=dim, descending=descending)
        ttnn_sort_values_torch = ttnn.to_torch(ttnn_sort_values)

        assert torch_sort_values.shape == ttnn_sort_values.shape
        assert torch_sort_indices.shape == ttnn_sort_indices.shape

        assert list(ttnn_sort_values.shape) == shape
        assert list(ttnn_sort_indices.shape) == shape

        assert_with_pcc(torch_sort_values, ttnn_sort_values_torch)
        ttnn.synchronize_device(device)
    cache_entries = device.num_program_cache_entries()
    device.disable_and_clear_program_cache()
    assert cache_entries == 1, "Expected only one program cache entry for sort operation, but found {}".format(
        cache_entries
    )
