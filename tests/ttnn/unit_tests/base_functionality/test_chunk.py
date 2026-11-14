# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
import math
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "shape, num_chunks, dim",
    [
        ((50,), 7, 0),
        ((200,), 1, 0),
        ((4, 3), 5, 0),
        ((2, 4, 3), 2, 0),
        ((2, 4, 3), 2, 1),
        ((2, 3, 4, 2), 4, 2),
        ((2, 2, 2), 2, 0),
        ((10, 1, 1), 5, 0),
        ((2, 3, 4, 4, 2), 5, 3),
        ((1, 2, 2, 2, 2, 2), 3, 2),
        ((1, 2, 2, 2, 2, 2), 5, 4),
        ((1, 1, 1, 4, 2, 2), 5, 2),
        ((1, 1, 1, 2, 2, 3), 5, 3),
        ((1, 1, 1, 3, 2, 4), 5, 4),
        ((1, 1, 1, 3, 2, 4), 4, 2),
        ((1, 1, 1, 3, 2, 4), 6, 1),
        ((1, 1, 1, 3, 2, 4), 3, 5),
        ((1, 1, 1, 3, 2, 4), 7, 0),
        ((1, 1, 1, 4, 2, 4), 5, 4),
        ((1, 1, 1, 3, 2, 4), 8, 1),
    ],
)
def test_chunking(device, shape, num_chunks, dim):
    numel = math.prod(shape)
    tensor = torch.arange(numel, dtype=torch.float32).reshape(shape)
    ttnn_tensor = ttnn.from_torch(tensor)

    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    torch_chunks = torch.chunk(tensor, num_chunks, dim)
    ttnn_chunks = ttnn.chunk(ttnn_tensor, num_chunks, dim)

    ttnn_chunks_torch = [ttnn.to_torch(chunk) for chunk in ttnn_chunks]
    assert_equal(torch.cat(torch_chunks, dim=-1), torch.cat(ttnn_chunks_torch, dim=-1))


@pytest.mark.parametrize(
    "shape, num_chunks, dim",
    [
        ((50,), 7, -1),
        ((2, 4, 3), 2, -3),
        ((2, 3, 4, 2), 4, -2),
        ((2, 3, 4, 4, 2), 5, -5),
        ((1, 1, 1, 3, 2, 4), 3, -6),
    ],
)
def test_chunking_negative_dim(device, shape, num_chunks, dim):
    numel = math.prod(shape)
    tensor = torch.arange(numel, dtype=torch.float32).reshape(shape)
    ttnn_tensor = ttnn.to_device(ttnn.from_torch(tensor), device, memory_config=ttnn.L1_MEMORY_CONFIG)

    torch_chunks = torch.chunk(tensor, num_chunks, dim)
    ttnn_chunks = ttnn.chunk(ttnn_tensor, num_chunks, dim)

    assert len(ttnn_chunks) == len(torch_chunks)
    for torch_chunk, ttnn_chunk in zip(torch_chunks, ttnn_chunks):
        assert_equal(torch_chunk, ttnn.to_torch(ttnn_chunk))


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((50,), -2),
        ((4, 3), -3),
        ((4, 32, 32), -4),
        ((2, 4, 32, 32), -5),
        ((2, 4, 32, 32), -6),
        ((2, 4, 32, 32), 4),
    ],
)
def test_chunking_dim_out_of_range(device, shape, dim, expect_error):
    # torch.chunk raises for a dim outside [-rank, rank - 1]; ttnn.chunk must not wrap it into a slice index.
    tensor = torch.zeros(shape, dtype=torch.float32)
    ttnn_tensor = ttnn.to_device(ttnn.from_torch(tensor), device, memory_config=ttnn.L1_MEMORY_CONFIG)

    with expect_error(IndexError, "Dimension out of range"):
        torch.chunk(tensor, 2, dim)
    with expect_error(RuntimeError, "out of range"):
        ttnn.chunk(ttnn_tensor, 2, dim)
