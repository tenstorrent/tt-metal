# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
import math
from tests.ttnn.utils_for_testing import assert_with_pcc


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
    assert_with_pcc(torch.cat(torch_chunks, dim=-1), torch.cat(ttnn_chunks_torch, dim=-1))
