# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "N, K, C, W",
    [
        (2, 5, 3, 4),
        (1, 1, 1, 1),
        (4, 4, 4, 4),
        (128, 64, 128, 32),
        (16, 16, 16, 16),
        (2, 256, 2, 32),
        (2, 32, 96, 32),
        (1, 2048, 1, 32),
        (64, 128, 256, 128),
        (128, 128, 128, 64),
    ],
)
def test_tosa_gather_general(N, K, C, W, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn((N, K, C), dtype=torch_dtype)
    index = torch.randint(0, K, (N, W), dtype=torch.bfloat16)

    torch_index = index.to(torch.int64)
    torch_gather = torch.gather(input, dim=1, index=torch_index.unsqueeze(-1).expand(-1, -1, C))

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.tosa_gather(ttnn_input, ttnn_index)

    assert ttnn_gather.shape == torch_gather.shape
    assert_with_pcc(torch_gather, ttnn.to_torch(ttnn_gather))
