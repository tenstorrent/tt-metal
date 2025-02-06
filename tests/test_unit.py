# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test(device, reset_seeds):
    cache_position = torch.normal(0.0, 30.0, size=(1, 256))
    cache_position = cache_position.abs()
    cache_position = cache_position.to(torch.int64)
    cache_position = cache_position[0]

    ttnn_cache_position = ttnn.from_torch(
        cache_position,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(
        "ttnn_cache_position", ttnn_cache_position
    )  # ttnn.Tensor([   31,    27,  ...,    39,    35], shape=Shape([256]), dtype=DataType::UINT32, layout=Layout::TILE)
    # Even tried with ttnn_cache_position = ttnn.unsqueeze(ttnn_cache_position,1) instead of next two steps, but same value mismatch
    ttnn_cache_position = ttnn.unsqueeze(ttnn_cache_position, 0)
    ttnn_cache_position = ttnn.permute(ttnn_cache_position, (1, 0))
    print(
        "ttnn_cache_position", ttnn_cache_position
    )  # ttnn.Tensor([[260046878],[    0], ..., [125829135], [    0]], shape=Shape([256, 1]), dtype=DataType::UINT32, layout=Layout::TILE)
