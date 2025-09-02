# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "resolution",
    [(400), (1600), (6400)],
)
def test_softmax(device, resolution):
    print("resolution", resolution)
    torch_input_a = torch.randn(1, 17, 4, resolution)

    # Define core ranges (8 cores in a 2x4 grid)
    core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    # Create a tensor specification with height sharding
    tensor_spec = ttnn.TensorSpec(
        shape=(1, 17, 4, resolution),  # Batch=2, Height=128, Width=256
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.L1,
    ).block_sharded(core_ranges)

    # Create tensor from PyTorch tensor (using shape from spec)
    tt_tensor = ttnn.from_torch(torch_input_a, spec=tensor_spec, device=device)

    ttnn_output = ttnn.softmax(tt_tensor, dim=1)
