# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, channels,height, width",
    [
        [1, 3, 320, 800],
    ],
)
def test_interleaved_permute(device, batch_size, channels, height, width, min_channels=8):
    # interleaved permute
    torch_input = torch.randn((batch_size, channels, height, width), dtype=torch.bfloat16)
    channel_padding_needed = min_channels - torch_input.shape[1]
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, ttnn.L1_MEMORY_CONFIG)
    ttnn_out = ttnn.pad(ttnn_input, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
    ttnn_out = ttnn.permute(ttnn_out, (0, 2, 3, 1))
    # sharded permute


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, channels,height, width",
    [
        [1, 3, 320, 800],
    ],
)
def test_sharded_permute(device, batch_size, channels, height, width, min_channels=8):
    torch_input = torch.randn((batch_size, channels, height, width), dtype=torch.bfloat16)
    channel_padding_needed = min_channels - torch_input.shape[1]
    ttnn_input_2 = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_mem_config = ttnn.create_sharded_memory_config(
        [batch_size, min_channels, height, width],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_2 = ttnn_input_2.to(device, input_mem_config)
    ttnn_out_padd = ttnn.pad(ttnn_input_2, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
    ttnn.deallocate(ttnn_input_2)
    ttnn_out_2 = ttnn.permute(ttnn_out_padd, (0, 2, 3, 1))
    ttnn.deallocate(ttnn_out_padd)
    ttnn.deallocate(ttnn_out_2)
