# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("sequence", [1, 2])
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64, 32])
def test_concatenate_heads(device, batch, sequence, height, width):
    torch_input_tensor = torch.rand((batch, sequence, height, width), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.transformer.concatenate_heads(input_tensor)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


def test_concatenate_heads_sharded_input_interleaved_output(device):
    """Regression test for hang when input is height-sharded L1 and output is DRAM interleaved.
    See https://github.com/tenstorrent/tt-metal/issues/40925
    """
    torch_input_tensor = torch.randn(32, 32, 18, 64, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    input_tensor = ttnn.to_memory_config(
        input_tensor,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [512, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    output = ttnn.transformer.concatenate_heads(
        input_tensor,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
