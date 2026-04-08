# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, comp_pcc


def test_s2i_dram_height_sharded(device):
    torch_weight_tensor = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    torch_input_tensor = torch.rand([1, 1, 320, 32], dtype=torch.bfloat16)

    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7)),
        }
    )
    input_shard_shape = (32, 32)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=input_memory_config
    )

    # Allocate a dummy tensor so we can put weight after in DRAM
    output_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )  # Alternatively: ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
    weight_tensor = ttnn.from_torch(
        torch_weight_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )  # This will be allocated after output tensor in DRAM

    # Deallocated output to create slot and then reshard from L1 again
    ttnn.deallocate(output_tensor)
    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    output_passed, output_message = comp_pcc(torch_input_tensor, ttnn.to_torch(output_tensor), 1.0)
    weight_passed, weight_message = comp_pcc(torch_weight_tensor, ttnn.to_torch(weight_tensor), 1.0)

    assert (
        output_passed and weight_passed
    ), f"Output result: (passed?={output_passed}, pcc={round(output_message, 4)}) - Weight result: (passed?={weight_passed}, pcc={round(weight_message, 4)})"
