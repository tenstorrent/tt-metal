# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger


import ttnn
from models.utility_functions import comp_pcc, is_wormhole_b0, is_blackhole
import torch
import ttnn

shapes = [
    [1, 1, 25088, 64],
]


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="disabled due to watcher error, see issue #5863")
@pytest.mark.parametrize("shape", shapes)
def test_move_op(shape, device):
    run_move_op(shape, device)


def run_move_op(shape, device):
    """
    For non_overlap, multi-core is run for num_tiles > 1.
    """
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    if (compute_grid_size.x * compute_grid_size.y) < 98:
        core_count = 25
        shape[2] = 25050
    else:
        core_count = 98

    dtype = ttnn.bfloat16
    layout = ttnn.ROW_MAJOR_LAYOUT
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    if (compute_grid_size.x * compute_grid_size.y) < 98:
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        )
    else:
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(11, 7),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 8),
                    ttnn.CoreCoord(1, 8),
                ),
            }
        )
    assert shape[0] == 1 and shape[1] == 1
    assert shape[2] % core_count == 0 and shape[3] % 32 == 0
    shard_shape = [(int)(shape[2] / core_count), shape[3]]
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        shard_shape,
        shard_orientation,
        False,
    )
    # make dummy shape half of shape, so we will test move sharded with overlap
    dummy_shape = [shape[0], shape[1], (int)(shape[2] / 2), shape[3]]
    dummy_shard_shape = [(int)(dummy_shape[2] / core_count), dummy_shape[3]]
    dummy_shard_spec = ttnn.ShardSpec(
        shard_grid,
        dummy_shard_shape,
        shard_orientation,
        False,
    )
    dummy_tensor = torch.zeros(dummy_shape)
    tt_dummy_tensor = ttnn.Tensor(dummy_tensor, dtype)
    dummy_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=dummy_shard_spec,
    )
    tt_dummy_tensor = tt_dummy_tensor.to(device, dummy_mem_config)
    logger.info(f"shape={shape}")
    input_volume = shape[2] * shape[3]
    tensor = []
    for val in range(1, input_volume + 1):
        tensor.append(val)
    torch_tensor = torch.tensor(tensor).reshape(shape)
    tt_tensor = ttnn.Tensor(torch_tensor, dtype)
    height_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_tensor = tt_tensor.to(device, height_sharded_mem_config)

    # Free up dummy tensor from memory to make available to move
    tt_dummy_tensor.deallocate()

    output = ttnn.move(tt_tensor, memory_config=height_sharded_mem_config)

    tt_host_rm = output.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    pyt_got_back_rm = tt_host_rm.to_torch()

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm, torch_tensor, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc
