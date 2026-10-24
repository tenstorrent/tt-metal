# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "shape,cores",
    [
        ((2, 7, 65), 2),  # One padded matrix per core.
        ((3, 7, 65), 3),
        ((4, 7, 65), 2),  # Two matrices per core.
        ((2, 40, 31), 4),  # Each matrix spans two cores.
        ((3, 40, 65), 2),  # A core boundary splits the middle matrix.
        ((2, 32, 64), 2),  # No interior padding.
        ((1, 64, 64), 2),  # A single aligned matrix split across cores.
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.bfloat16])
def test_untilize_height_sharded_batch_rows(device, shape, cores, dtype):
    torch.manual_seed(0)
    values = torch.randint(-32, 32, shape, dtype=torch.int32)
    if dtype == ttnn.bfloat16:
        values = values.to(torch.bfloat16)
    padded = [shape[0], (shape[1] + 31) // 32 * 32, (shape[2] + 31) // 32 * 32]
    memory = ttnn.create_sharded_memory_config(
        padded,
        core_grid=ttnn.CoreGrid(y=cores, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tensor = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
    assert torch.equal(ttnn.to_torch(tensor), values)
    control = ttnn.to_layout(ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(control), values)
    result = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert torch.equal(ttnn.to_torch(result), values)
