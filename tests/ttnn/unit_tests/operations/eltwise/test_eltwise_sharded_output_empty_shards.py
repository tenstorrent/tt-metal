# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

SHARDED_OUTPUTS = {
    "height-2_cores": (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 0), [32, 256]),
    "height-4_cores": (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (3, 0), [32, 256]),
    "width-4_cores": (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (3, 0), [64, 64]),
    "width-8_cores": (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (7, 0), [64, 64]),
    "block-2x2_cores": (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 1), [32, 128]),
    "block-3x2_cores": (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (2, 1), [32, 128]),
}


def _sharded_output(name):
    layout, end, shard_shape = SHARDED_OUTPUTS[name]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*end))})
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize("output", list(SHARDED_OUTPUTS))
@pytest.mark.parametrize("op", ["add", "multiply", "where", "addcmul"])
def test_l1_interleaved_inputs_sharded_output(device, op, output):
    torch.manual_seed(0)
    torch_inputs = [torch.randn(1, 1, 64, 256, dtype=torch.bfloat16) for _ in range(3)]
    torch_inputs[0] = (torch_inputs[0] > 0).bfloat16() if op == "where" else torch_inputs[0]
    inputs = [
        ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        for t in torch_inputs
    ]
    memory_config = _sharded_output(output)
    a, b, c = (t.float() for t in torch_inputs)
    if op == "add":
        output, expected = ttnn.add(inputs[0], inputs[1], memory_config=memory_config), a + b
    elif op == "multiply":
        output, expected = ttnn.multiply(inputs[0], inputs[1], memory_config=memory_config), a * b
    elif op == "where":
        output, expected = ttnn.where(inputs[0], inputs[1], inputs[2], memory_config=memory_config), torch.where(
            a > 0, b, c
        )
    else:
        output = ttnn.addcmul(inputs[0], inputs[1], inputs[2], value=0.5, memory_config=memory_config)
        expected = a + 0.5 * b * c
    assert_with_pcc(expected, ttnn.to_torch(output), 0.999)
