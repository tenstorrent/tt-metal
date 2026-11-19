# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# A height-sharded output must fold every leading dim into its height when sizing the last shard;
# dim -6 was left out, so the last tile row of a rank-6 output was never written (#64827).

import pytest
import torch
import ttnn

SHARDINGS = {
    "1_core": ((0, 0), (2, 1, 1, 1, 32, 32)),
    "2_cores": ((1, 0), (4, 1, 1, 1, 32, 32)),
}


def _height_sharded(end):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*end))})
    shard_spec = ttnn.ShardSpec(grid, [64, 32], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize("rank", [6, 4])
@pytest.mark.parametrize("sharding", list(SHARDINGS))
@pytest.mark.parametrize("op", ["relu", "where", "multiply"])
def test_height_sharded_output_last_shard_is_written(device, op, sharding, rank):
    end, shape = SHARDINGS[sharding]
    if rank == 4:
        shape = (1, 1, shape[0] * shape[-2], shape[-1])
    memory_config = _height_sharded(end)

    def to_device(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    torch.manual_seed(0)
    output = to_device(torch.full(shape, -99.0))
    if op == "relu":
        x = torch.randn(shape).bfloat16()
        ttnn.relu(to_device(x), output_tensor=output)
        expected = torch.relu(x)
    elif op == "where":
        predicate, true_value, false_value = (torch.full(shape, v) for v in (1.0, 3.0, 7.0))
        ttnn.where(to_device(predicate), to_device(true_value), to_device(false_value), output_tensor=output)
        expected = true_value.bfloat16()
    else:
        x = torch.ones(shape)
        ttnn.multiply(to_device(x), 2.0, output_tensor=output)
        expected = (x * 2).bfloat16()

    actual = ttnn.to_torch(output)
    wrong = int((actual != expected).sum())
    assert wrong == 0, f"{wrong} of {actual.numel()} output values were not written correctly"
