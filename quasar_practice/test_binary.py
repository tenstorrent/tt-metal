# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.binary_practice_quasar against torch, for 2D tensors of the same shape."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# op name -> torch golden. Only add exists for now; sub, mul, ... get a row here as the op grows.
GOLDEN = {
    "add": torch.add,
}


@pytest.mark.parametrize("op", GOLDEN.keys())
@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),  # one tile, one node
        (64, 128),  # 8 tiles, one per node
        (32, 32 * 40),  # 40 tiles: more tiles than nodes, uneven split
        (30, 50),  # not a multiple of 32: padded to 32x64
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_binary(device, op, shape, memory_config):
    torch.manual_seed(0)
    a = torch.randn(shape)
    b = torch.randn(shape)

    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    out = ttnn.to_torch(ttnn.binary_practice_quasar(tt_a, tt_b))

    assert out.shape == a.shape
    assert_with_pcc(GOLDEN[op](a, b), out.float(), pcc=0.999)


@pytest.mark.parametrize("op", GOLDEN.keys())
@pytest.mark.parametrize(
    "shape, num_nodes",
    [
        ((256, 64), 8),  # 8 shards of 32x64 = 2 tiles each
        ((1024, 96), 32),  # every node of the chip, 3 tiles each
    ],
)
def test_binary_height_sharded(device, op, shape, num_nodes):
    torch.manual_seed(0)
    a = torch.randn(shape)
    b = torch.randn(shape)

    # Height sharding: rows split evenly across num_nodes nodes, each node holds its shard in its own L1.
    nodes = ttnn.num_cores_to_corerangeset(num_nodes, device.compute_with_storage_grid_size(), row_wise=True)
    print("GGG nodes", nodes)
    memory_config = ttnn.create_sharded_memory_config(
        shape=(shape[0] // num_nodes, shape[1]),
        core_grid=nodes,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    tt_out = ttnn.binary_practice_quasar(tt_a, tt_b)
    assert tt_out.memory_config() == memory_config

    out = ttnn.to_torch(tt_out)
    assert out.shape == a.shape
    assert_with_pcc(GOLDEN[op](a, b), out.float(), pcc=0.999)
