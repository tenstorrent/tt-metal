# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ],
)
def test_serialize_memory_config_without_shard_spec(tmp_path, memory_config):
    memory_config_path = str(tmp_path / "memory_config.bin")
    ttnn.dump_memory_config(memory_config_path, memory_config)
    deserialized_memory_config = ttnn.load_memory_config(memory_config_path)
    assert memory_config == deserialized_memory_config
    assert hash(memory_config) == hash(deserialized_memory_config)


def test_serialize_memory_config_with_shard_spec(tmp_path):
    memory_config_path = str(tmp_path / "memory_config.bin")
    memory_config = ttnn.create_sharded_memory_config(
        (32, 128),
        ttnn.CoreGrid(y=8, x=4),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn.dump_memory_config(memory_config_path, memory_config)
    deserialized_memory_config = ttnn.load_memory_config(memory_config_path)
    assert memory_config == deserialized_memory_config
    assert hash(memory_config) == hash(deserialized_memory_config)


def test_serialize_memory_config_with_shard_spec_over_core_range_set(tmp_path):
    memory_config_path = str(tmp_path / "memory_config.bin")
    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(4, 4),
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(5, 5),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )

    num_attention_heads = 16
    sequence_lengtb = 64

    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                num_attention_heads * sequence_lengtb // 32,
                sequence_lengtb,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    ttnn.dump_memory_config(memory_config_path, memory_config)
    deserialized_memory_config = ttnn.load_memory_config(memory_config_path)
    assert memory_config == deserialized_memory_config
    assert hash(memory_config) == hash(deserialized_memory_config)
