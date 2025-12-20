# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "dtype",
    [ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32, ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
)
@pytest.mark.parametrize(
    "input_shape, dim, start, length, memory_config, layout",
    [
        ((32, 32, 17, 32), 0, 12, 16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM), ttnn.TILE_LAYOUT),
        ((1, 32, 12, 16), 1, 5, 8, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM), ttnn.ROW_MAJOR_LAYOUT),
        (
            (1, 8, 64, 128),
            1,
            4,
            4,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (32, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
        (
            (1, 8, 64, 128),
            3,
            32,
            32,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (32, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
        (
            (1, 8, 128, 128),
            2,
            96,
            32,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (64, 128),
                    ttnn.ShardOrientation.COL_MAJOR,
                ),
            ),
            ttnn.ROW_MAJOR_LAYOUT,
        ),
        (
            (1, 1, 32, 576),
            -1,
            512,
            64,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))}),
                    (32, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
        (
            (1, 1, 8, 576),
            3,
            512,
            64,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))}),
                    (8, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.ROW_MAJOR_LAYOUT,
        ),
        (
            (1, 32, 16, 192),
            3,
            128,
            64,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 7))}),
                    (128, 32),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
    ],
    ids=[
        "dram_dim0_tile",
        "dram_dim1_rm",
        "l1_height_sharded_dim1_tile",
        "l1_height_sharded_dim3_tile",
        "l1_height_sharded_dim2_rm",
        "l1_width_sharded_dim3_tile",
        "l1_width_sharded_dim3_rm",
        "l1_block_sharded_dim3_tile",
    ],
)
def test_narrow(input_shape, dim, start, length, memory_config, layout, dtype, device):
    if (dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Skipping test for bfloat8_b or bfloat4_b with ROW_MAJOR_LAYOUT")

    if dtype in [ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32]:
        torch_input_tensor = torch.randint(0, 128, input_shape, dtype=tt_dtype_to_torch_dtype[dtype])
    else:
        torch_input_tensor = torch.randn(input_shape, dtype=tt_dtype_to_torch_dtype[dtype])
    torch_result = torch.narrow(torch_input_tensor, dim, start, length)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.narrow(input_tensor, dim, start, length)

    assert layout == ttnn_output.layout
    assert memory_config.buffer_type == ttnn_output.memory_config().buffer_type
    assert memory_config.memory_layout == ttnn_output.memory_config().memory_layout
    output = ttnn.to_torch(ttnn_output)
    if dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b:
        target_pcc = 0.95 if dtype == ttnn.bfloat4_b else 0.99
        assert_with_pcc(torch_result, output, target_pcc)
    else:
        assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "input_shape, dim, start, length, memory_config, layout",
    [
        (
            (8, 4, 128, 128),
            3,
            32,
            32,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (32, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
    ],
    ids=["l1_height_sharded"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_narrow_mesh(input_shape, dim, start, length, memory_config, layout, mesh_device):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_result = torch.narrow(torch_input_tensor, dim, start, length)
    mesh_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape)

    input_tensor_mesh = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_config),
    )
    ttnn_output = ttnn.narrow(input_tensor_mesh, dim, start, length)

    output = ttnn.to_torch(
        ttnn_output, mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[0, 1]))
    )
    assert_equal(torch_result, output)
