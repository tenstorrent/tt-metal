# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import copy
import torch
import torch.nn as nn
import ttnn
from models.common.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3, -1],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_like_int(device, input_shape, fill_value, layout):
    torch_input = torch.randint(0, 100, (input_shape), dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
        [3, 91, 67, 77],  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_like_float(device, input_shape, fill_value, dtype, layout):
    torch_input = torch.rand((input_shape), dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_like_callback(device, input_shape, fill_value, layout):
    for i in range(2):
        torch_input = torch.randint(0, 100, (input_shape), dtype=torch.int32)
        torch_output = torch.full_like(torch_input, fill_value)

        tt_input = ttnn.from_torch(torch_input, layout=layout, device=device)
        with device.cache_entries_counter.measure():
            tt_output = ttnn.moreh_full_like(tt_input, fill_value)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            first_count = device.cache_entries_counter.total
            assert first_count > 0
        else:
            assert device.cache_entries_counter.total == first_count
    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([32, 32], [32, 32]),
        ([5, 96, 64], [1, 32, 32]),
        ([3, 160, 160], [2, 64, 64]),
        ([2, 4, 128, 160], [2, 3, 96, 96]),
    ],
)
@pytest.mark.parametrize("fill_value", [3, -1])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
    ],
)
def test_full_like_int_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([32, 32], [32, 32]),
        ([5, 96, 64], [1, 32, 32]),
        ([3, 160, 160], [2, 64, 64]),
        ([2, 4, 128, 160], [2, 3, 96, 96]),
        ([7, 67, 77], [3, 32, 64]),  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
    ],
)
def test_full_like_float_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, dtype, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.rand(input_shape, dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([4, 128, 128], [3, 96, 96]),
    ],
)
@pytest.mark.parametrize("fill_value", [3])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
    ],
)
def test_full_like_callback_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    for i in range(2):
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
        torch_output = torch.full_like(torch_input, fill_value)

        tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
        tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([3, 160, 160], [3, 160, 64]),
        ([3, 160, 160], [2, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
    ],
)
def test_full_like_float_DRAM_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, dtype, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    torch_input = torch.rand(input_shape, dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded
        (
            [4, 128, 64],
            [128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        (
            [2, 64, 64],
            [64, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # Width sharded
        (
            [2, 64, 128],
            [128, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Block sharded (shard_shape must be [height/grid.y, width/grid.x])
        (
            [4, 128, 128],
            [256, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [2, 128, 128],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # Block sharded, uneven (shard coverage > tensor size, last shards have padding)
        (
            [2, 96, 96],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [3, -1])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_full_like_int_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded
        (
            [4, 128, 64],
            [128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        (
            [2, 64, 64],
            [64, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # Width sharded
        (
            [2, 64, 128],
            [128, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Block sharded
        (
            [4, 128, 128],
            [256, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [2, 128, 128],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # Block sharded, uneven
        (
            [2, 96, 96],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_full_like_float_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, dtype, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.rand(input_shape, dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded, uneven, DRAM (1D grid required for DRAM)
        # height=192, shard_height=128, 2 cores: 128*2=256 > 192, last core has 64 valid rows
        (
            [2, 96, 96],
            [128, 96],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [3, -1])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_full_like_int_legacy_DRAM_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded
        (
            [4, 128, 64],
            [128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Width sharded
        (
            [2, 64, 128],
            [128, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Block sharded
        (
            [2, 128, 128],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [3])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR],
)
def test_full_like_callback_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    for i in range(2):
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
        torch_output = torch.full_like(torch_input, fill_value)

        tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
        tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
    assert torch.equal(torch_output, tt_output_cpu)


# ---------------------------------------------------------------------------
# Mesh topology
#
# prim::full has no tensor inputs, so the device-operation framework cannot derive the output's mesh placements
# from the input the way it does for every other op. moreh_full_like must carry them through itself, or a
# full_like of a sharded tensor comes back labelled fully replicated and anything that trusts tensor_topology()
# (mesh composers, checkpointing) treats one device's shard as the whole tensor.
# ---------------------------------------------------------------------------


def _assert_same_topology(tt_input, tt_output):
    input_topology = tt_input.tensor_topology()
    output_topology = tt_output.tensor_topology()
    assert input_topology == output_topology, f"Topology mismatch: input={input_topology}, output={output_topology}"


def _placement_names(topology):
    return [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]


@pytest.mark.parametrize("shape_suffix", [[1, 32, 32], [3, 64, 96]])
@pytest.mark.parametrize("fill_value", [3.0, -1.5])
def test_full_like_preserves_topology_replicate(mesh_device, shape_suffix, fill_value):
    torch_input = torch.zeros([2] + shape_suffix, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_output = ttnn.moreh_full_like(tt_input, fill_value)

    _assert_same_topology(tt_input, tt_output)
    expected = torch.full_like(torch_input, fill_value)
    shards = ttnn.get_device_tensors(tt_output)
    assert len(shards) == mesh_device.get_num_devices()
    for shard in shards:
        assert torch.equal(ttnn.to_torch(shard), expected)


@pytest.mark.parametrize("shape_suffix", [[1, 32, 32], [3, 64, 96]])
@pytest.mark.parametrize("fill_value", [3.0, -1.5])
def test_full_like_preserves_topology_shard(mesh_device, shape_suffix, fill_value):
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("needs a mesh with at least two devices to shard over")
    full_shape = [num_devices] + shape_suffix
    torch_input = torch.zeros(full_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    assert _placement_names(tt_input.tensor_topology()) == ["Shard(0)"], "precondition: input is sharded on dim 0"

    tt_output = ttnn.moreh_full_like(tt_input, fill_value)

    _assert_same_topology(tt_input, tt_output)
    # Each device holds one shard; composing them along the sharded dim gives the full tensor back, not one shard.
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert list(composed.shape) == full_shape
    assert torch.equal(composed, torch.full(full_shape, fill_value, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "dtype, torch_dtype, fill_value",
    [
        (ttnn.float32, torch.float32, 2.5),
        (ttnn.int32, torch.int32, 7),
    ],
)
def test_full_like_preserves_topology_with_dtype_override(mesh_device, dtype, torch_dtype, fill_value):
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("needs a mesh with at least two devices to shard over")
    full_shape = [num_devices, 2, 32, 64]
    tt_input = ttnn.from_torch(
        torch.zeros(full_shape, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    tt_output = ttnn.moreh_full_like(tt_input, fill_value, dtype=dtype)

    assert tt_output.dtype == dtype
    _assert_same_topology(tt_input, tt_output)
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert torch.equal(composed, torch.full(full_shape, fill_value, dtype=torch_dtype))


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("dims", [(0, 1), (0, 3), (None, 2)])
def test_full_like_preserves_topology_shard_2d(mesh_device, dims):
    mesh_shape = tuple(mesh_device.shape)
    assert mesh_shape == (2, 2), f"fixture opened a {mesh_shape} mesh"

    full_shape = [2, 2, 64, 64]
    tt_input = ttnn.from_torch(
        torch.zeros(full_shape, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=list(mesh_shape), dims=dims),
    )
    expected_names = [f"Shard({d})" if d is not None else "Replicate" for d in dims]
    assert _placement_names(tt_input.tensor_topology()) == expected_names, "precondition: 2-D placements"
    assert list(tt_input.tensor_topology().distribution_shape()) == list(mesh_shape)

    tt_output = ttnn.moreh_full_like(tt_input, -3.0)

    _assert_same_topology(tt_input, tt_output)
    for shard in ttnn.get_device_tensors(tt_output):
        assert torch.all(ttnn.to_torch(shard) == -3.0)


def test_full_like_does_not_adopt_sub_mesh_topology(mesh_device):
    """An input distributed over part of the mesh: full_like allocates on the whole mesh, so it must not copy a
    topology whose distribution is smaller than the mesh (same rule as ttnn.empty_like) -- that would describe
    shards the allocation does not have. The output falls back to fully replicated over the mesh."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("needs a mesh with at least two devices")
    one_device_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate()], mesh_shape_override=ttnn.MeshShape([1])),
    )
    tt_input = ttnn.from_torch(
        torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=one_device_mapper,
    )
    assert tt_input.tensor_topology().distribution_shape().mesh_size() == 1, "precondition: one-device distribution"

    tt_output = ttnn.moreh_full_like(tt_input, 1.0)

    output_topology = tt_output.tensor_topology()
    assert output_topology.distribution_shape().mesh_size() == num_devices
    assert all(isinstance(p, ttnn.PlacementReplicate) for p in output_topology.placements())
    shards = ttnn.get_device_tensors(tt_output)
    assert len(shards) == num_devices
    for shard in shards:
        assert torch.equal(ttnn.to_torch(shard), torch.ones([1, 1, 32, 32], dtype=torch.bfloat16))


def test_full_like_topology_survives_program_cache(mesh_device):
    """Same shape/dtype/layout, different distributions: the cached program is reused, and each output must still
    report its own input's topology (the topology is set when the output is created, not by the program)."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("needs a mesh with at least two devices to shard over")
    # Same per-device shape in every case ([1, 1, 32, 32]), so the program (and its cache entry) is identical.
    cases = [
        ([num_devices, 1, 32, 32], ttnn.ShardTensorToMesh(mesh_device, dim=0)),
        ([1, 1, 32, 32], ttnn.ReplicateTensorToMesh(mesh_device)),
        ([num_devices, 1, 32, 32], ttnn.ShardTensorToMesh(mesh_device, dim=0)),
    ]
    counts = []
    for shape, mapper in cases:
        tt_input = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        with mesh_device.cache_entries_counter.measure():
            tt_output = ttnn.moreh_full_like(tt_input, 5.0)
        counts.append(mesh_device.cache_entries_counter.total)
        _assert_same_topology(tt_input, tt_output)
    assert counts[0] > 0
    assert counts[1] == counts[2] == counts[0], f"program cache entries changed across topologies: {counts}"
