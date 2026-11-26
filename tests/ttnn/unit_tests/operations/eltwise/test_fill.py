# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("fill_value", [2147483647, -2147483648, 15.5])
def test_fill_int32(device, fill_value):
    torch_input_tensor = torch.zeros((1, 2), dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    output_tensor = ttnn.to_torch(output)

    equal_passed, equal_message = assert_equal(torch_output_tensor, output_tensor)
    assert equal_passed


@pytest.mark.parametrize("fill_value", [2147483647, 25.5, 4294967293, 1000000000, 4294967295])
def test_fill_uint32(device, fill_value):
    torch_input_tensor = torch.ones((1, 2), dtype=torch.uint32)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    torch_output = ttnn.from_torch(torch_output_tensor, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.eq(output, torch_output)
    assert torch.all(ttnn.to_torch(result))


@pytest.mark.parametrize("fill_value", [2, 43.5, 65535])
def test_fill_uint16(device, fill_value):
    torch_input_tensor = torch.ones((1, 2), dtype=torch.uint16)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    torch_output = ttnn.from_torch(torch_output_tensor, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.eq(output, torch_output)
    assert torch.all(ttnn.to_torch(result))


def _alloc_with_neighbors(device, shape, dtype, mc, tile_shape, target_value):
    """Allocates a non-32x32-tile target between 32x32-tile guards, so a write past its end lands in a guard."""
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    guards, target = [], None
    for i in range(3):
        g = torch.full(shape, float(i + 2), dtype=tdt)
        guards.append((g, ttnn.from_torch(g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)))
        if i == 0:
            target = ttnn.from_torch(
                torch.full(shape, target_value, dtype=tdt),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mc,
                tile=ttnn.Tile(tile_shape),
            )
    return tdt, target, guards


@pytest.mark.parametrize("tile_shape", [(16, 32), (32, 16), (8, 32)], ids=lambda s: f"tile_{s[0]}x{s[1]}")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("mc", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize("value", [0.0, 7.0])
def test_fill_non_default_tile_keeps_neighbors(device, tile_shape, dtype, mc, value):
    shape = (1, 1, 64, 256)
    tdt, target, guards = _alloc_with_neighbors(device, shape, dtype, mc, tile_shape, -1.0)

    out = ttnn.fill(target, value, output_tensor=target)

    assert out.tile.tile_shape == list(tile_shape)
    assert torch.equal(ttnn.to_torch(out), torch.full(shape, value, dtype=tdt))
    for g, tg in guards:
        assert torch.equal(ttnn.to_torch(tg), g), "a neighboring tensor was overwritten"


@pytest.mark.parametrize("tile_shape", [(16, 32), (32, 16), (8, 32)], ids=lambda s: f"tile_{s[0]}x{s[1]}")
@pytest.mark.parametrize("mc", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_zeros_like_non_default_tile_keeps_neighbors(device, tile_shape, mc):
    shape = (1, 1, 96, 128)
    tdt, target, guards = _alloc_with_neighbors(device, shape, ttnn.bfloat16, mc, tile_shape, 5.0)

    ttnn.zeros_like(target, optional_tensor=target)

    assert torch.equal(ttnn.to_torch(target), torch.zeros(shape, dtype=tdt))
    for g, tg in guards:
        assert torch.equal(ttnn.to_torch(tg), g), "a neighboring tensor was overwritten"


def _height_sharded(num_cores, shard_height, width):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_height, width], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("tile_shape", [(16, 32), (32, 16), (8, 32)], ids=lambda s: f"tile_{s[0]}x{s[1]}")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_fill_non_default_tile_height_sharded_keeps_neighbors(device, tile_shape, dtype):
    # The sharded path borrows the shard as its DFBs, which must be sized by the tensor's tile too.
    shape = (1, 1, 64, 256)
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    # A shard as short as the tile allows: 16 rows on 4 cores for 16- and 8-row tiles, 32 rows on 2 for 32-row ones.
    shard_height = max(tile_shape[0], 16)
    target_config = _height_sharded(shape[2] // shard_height, shard_height, shape[3])
    guard_config = _height_sharded(2, 32, shape[3])

    guards, target = [], None
    for i in range(3):
        g = torch.full(shape, float(i + 2), dtype=tdt)
        guards.append(
            (g, ttnn.from_torch(g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=guard_config))
        )
        if i == 0:
            target = ttnn.from_torch(
                torch.full(shape, -1.0, dtype=tdt),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=target_config,
                tile=ttnn.Tile(tile_shape),
            )

    out = ttnn.fill(target, 7.0, output_tensor=target)

    assert out.is_sharded()
    assert out.tile.tile_shape == list(tile_shape)
    assert torch.equal(ttnn.to_torch(out), torch.full(shape, 7.0, dtype=tdt))
    for g, tg in guards:
        assert torch.equal(ttnn.to_torch(tg), g), "a neighboring tensor was overwritten"
