# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.nlp_concat_heads_decode import num_to_corerange, run_test_concat_head

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn
import ttnn.core

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from models.common.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    nearest_32,
)


@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim, batch_size",
    (
        # Single head-tile (padded_local_heads == 32)
        (8, 32, 128, 32),
        (17, 32, 96, 32),
        (32, 32, 64, 32),
        (8, 32, 128, 16),
        # Multiple head-tiles (padded_local_heads > 32)
        (8, 64, 64, 32),  # 2 head-tiles, low utilization
        (32, 64, 64, 32),  # 2 head-tiles, fully populated within first tile
        (32, 64, 128, 32),  # 2 head-tiles + multi-tile head_dim
        (16, 96, 64, 32),  # 3 head-tiles
        # n_local_heads > 32 — exercises head_tile_idx > 0 in the offset formula
        # (batch is capped at 32 by the op; n_local_heads uses device compute grid)
        (40, 64, 64, 8),  # heads 32..39 read from 2nd head-tile
        (64, 64, 64, 8),  # all 32 heads in 2nd head-tile fully populated
        (40, 96, 64, 8),  # 3 head-tiles, heads 32..39 from 2nd tile (non-power-of-2 padded_local_heads)
    ),
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1_grid")], indirect=True)
def test_concat_head(
    n_local_heads,
    padded_local_heads,
    head_dim,
    batch_size,
    mesh_device,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head(mesh_device, n_local_heads, padded_local_heads, head_dim, batch_size)


@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim, batch_size, input_sub_core_grids, compute_sub_core_grids",
    (
        (  # Test Case 0: Input is sharded on 8 cores, use all 8 cores for compute
            8,
            32,
            128,
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
        ),
        (  # Test Case 1: Input is sharded on only 1 core, use the compute sub core grids for compute
            8,
            32,
            128,
            1,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
        ),
        (  # Test Case 2: padded_local_heads=64 (multi head-tile path) on subcoregrids
            8,
            64,
            128,
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
        ),
        (  # Test Case 3: n_local_heads=64 — exercises head_tile_idx > 0 on subcoregrids
            64,
            64,
            64,
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
                ]
            ),
        ),
    ),
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1_grid")], indirect=True)
def test_concat_head_subcoregrids(
    n_local_heads,
    padded_local_heads,
    head_dim,
    batch_size,
    input_sub_core_grids,
    compute_sub_core_grids,
    mesh_device,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head(
            mesh_device,
            n_local_heads,
            padded_local_heads,
            head_dim,
            batch_size,
            input_sub_core_grids,
            compute_sub_core_grids,
        )


@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("grid_kind", ["origin", "origin_subgrid", "offset", "disjoint", "wide", "wide_subgrid"])
@pytest.mark.parametrize("num_heads,padded_heads,head_dim", [(8, 32, 32), (24, 32, 128), (40, 64, 64)])
def test_concat_head_input_shard_order(device, orientation, dtype, grid_kind, num_heads, padded_heads, head_dim):
    available = device.compute_with_storage_grid_size()
    users = 32 if grid_kind.startswith("wide") else 8
    width, height = (8, 4) if users == 32 else (2, 4)
    required = {"offset": (3, 5), "disjoint": (5, 5)}.get(grid_kind, (width, height))
    if available.x < required[0] or available.y < required[1] or available.x * available.y < num_heads:
        pytest.skip("Input placement and one output core per head must fit the device grid")
    start = ttnn.CoreCoord(1, 1) if grid_kind == "offset" else ttnn.CoreCoord(0, 0)
    if grid_kind == "disjoint":
        grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(3, 3), ttnn.CoreCoord(4, 4)),
            }
        )
    else:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(start, ttnn.CoreCoord(start.x + width - 1, start.y + height - 1))})
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [padded_heads, head_dim], orientation),
    )
    kwargs = {}
    if grid_kind not in ("origin", "wide"):
        kwargs["sub_core_grids"] = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(available.x - 1, available.y - 1))}
        )
    # Keep both launches allocated so the cache hit must bind different buffer addresses.
    keep_alive = []
    cache_entries = None
    for offset in [0, 7]:
        values = ((torch.arange(users * padded_heads * head_dim) + offset) % 97).reshape(
            1, users, padded_heads, head_dim
        )
        values = values.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
        tensor = ttnn.from_torch(values, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory)
        result = ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=num_heads, **kwargs)
        actual = ttnn.to_torch(result)[..., :users, :]
        expected = values[:, :, :num_heads].reshape(1, 1, users, num_heads * head_dim)
        assert torch.equal(actual, expected)
        keep_alive += [tensor, result]
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == cache_entries


@pytest.mark.parametrize("input_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("output_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("use_subgrid", [False, True])
@pytest.mark.parametrize("output_grid", ["origin", "offset", "disjoint"])
def test_concat_head_output_shard_order(device, input_orientation, output_orientation, dtype, use_subgrid, output_grid):
    available = device.compute_with_storage_grid_size()
    required = {"origin": (2, 4), "offset": (4, 7), "disjoint": (6, 6)}[output_grid]
    if available.x < required[0] or available.y < required[1]:
        pytest.skip("Supplied output placement must fit the device grid")
    input_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
    if output_grid == "disjoint":
        grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2)),
                ttnn.CoreRange(ttnn.CoreCoord(4, 4), ttnn.CoreCoord(5, 5)),
            }
        )
    elif output_grid == "offset":
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(3, 6))})
    else:
        grid = input_grid
    input_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_grid, [32, 32], input_orientation),
    )
    output_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 32], output_orientation),
    )
    kwargs = {"sub_core_grids": input_grid} if use_subgrid else {}
    # Keep both launches allocated so the cache hit must bind different buffer addresses.
    keep_alive = []
    cache_entries = None
    for offset in [0, 7]:
        values = ((torch.arange(8 * 32 * 32) + offset) % 97).reshape(1, 8, 32, 32)
        values = values.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
        tensor = ttnn.from_torch(
            values, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=input_memory
        )
        output = ttnn.from_torch(
            torch.full((1, 1, 32, 256), -31.0 - offset),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memory,
        )
        result = ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, output_tensor=output, **kwargs)
        assert torch.equal(ttnn.to_torch(result)[..., :8, :], values[:, :, :8].reshape(1, 1, 8, 256))
        keep_alive += [tensor, output, result]
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == cache_entries


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("use_subgrid", [False, True])
@pytest.mark.parametrize("small_tensor", ["input", "output"])
@pytest.mark.parametrize("tile_shape", [(32, 16), (8, 32)], ids=["small_width", "small_face"])
def test_concat_head_requires_standard_tiles(device, expect_error, dtype, use_subgrid, small_tensor, tile_shape):
    available = device.compute_with_storage_grid_size()
    if available.x < 2 or available.y < 4:
        pytest.skip("Tile validation regression needs a 2x4 input grid")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
    input_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tile = ttnn.Tile(tile_shape) if small_tensor == "input" else ttnn.Tile([32, 32])
    values = torch.arange(8 * 32 * 32, dtype=torch.float32).reshape(1, 8, 32, 32) % 97
    tensor = ttnn.from_torch(
        values, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=input_memory, tile=tile
    )
    kwargs = {"sub_core_grids": grid} if use_subgrid else {}
    if small_tensor == "output":
        output_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        kwargs["output_tensor"] = ttnn.from_torch(
            torch.zeros([1, 1, 32, 256]),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memory,
            tile=ttnn.Tile(tile_shape),
        )
    with expect_error(RuntimeError, f"{small_tensor.capitalize()} tile must be non-transposed 32x32"):
        ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, **kwargs)


@pytest.mark.parametrize("invalid_tensor", ["input", "output"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_concat_head_rejects_transposed_tiles(mesh_device, expect_error, invalid_tensor):
    available = mesh_device.compute_with_storage_grid_size()
    if available.x < 2 or available.y < 4:
        pytest.skip("Tile validation regression needs a 2x4 input grid")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
    input_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tile = ttnn.Tile([32, 32], transpose_tile=True)
    values = torch.arange(8 * 32 * 32, dtype=torch.float32).reshape(1, 8, 32, 32)
    tensor = ttnn.from_torch(
        values, device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=input_memory
    )
    output = ttnn.from_torch(
        torch.zeros([1, 1, 32, 256]),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_memory,
    )
    # Transpose flags do not distinguish cache entries; validate even after a standard-tile launch.
    ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, output_tensor=output)
    if invalid_tensor == "input":
        tensor = ttnn.from_torch(
            values,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_memory,
            tile=tile,
        )
    else:
        output = ttnn.from_torch(
            torch.zeros([1, 1, 32, 256]),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memory,
            tile=tile,
        )
    with expect_error(RuntimeError, f"{invalid_tensor.capitalize()} tile must be non-transposed 32x32"):
        ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, output_tensor=output)


@pytest.mark.parametrize(
    "case,error",
    [
        ("narrow16", "Output shard shape must be"),
        ("narrow32", "Output shard shape must be"),
        ("wide4", "Output grid must have at least num_heads"),
        ("shard_width", "Output shard shape must be"),
        ("dtype", "Output dtype must match input dtype"),
        ("interleaved", "Output memory layout must be WIDTH_SHARDED"),
        ("height_sharded", "Output memory layout must be WIDTH_SHARDED"),
        ("row_major", "Output layout must be TILE"),
        ("output_shape", "Output padded shape must be"),
        ("output_width", "Output logical width must be"),
        ("output_rows", "Output must contain all"),
        ("input_dram", "Input buffer must be L1"),
        ("output_dram", "Output buffer must be L1"),
    ],
    ids=[
        "narrow16",
        "narrow32",
        "wide4",
        "shard_width",
        "dtype",
        "interleaved",
        "height_sharded",
        "row_major",
        "output_shape",
        "output_width",
        "output_rows",
        "input_dram",
        "output_dram",
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_concat_head_requires_valid_buffers(mesh_device, expect_error, case, error):
    available = mesh_device.compute_with_storage_grid_size()
    required_cores = {"narrow16": 16, "narrow32": 32}.get(case, 8)
    if available.x < 8 or available.x * available.y < required_cores:
        pytest.skip("Buffer validation placement must fit the device grid")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    head_dim = {"narrow16": 64, "narrow32": 128}.get(case, 32)
    input_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM if case == "input_dram" else ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_grid = grid
    shard_shape = [32, head_dim]
    if case in ("narrow16", "narrow32"):
        output_grid = ttnn.num_cores_to_corerangeset(required_cores, available, True)
        shard_shape = [32, 32]
    elif case == "wide4":
        output_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [32, 64]
    elif case == "shard_width":
        shard_shape = [32, 64]
    elif case == "output_shape":
        shard_shape = [64, 32]
    elif case == "height_sharded":
        shard_shape = [32, 256]
    output_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED if case == "height_sharded" else ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM if case == "output_dram" else ttnn.BufferType.L1,
        ttnn.ShardSpec(output_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    if case == "interleaved":
        output_memory = ttnn.L1_MEMORY_CONFIG
    values = torch.arange(8 * 32 * head_dim, dtype=torch.float32).reshape(1, 8, 32, head_dim) % 97
    tensor = ttnn.from_torch(
        values, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=input_memory
    )
    shape = [
        1,
        1,
        {"output_shape": 64, "output_rows": 4}.get(case, 32),
        255 if case == "output_width" else 8 * head_dim,
    ]
    output = ttnn.from_torch(
        torch.zeros(shape),
        device=mesh_device,
        dtype=ttnn.float32 if case == "dtype" else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if case == "row_major" else ttnn.TILE_LAYOUT,
        memory_config=output_memory,
    )
    with expect_error(RuntimeError, error):
        ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, output_tensor=output)


@pytest.fixture
def concat_device_pair():
    if ttnn.get_num_devices() < 2:
        pytest.skip("Requires two devices for supplied-output placement validation")
    devices = ttnn.CreateDevices([0, 1])
    try:
        yield devices[0], devices[1]
    finally:
        ttnn.CloseDevices(devices)


@pytest.mark.parametrize("warm_cache", [False, True], ids=["cache_miss", "cache_hit"])
def test_concat_head_rejects_other_device_output(concat_device_pair, expect_error, warm_cache):
    a, b = concat_device_pair
    for device in (a, b):
        available = device.compute_with_storage_grid_size()
        if available.x < 2 or available.y < 4:
            pytest.skip("Foreign-output regression requires a 2x4 core grid on both devices")

    def call_with_foreign_output():
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
        input_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tensor = ttnn.from_torch(
            torch.ones((1, 8, 32, 32)),
            device=a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_memory,
        )
        local_output = ttnn.from_torch(
            torch.full((1, 1, 32, 256), -31.0),
            device=a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memory,
        )
        if warm_cache:
            ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, output_tensor=local_output)
        output = ttnn.from_torch(
            torch.full((1, 1, 32, 256), -31.0),
            device=b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memory,
        )
        with expect_error(RuntimeError, "Output tensor must be on the same device as input"):
            ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=8, output_tensor=output)

    call_with_foreign_output()


@pytest.mark.parametrize("logical_width,heads,spare", [(32, 2, 1), (32, 32, 1), (48, 1, 0), (48, 1, 2)])
@pytest.mark.parametrize("col_major", [False, True])
@pytest.mark.parametrize("use_subgrid", [False, True])
def test_concat_head_supplied_padding(device, logical_width, heads, spare, col_major, use_subgrid):
    padded_width = ((logical_width + 31) // 32) * 32
    available = device.compute_with_storage_grid_size()
    start_y = 1 if use_subgrid else 0
    if available.x < 8 or available.y < start_y + 4 or available.x * available.y < heads + spare:
        pytest.skip("Supplied output regression needs enough cores for the requested heads and spare shards")
    input_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, start_y), ttnn.CoreCoord(7, start_y + 3))})
    values = (torch.arange(32 * 32 * logical_width).reshape(1, 32, 32, logical_width) % 97).bfloat16()
    input_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_grid, [32, padded_width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT, memory_config=input_mem)
    output_grid = ttnn.num_cores_to_corerangeset(heads + spare, available, True)
    output_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_grid,
            [32, padded_width],
            ttnn.ShardOrientation.COL_MAJOR if col_major else ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    output = ttnn.from_torch(
        torch.full((1, 1, 32, heads * logical_width), -31).bfloat16(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem,
    )
    kwargs = dict(num_heads=heads, output_tensor=output)
    if use_subgrid:
        kwargs["sub_core_grids"] = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(available.x - 1, available.y - 1))}
        )
    for _ in range(2):
        result = ttnn.experimental.nlp_concat_heads_decode(input_tensor, **kwargs)
        expected = values[:, :, :heads, :].reshape(1, 1, 32, heads * logical_width)
        assert result.buffer_address() == output.buffer_address()
        assert torch.equal(ttnn.to_torch(result), expected)
