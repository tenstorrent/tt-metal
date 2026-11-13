# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""paged_update_cache, paged_fused_update_cache and paged_fill_cache on a mesh with `mesh_coords`.

Their mesh factories emit one program per rectangle of coordinates that run the same program (the selected ones, and
for fill_cache also the unselected ones, which run a noop program). These tests check, for irregular selections with
holes, that every selected device is updated, every other device is left untouched, and that a second call on the
same cache entry with new tensors does the same.
"""

import pytest
import torch
import ttnn

MESH_SHAPES = [pytest.param((2, 4), id="2x4"), pytest.param((8, 4), id="8x4")]
PATTERNS = ["single", "row", "checker", "holes", "all"]


def selected_coords(pattern: str, shape) -> set[tuple[int, int]]:
    rows, cols = shape
    every = {(r, c) for r in range(rows) for c in range(cols)}
    if pattern == "single":
        return {(rows - 1, cols - 1)}
    if pattern == "row":
        return {(0, c) for c in range(cols)}
    if pattern == "checker":
        return {(r, c) for (r, c) in every if (r + c) % 2 == 0}
    if pattern == "holes":
        return every - {(0, 1), (rows - 1, 0)}
    assert pattern == "all"
    return every


def to_mesh_coords(coords) -> set:
    return {ttnn.MeshCoordinate(r, c) for (r, c) in coords}


def replicate(mesh_device, tensor, **kwargs):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        **kwargs,
    )


def per_device(mesh_device, tensor) -> list[torch.Tensor]:
    """The device shards of a replicated tensor, in row-major mesh order."""
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def check_devices(mesh_device, outputs, expected_selected, expected_other, selected):
    rows, cols = tuple(mesh_device.shape)
    assert len(outputs) == rows * cols
    for index, output in enumerate(outputs):
        coord = (index // cols, index % cols)
        expected = expected_selected if coord in selected else expected_other
        assert torch.equal(output, expected), f"device {coord} ({'selected' if coord in selected else 'not selected'})"


@pytest.mark.parametrize("mesh_device", MESH_SHAPES, indirect=True)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_paged_update_cache_mesh_coords(mesh_device, pattern):
    shape = tuple(mesh_device.shape)
    selected = selected_coords(pattern, shape)
    seq_len, head_dim, num_heads = 64, 64, 1
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    for call, (index, value) in enumerate([(5, 3.0), (40, 7.0)]):
        # New tensors on every call: the second call is a cache hit with new buffer addresses.
        cache = torch.zeros((1, num_heads, seq_len, head_dim))
        update = torch.full((1, 1, 32, head_dim), value)
        cache_tt = replicate(mesh_device, cache, dtype=ttnn.bfloat16)
        update_tt = replicate(mesh_device, update, dtype=ttnn.bfloat16, memory_config=input_memory_config)
        entries_before = mesh_device.num_program_cache_entries()
        ttnn.experimental.paged_update_cache(
            cache_tt, update_tt, update_idxs=[index], mesh_coords=to_mesh_coords(selected)
        )
        new_entries = mesh_device.num_program_cache_entries() - entries_before
        expected = cache.clone()
        expected[0, 0, index, :] = value
        check_devices(mesh_device, per_device(mesh_device, cache_tt), expected, cache, selected)
        assert new_entries == (1 if call == 0 else 0), f"call {call}: {new_entries} new program cache entries"


@pytest.mark.parametrize("mesh_device", MESH_SHAPES, indirect=True)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tiled", "row_major"])
def test_paged_fused_update_cache_mesh_coords(mesh_device, pattern, input_layout):
    """Tiled inputs select the tiled fused factory, row-major inputs the row-major one."""
    shape = tuple(mesh_device.shape)
    selected = selected_coords(pattern, shape)
    # The row-major fused factory takes 8 rows of 128 values.
    tiled = input_layout == ttnn.TILE_LAYOUT
    seq_len, head_dim = 64, 64 if tiled else 128
    input_rows = 32 if tiled else 8
    memory_configs = []
    for core in range(2):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core, 0), ttnn.CoreCoord(core, 0))})
        shard_spec = ttnn.ShardSpec(grid, [input_rows, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        memory_configs.append(ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec))
    for call, index in enumerate([5, 37]):
        cache = torch.zeros((1, 1, seq_len, head_dim))
        caches = [replicate(mesh_device, cache, dtype=ttnn.bfloat16) for _ in range(2)]
        values = (3.0 + call, 5.0 + call)
        inputs = [
            ttnn.from_torch(
                torch.full((1, 1, input_rows, head_dim), values[i]),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=input_layout,
                dtype=ttnn.bfloat16,
                memory_config=memory_configs[i],
            )
            for i in range(2)
        ]
        entries_before = mesh_device.num_program_cache_entries()
        ttnn.experimental.paged_fused_update_cache(
            caches[0], inputs[0], caches[1], inputs[1], update_idxs=[index], mesh_coords=to_mesh_coords(selected)
        )
        new_entries = mesh_device.num_program_cache_entries() - entries_before
        for i in range(2):
            expected = cache.clone()
            expected[0, 0, index, :] = values[i]
            check_devices(mesh_device, per_device(mesh_device, caches[i]), expected, cache, selected)
        assert new_entries == (1 if call == 0 else 0), f"call {call}: {new_entries} new program cache entries"


@pytest.mark.parametrize("mesh_device", MESH_SHAPES, indirect=True)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_paged_fill_cache_mesh_coords(mesh_device, pattern):
    shape = tuple(mesh_device.shape)
    selected = selected_coords(pattern, shape)
    num_blocks, block_size, head_dim, seq_len = 8, 32, 64, 128
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    for call in range(2):
        cache = torch.randn((num_blocks, 1, block_size, head_dim)).bfloat16().float()
        fill = torch.randn((1, 1, seq_len, head_dim)).bfloat16().float()
        cache_tt = replicate(mesh_device, cache, dtype=ttnn.bfloat16)
        fill_tt = replicate(mesh_device, fill, dtype=ttnn.bfloat16)
        entries_before = mesh_device.num_program_cache_entries()
        ttnn.experimental.paged_fill_cache(
            cache_tt, fill_tt, page_table_tt, batch_idx=0, mesh_coords=to_mesh_coords(selected)
        )
        new_entries = mesh_device.num_program_cache_entries() - entries_before
        expected = cache.clone()
        for block in range(seq_len // block_size):
            expected[block, 0] = fill[0, 0, block * block_size : (block + 1) * block_size]
        check_devices(mesh_device, per_device(mesh_device, cache_tt), expected, cache, selected)
        assert new_entries == (1 if call == 0 else 0), f"call {call}: {new_entries} new program cache entries"
