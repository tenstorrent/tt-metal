# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


def golden_indexed_fill(input_a, input_b, batch_id, dim):
    # Reference for ttnn.indexed_fill: replace slice `batch_id[k]` of input_a along `dim`
    # with slice k of input_b. Applied sequentially so that on duplicate indices the
    # last-listed value wins, matching the kernel's full-scan (no early break) semantics.
    out = input_a.clone()
    for k, v in enumerate(batch_id.flatten().tolist()):
        out.index_copy_(dim, torch.tensor([v]), input_b.index_select(dim, torch.tensor([k])))
    return out


@pytest.mark.parametrize(
    "input_a_shape, input_b_shape",
    [
        ((32, 1, 20, 24), (6, 1, 20, 24)),
        ((29, 1, 32, 56), (14, 1, 32, 56)),
        ((22, 1, 55, 29), (17, 1, 55, 29)),
        ((25, 1, 67, 83), (14, 1, 67, 83)),
        ((17, 1, 10, 12), (18, 1, 10, 12)),
    ],
)
def test_indexed_fill_tile_layout(device, input_a_shape, input_b_shape):
    # The runtime pads the last two dimensions to the next tile boundary (32) automatically,
    # so the logical shape is preserved while the padded shape satisfies tile alignment.
    B = input_a_shape[0]  # number of batches in input_a
    b = input_b_shape[0]  # number of replacement slabs; batch_id must have exactly b indices

    # batch_id: b indices, each in [0, B) — tells the op which input_a batch to replace.
    batch_id = torch.randint(0, B, (1, 1, 1, b))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    # Both data inputs are in TILE layout.
    torch_a = torch.rand(input_a_shape, dtype=torch.bfloat16)
    torch_b = torch.rand(input_b_shape, dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Perform indexed fill - output preserves TILE layout.
    output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b)
    logger.info("Indexed Fill (TILE) Output Tensor Shape:", output_tensor.shape)
    logger.info("Indexed Fill (TILE) Output Tensor Layout:", output_tensor.layout)

    golden = golden_indexed_fill(torch_a, torch_b, batch_id, dim=0)
    assert_with_pcc(golden, ttnn.to_torch(output_tensor), 0.9999)


@pytest.mark.parametrize(
    "B, b, D",
    [
        (8, 3, 64),
        (4, 2, 32),
        (6, 4, 128),
        (2, 1, 64),
    ],
    ids=["B8-b3-D64", "B4-b2-D32", "B6-b4-D128", "B2-b1-D64"],
)
def test_indexed_fill_sharded(device, B, b, D):
    # HEIGHT_SHARDED L1 example using the native CB-aliased fast path: input_a and the
    # output share the same shard geometry (one batch slab per core) so the kernel writes
    # the result directly into the output's per-core L1 shard with zero copy.
    input_a_shape = (B, 1, 1, D)
    input_b_shape = (b, 1, 1, D)

    # Batch-id tensor (must remain ROW_MAJOR, interleaved L1).
    batch_id = torch.randint(0, B, (1, 1, 1, b))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    # HEIGHT_SHARDED L1 memory config: B cores in a 1xB grid, one (1, D) shard per core.
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(B, 1, 1, D),
        core_grid=ttnn.CoreGrid(y=1, x=B),
        strategy=ttnn.ShardStrategy.HEIGHT,
    )
    interleaved_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Build input_a sharded; input_b stays interleaved (its buffer type is unrestricted).
    torch_a = torch.rand(input_a_shape, dtype=torch.bfloat16)
    torch_b = torch.rand(input_b_shape, dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded_mem_config
    )
    input_tensor_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=interleaved_l1
    )

    # Request a sharded output with the same geometry to enable the native fast path.
    output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, memory_config=sharded_mem_config)
    logger.info("Indexed Fill (sharded) Output Tensor Shape:", output_tensor.shape)
    logger.info("Indexed Fill (sharded) Output Memory Layout:", output_tensor.memory_config().memory_layout)

    golden = golden_indexed_fill(torch_a, torch_b, batch_id, dim=0)
    assert_with_pcc(golden, ttnn.to_torch(output_tensor), 0.9999)


@pytest.mark.parametrize(
    "B, H, W, b, core_grid_y, core_grid_x",
    [
        (4, 64, 64, 2, 2, 2),
        (8, 64, 128, 3, 2, 4),
        (4, 128, 64, 1, 2, 2),
        (4, 64, 128, 2, 2, 2),
    ],
    ids=["B4-H64-W64-b2-2x2", "B8-H64-W128-b3-2x4", "B4-H128-W64-b1-2x2", "B4-H64-W128-b2-2x2"],
)
def test_indexed_fill_block_sharded_tile(device, B, H, W, b, core_grid_y, core_grid_x):
    # BLOCK_SHARDED + TILE layout exercises the SHARD_LOCAL_INTERLEAVED_B path with
    # 2D shard geometry: each core owns a (shard_H × shard_W) tile block of input_a
    # and reads its replacement rows from interleaved input_b.
    batch_id = torch.randint(0, B, (1, 1, 1, b))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(B, 1, H, W),
        core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
    )
    interleaved_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    torch_a = torch.rand((B, 1, H, W), dtype=torch.bfloat16)
    torch_b = torch.rand((b, 1, H, W), dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=block_sharded_mem_config
    )
    input_tensor_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=interleaved_l1
    )

    output_tensor = ttnn.indexed_fill(
        batch_id_ttnn, input_tensor_a, input_tensor_b, memory_config=block_sharded_mem_config
    )
    assert tuple(output_tensor.shape) == (B, 1, H, W)
    assert output_tensor.layout == ttnn.TILE_LAYOUT
    logger.info(
        f"Indexed Fill (BLOCK_SHARDED+TILE) Output Shape: {output_tensor.shape}, "
        f"Memory Layout: {output_tensor.memory_config().memory_layout}"
    )

    golden = golden_indexed_fill(torch_a, torch_b, batch_id, dim=0)
    assert_with_pcc(golden, ttnn.to_torch(output_tensor), 0.9999)


@pytest.mark.parametrize(
    "shape_a, b, dim",
    [
        # Replace whole batches along dim=0 (the default).
        ((8, 1, 32, 32), 3, 0),
        # Replace 2 channel slices (dim=1).
        ((4, 6, 32, 32), 2, 1),
        # Replace 4 height slices (dim=2).
        ((4, 3, 8, 32), 4, 2),
        # Replace 8 columns (dim=3).
        ((2, 3, 4, 64), 8, 3),
        # Negative dim: -1 == rank-1 == 3.
        ((4, 3, 8, 32), 5, -1),
        # Negative dim: -2 == rank-2 == 2.
        ((4, 3, 8, 32), 3, -2),
    ],
    ids=["dim=0", "dim=1", "dim=2", "dim=3", "dim=-1", "dim=-2"],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    ids=["row_major", "tile"],
)
def test_indexed_fill_dim(device, shape_a, b, dim, layout):
    # input_b has the same shape as input_a except along `dim`, where its size
    # equals the number of indices (b).
    shape_b = list(shape_a)
    shape_b[dim] = b

    # batch_id selects which slices along `dim` of input_a get overwritten.
    batch_id = torch.randint(0, shape_a[dim], (1, 1, 1, b))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    torch_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_b = torch.rand(tuple(shape_b), dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=layout, device=device)
    input_tensor_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=layout, device=device)

    # Indexed fill along the requested dim.
    output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, dim=dim)

    # Output preserves input_a's shape and layout.
    assert tuple(output_tensor.shape) == shape_a
    assert output_tensor.layout == layout

    golden = golden_indexed_fill(torch_a, torch_b, batch_id, dim=dim)
    assert_with_pcc(golden, ttnn.to_torch(output_tensor), 0.9999)

    logger.info(
        f"Indexed Fill (dim={dim}, layout={layout}) Output Shape: {output_tensor.shape}, "
        f"Layout: {output_tensor.layout}"
    )


@pytest.mark.parametrize(
    "variant",
    ["interleaved_tile", "height_sharded"],
)
def test_indexed_fill_program_cache(device, variant):
    # Program-cache-hit correctness. The descriptor factory does not run create_descriptor()
    # again on a cache hit: per-core buffer addresses are patched via Buffer* bindings, and
    # the native/shard-local paths re-point the output-aliased CB (CBDescriptor::buffer) to
    # the new output buffer. Run the op twice with freshly allocated tensors — kept alive in
    # `held` so the allocator hands out DIFFERENT addresses on the second (cache-hit) run —
    # and verify both results are numerically correct and that the hit reuses the cached
    # program instead of building a new one. A stale-address bug would fail the second PCC.
    B, b, D, dim = 8, 3, 32, 0
    shape_a = (B, 1, 1, D)
    shape_b = (b, 1, 1, D)
    sharded = variant == "height_sharded"
    interleaved_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    sharded_mem_config = (
        ttnn.create_sharded_memory_config(
            shape=shape_a, core_grid=ttnn.CoreGrid(y=1, x=B), strategy=ttnn.ShardStrategy.HEIGHT
        )
        if sharded
        else None
    )
    layout = ttnn.ROW_MAJOR_LAYOUT if sharded else ttnn.TILE_LAYOUT

    held = []  # keep prior-iteration tensors alive so the cache-hit run gets new addresses
    entries = None
    for i in range(2):
        batch_id = torch.randint(0, B, (1, 1, 1, b))
        batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(device, interleaved_l1)
        torch_a = torch.rand(shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(shape_b, dtype=torch.bfloat16)
        if sharded:
            input_a = ttnn.from_torch(
                torch_a, dtype=ttnn.bfloat16, device=device, layout=layout, memory_config=sharded_mem_config
            )
            input_b = ttnn.from_torch(
                torch_b, dtype=ttnn.bfloat16, device=device, layout=layout, memory_config=interleaved_l1
            )
            output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_a, input_b, memory_config=sharded_mem_config)
        else:
            input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, device=device, layout=layout)
            input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, device=device, layout=layout)
            output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_a, input_b)

        golden = golden_indexed_fill(torch_a, torch_b, batch_id, dim=dim)
        assert_with_pcc(golden, ttnn.to_torch(output_tensor), 0.9999)
        held.extend([batch_id_ttnn, input_a, input_b, output_tensor])

        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries == 1, f"indexed_fill should cache exactly one program, got {entries}"
        else:
            assert (
                device.num_program_cache_entries() == entries
            ), "cache-hit run created a new program entry instead of reusing the cached one"


def test_indexed_fill_dim_out_of_bounds(device, expect_error):
    # Verify that a dim outside [-rank, rank) raises a fatal error.
    input_tensor_a = ttnn.rand((4, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.rand((2, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    batch_id = torch.randint(0, 4, (1, 1, 1, 2))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    with expect_error(RuntimeError, "is out of bounds for rank"):
        ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, dim=4)  # rank=4, so dim=4 is out of bounds

    with expect_error(RuntimeError, "is out of bounds for rank"):
        ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, dim=-5)  # -5 < -rank=-4
