# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_allclose

TILE_HEIGHT = 32
TILE_WIDTH = 32


def _residual_ht_shapes(device):
    """
    Build Ht values where Ht % grid_x != 0 and grid_x <= Ht < grid_x * grid_y.
    This is the exact condition that enters the additional CoreRange branch
    in SingleRowSingleCore / SingleRowMultiCore sort program factories.
    """
    grid = device.compute_with_storage_grid_size()
    total = grid.x * grid.y
    shapes = []
    for r in range(1, grid.x):
        ht = grid.x + r
        if ht < total:
            shapes.append(ht)
    if grid.x * 2 + 1 < total:
        shapes.append(grid.x * 2 + 1)
    return shapes


def test_sort_residual_core_range(device):
    """
    Regression test for an off-by-one in the additional CoreRange end
    coordinate that allocated one extra core, causing OOB DRAM writes.
    Shapes are derived from the device grid so the path is always hit.
    """
    ht_values = _residual_ht_shapes(device)
    for ht in ht_values:
        for descending in (False, True):
            torch.manual_seed(0)
            shape = [ht * TILE_HEIGHT, TILE_WIDTH]
            input_tensor = torch.randn(shape, dtype=torch.bfloat16)

            ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
            torch_values, _ = torch.sort(input_tensor, dim=-1, descending=descending)
            ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)

            assert list(ttnn_values.shape) == shape
            assert_equal(torch_values, ttnn.to_torch(ttnn_values))
            ttnn_gathered = torch.gather(input_tensor, -1, ttnn.to_torch(ttnn_indices).to(torch.int64))
            assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, False),
        ([32, 128], -1, False),
        ([1, 1, 32, 64], -1, True),
        ([32, 128], 1, True),
        ([1], 0, True),
        ([], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
        ([1, 1, 512, 64], -1, False),
        ([1, 1, 2112, 64], -1, False),
        ([1, 64, 64], 0, False),
        ([1, 64, 64], 1, True),
        ([1, 64, 64], 2, False),
        ([1, 64], 0, False),
        ([1, 64], 1, True),
        ([237], 0, False),
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_sort_standard(shape, dim, descending, device, torch_dtype, ttnn_dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn_dtype, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
        assert torch_sort_indices == ttnn.to_torch(ttnn_sort_indices).to(torch.int64)
    else:
        # Validate sorted values
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values, dtype=torch_dtype))

        # Validate that the indices correctly index into the original tensor
        ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn.to_torch(ttnn_sort_indices).to(torch.int64))
        assert_equal(torch_sort_values, ttnn_torch_gather_from_indices)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, False),
        ([32, 128], -1, False),
        ([1, 1, 32, 64], -1, True),
        ([32, 128], 1, True),
        ([1], 0, True),
        ([], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
        ([1, 1, 512, 64], -1, False),
        ([1, 1, 2112, 64], -1, False),
    ],
)
def test_sort_prealocated_output(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    ttnn_sort_values = ttnn.zeros_like(ttnn_input)
    ttnn_sort_indices = ttnn.zeros_like(ttnn_input, dtype=ttnn.uint16)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(ttnn_sort_values, ttnn_sort_indices))

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([1, 1, 1, 2 * TILE_WIDTH], -1, False),
        ([1, 1, 1, 8192 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 96 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
        ([1, 151936], -1, False),
        ([1, 128256], -1, False),
        ([1, 16384 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_long_tensor(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 96], -1, True),
        ([1, 1, 32, 96 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_l1_memory_tensor(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        input,
        ttnn.bfloat16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([32, 128], -1, True),
        ([1, 1, 32, 128 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_program_cache(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    test_iterations = 3
    for _ in range(test_iterations):
        # Run the sort operation multiple times to fill the program cache
        with device.cache_entries_counter.measure():
            ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)
        ttnn_sort_values_torch = ttnn.to_torch(ttnn_sort_values)

        assert torch_sort_values.shape == ttnn_sort_values.shape
        assert torch_sort_indices.shape == ttnn_sort_indices.shape

        assert list(ttnn_sort_values.shape) == shape
        assert list(ttnn_sort_indices.shape) == shape

        assert_equal(torch_sort_values, ttnn_sort_values_torch)
        ttnn.synchronize_device(device)
    device.disable_and_clear_program_cache()
    assert (
        device.cache_entries_counter.total == 1
    ), "Expected only one program cache entry for sort operation, but found {}".format(
        device.cache_entries_counter.total
    )


@pytest.mark.parametrize(
    "shape, dim, descending, torch_value_dtype, ttnn_value_dtype, ttnn_index_dtype",
    [
        ([32, 64], -1, False, torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        ([32, 64], -1, False, torch.bfloat16, ttnn.bfloat16, ttnn.uint32),
        ([32, 64], -1, False, torch.uint8, ttnn.uint16, ttnn.uint16),
        ([32, 64], -1, False, torch.uint8, ttnn.uint16, ttnn.uint32),
        # ([1, 8], -1, False, torch.uint8, ttnn.uint16, ttnn.uint16), # GH issue: #33473
    ],
)
def test_sort_datatypes(shape, dim, descending, torch_value_dtype, ttnn_value_dtype, ttnn_index_dtype, device):
    torch.manual_seed(0)

    if torch_value_dtype == torch.uint8 or torch_value_dtype == torch.int16:
        input = torch.randint(100, shape, dtype=torch_value_dtype)
    else:
        input = torch.randn(shape, dtype=torch_value_dtype)
    ttnn_input = ttnn.from_torch(input, ttnn_value_dtype, layout=ttnn.Layout.TILE, device=device)

    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    ttnn_sort_values = ttnn.zeros_like(ttnn_input, dtype=ttnn_value_dtype)
    ttnn_sort_indices = ttnn.zeros_like(ttnn_input, dtype=ttnn_index_dtype)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(ttnn_sort_values, ttnn_sort_indices))

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values, dtype=torch_value_dtype))


def create_descending_tensor(shape, dim, dtype=torch.bfloat16):
    size_along_dim = shape[dim]

    # Step 1: Create descending range [size-1, size-2, ..., 0]
    descending_values = torch.arange(size_along_dim - 1, -1, -1, dtype=dtype)

    # Step 2: Reshape to fit into the target dimension with unsqueeze
    view_shape = [1] * len(shape)
    view_shape[dim] = size_along_dim
    descending_values = descending_values.view(*view_shape)

    # Step 3: Broadcast to full shape
    descending_tensor = descending_values.expand(*shape)

    return descending_tensor


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
    ],
)
def test_sort_indices(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = create_descending_tensor(shape, dim, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    torch_converted_indices = ttnn.to_torch(ttnn_sort_indices).to(torch.int64)

    assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))
    assert_allclose(torch_sort_indices.to(torch.int64), torch_converted_indices)


@pytest.mark.parametrize(
    "shape, dim, descending, torch_dtype, ttnn_dtype",
    [
        ([64, 64], -1, False, torch.bfloat16, ttnn.bfloat16),
        ([32, 128], -1, True, torch.bfloat16, ttnn.bfloat16),
        ([1, 1, 32, 64], -1, False, torch.float32, ttnn.float32),
        ([1, 55, 43], -1, True, torch.bfloat16, ttnn.bfloat16),
        ([32, 128], 1, False, torch.bfloat16, ttnn.bfloat16),
        ([1, 64, 64], 0, True, torch.bfloat16, ttnn.bfloat16),
        ([1, 64, 64], 1, False, torch.float32, ttnn.float32),
        ([1, 1, 64, 64], 0, False, torch.float32, ttnn.float32),
        ([237], 0, False, torch.bfloat16, ttnn.bfloat16),
    ],
)
def test_sort_row_major_layout(shape, dim, descending, torch_dtype, ttnn_dtype, device):
    torch.manual_seed(0)

    input_t = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(input_t, ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_values, torch_indices = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert ttnn_values.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "Output layout must be ROW_MAJOR"
    assert ttnn_indices.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "Index layout must be ROW_MAJOR"

    assert list(ttnn_values.shape) == shape
    assert list(ttnn_indices.shape) == shape

    out_vals = ttnn.to_torch(ttnn_values, dtype=torch_dtype)
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    # For non-last-dim fp32, the composite layer wraps the device sort in a pair of
    # ttnn::transpose calls.  The RM transpose compute kernel routes data through DEST
    # (which holds ~10-bit mantissa on Wormhole even with fp32_dest_acc_en=true), so
    # fp32 values pick up ~1 TF32 ULP of error per hop and pairs of near-equal inputs
    # may swap places.  bf16's 7-bit mantissa is coarser than DEST, so bf16 stays
    # bit-exact and we keep the strict check there.
    is_dim_last_idx = (dim == -1) or (dim == len(shape) - 1)
    if torch_dtype == torch.float32 and not is_dim_last_idx:
        assert_allclose(torch_values, out_vals, rtol=1e-2, atol=1e-2)
        assert_allclose(torch_values, ttnn_gathered, rtol=1e-2, atol=1e-2)
    else:
        assert_equal(torch_values, out_vals)
        assert_equal(torch_values, ttnn_gathered)


def _make_sharded_cfg(memory_layout, grid_end_x, grid_end_y, shard_h, shard_w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_end_x, grid_end_y))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
        ([8 * TILE_HEIGHT, TILE_WIDTH * 2], -1, False),
    ],
)
def test_sort_sharded_input(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    shard_height = TILE_HEIGHT
    shard_width = shape[-1]
    sharded_cfg = _make_sharded_cfg(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, shard_height, shard_width
    )

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        input_t,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_cfg,
    )

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_values.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
        ([8 * TILE_HEIGHT, TILE_WIDTH * 2], -1, False),
    ],
)
def test_sort_sharded_output(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    shard_height = TILE_HEIGHT
    shard_width = shape[-1]
    sharded_cfg = _make_sharded_cfg(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, shard_height, shard_width
    )

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        input_t,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending, memory_config=sharded_cfg)

    assert ttnn_values.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert list(ttnn_values.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
    ],
)
def test_sort_row_major_sharded(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    shard_height = TILE_HEIGHT
    shard_width = shape[-1]
    sharded_cfg = _make_sharded_cfg(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, shard_height, shard_width
    )

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        input_t,
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_cfg,
    )

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_values.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
        ([1, 1, 32, 64], -1, False),
    ],
)
def test_sort_preallocated_row_major_outputs(shape, dim, descending, device):
    torch.manual_seed(0)

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_vals = ttnn.zeros(shape, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_idx = ttnn.zeros(shape, dtype=ttnn.uint16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(out_vals, out_idx))

    assert out_vals.get_layout() == ttnn.ROW_MAJOR_LAYOUT
    assert out_idx.get_layout() == ttnn.ROW_MAJOR_LAYOUT
    assert list(out_vals.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(out_vals))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(out_idx).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([8 * TILE_HEIGHT, TILE_WIDTH * 2], -1, True),
    ],
)
def test_sort_preallocated_sharded_outputs(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    sharded_cfg = _make_sharded_cfg(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, TILE_HEIGHT, shape[-1])

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_vals = ttnn.zeros(shape, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)
    out_idx = ttnn.zeros(shape, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(out_vals, out_idx))

    assert out_vals.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert list(out_vals.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(out_vals))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(out_idx).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], 0, False),
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], 2, True),
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], 3, False),
    ],
)
def test_sort_rank5_all_dims(shape, dim, descending, device):
    """Rank > 4 with sort dim ranging over all logical positions.

    The composite layer permutes the sort dim to the last position, squeezes
    leading dims into 4D, runs the kernel, then restores the original rank
    and order.  The rank-restoration reshape targets the *transposed* shape,
    keeping the last dim unchanged, so it routes through ttnn::reshape's
    `this_is_view` fast path (metadata-only `view`).  This avoids the device
    reshape kernel entirely — important because that kernel rejects UINT16.
    """
    torch.manual_seed(0)

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_values.shape) == shape
    assert list(ttnn_indices.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ([4 * TILE_HEIGHT, 2 * TILE_WIDTH], 0),
        ([2, 3 * TILE_HEIGHT, 2 * TILE_WIDTH], 1),
    ],
)
def test_fp32_non_last_dim_index_validation(shape, dim, device):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    ref_vals, _ = torch.sort(t, dim=dim)

    x = ttnn.from_torch(
        t, dtype=ttnn.float32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    v, i = ttnn.sort(x, dim=dim)

    assert i.dtype == ttnn.uint32, f"FP32 input must produce UINT32 indices, got {i.dtype}"

    out_vals = ttnn.to_torch(v).float()
    out_idx = ttnn.to_torch(i).to(torch.int64)

    # The kernel pipeline routes fp32 through DEST (which holds ~10-bit mantissa even
    # with fp32_dest_acc_en=true on Wormhole) during the bitonic merge stage. This
    # causes two kinds of small deviations vs torch.sort:
    #   * sorted-values output: per-element ATOL up to ~1 TF32 ULP at the value's
    #     magnitude (~4e-3 for values around 4).
    #   * returned indices: pairs of inputs that differ by less than the kernel's
    #     precision can swap their relative sort order, so a returned index may point
    #     to a neighbour of the position torch.sort picked — which is still "correct"
    #     up to the same ULP tolerance when we gather from the original fp32 input.
    assert_allclose(out_vals, ref_vals, rtol=1e-2, atol=1e-2)

    gathered = torch.gather(t, dim, out_idx)
    assert_allclose(gathered.float(), ref_vals.float(), rtol=1e-2, atol=1e-2)


def test_fp32_input_uint16_preallocated_index_rejected(device, expect_error):
    shape = [TILE_HEIGHT, 2 * TILE_WIDTH]
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(
        t, dtype=ttnn.float32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )
    out_v = ttnn.zeros(
        shape, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_i = ttnn.zeros(
        shape, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with expect_error(RuntimeError, "must be UINT32 when input dtype is FLOAT32"):
        ttnn.sort(x, dim=-1, out=(out_v, out_i))


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_width_sharded(layout, device):
    torch.manual_seed(0)
    shape = [2 * TILE_HEIGHT, 4 * TILE_WIDTH]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref_vals, _ = torch.sort(t, dim=-1)

    cfg = _make_sharded_cfg(ttnn.TensorMemoryLayout.WIDTH_SHARDED, 3, 0, 2 * TILE_HEIGHT, TILE_WIDTH)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, memory_config=cfg, layout=layout)
    v, i = ttnn.sort(x, dim=-1)

    out = ttnn.to_torch(v).float()
    assert list(v.shape) == shape
    assert torch.allclose(
        out, ref_vals.float(), rtol=1e-2, atol=1e-2
    ), f"values mismatch max_diff={(out - ref_vals.float()).abs().max():.4f}"
    gathered = torch.gather(t, -1, ttnn.to_torch(i).to(torch.int64))
    assert torch.allclose(gathered.float(), ref_vals.float(), rtol=1e-2, atol=1e-2), "index gather mismatch"


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@pytest.mark.timeout(600, method="thread")
@pytest.mark.parametrize("descending", [False, True])
def test_sort_multi_row_multi_core_no_deadlock(descending, device):
    """
    Guard for the DRAM multi-core sort path (SortProgramFactorySingleRowMultiCore).

    The coordinator core collects two logically distinct worker signals -- the reader's
    per-row "ready" and the writer's per-pair "done" -- on two separate cores->coordinator
    semaphores, one producer signal each.  They are kept separate so each coordinator wait
    has an exact, monotonic per-producer target: were both folded onto one shared counter,
    at a tile-row boundary (Ht >= 2) a fast reader's next-row "ready" increment could push
    the counter past the "done" target an exact-match wait is looking for, stranding the
    wait and deadlocking the op.

    This exercises the multi-core Ht >= 2 path -- otherwise only covered at Ht == 1 -- and
    checks it runs to completion with correct output.  It is not a deterministic deadlock
    reproducer: the mismatch a shared counter would cause is timing-dependent (the exact-match
    poll normally out-races the NoC atomics), so it needs timing pressure to surface.

    The worker-thread watchdog + pytest-timeout are a best-effort regression guard,
    not a clean recovery mechanism: a genuine deadlock wedges the device (which needs
    a reset regardless), so the join times out and the assertion below fires, but the
    function-scoped `device` fixture teardown (close_device) may then block until the
    process-level pytest-timeout terminates the run.  The guarantee is only that a
    regression surfaces as a CI *failure* (never a silent pass); cleanly isolating a
    hang from teardown would require running the op in a killable subprocess.

    The multi-core factory is only selected when the (power-of-two padded) tile width
    exceeds total_cores * 128, so the width is sized from the device grid.
    """
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    wt = _next_pow2(total_cores * 128 + 1)  # smallest pow2 Wt on the DRAM multi-core path
    shape = [1, 1, 2 * TILE_HEIGHT, wt * TILE_WIDTH]  # Ht = 2

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    result = {}

    def _run():
        try:
            values, indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)
            ttnn.synchronize_device(device)
            result["values"] = values
            result["indices"] = indices
        except Exception as exc:  # surface device/compile errors to the main thread
            result["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=300.0)

    assert not worker.is_alive(), (
        "ttnn.sort did not complete on the DRAM multi-core path (Ht=2): the coordinator's "
        "cores->coordinator wait was starved -- likely a regression of the ready/done "
        "semaphore split."
    )
    if "error" in result:
        raise result["error"]

    torch_values, _ = torch.sort(input_t, dim=-1, descending=descending)
    assert list(result["values"].shape) == shape
    assert_equal(torch_values, ttnn.to_torch(result["values"]))
    ttnn_gathered = torch.gather(input_t, -1, ttnn.to_torch(result["indices"]).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_block_sharded(layout, device):
    torch.manual_seed(0)
    shape = [2 * TILE_HEIGHT, 4 * TILE_WIDTH]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref_vals, _ = torch.sort(t, dim=-1)

    cfg = _make_sharded_cfg(ttnn.TensorMemoryLayout.BLOCK_SHARDED, 1, 1, TILE_HEIGHT, 2 * TILE_WIDTH)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, memory_config=cfg, layout=layout)
    v, i = ttnn.sort(x, dim=-1)

    out = ttnn.to_torch(v).float()
    assert list(v.shape) == shape
    assert torch.allclose(
        out, ref_vals.float(), rtol=1e-2, atol=1e-2
    ), f"values mismatch max_diff={(out - ref_vals.float()).abs().max():.4f}"
    gathered = torch.gather(t, -1, ttnn.to_torch(i).to(torch.int64))
    assert torch.allclose(gathered.float(), ref_vals.float(), rtol=1e-2, atol=1e-2), "index gather mismatch"
