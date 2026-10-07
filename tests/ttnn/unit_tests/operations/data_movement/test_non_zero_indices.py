# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal


def run_nonzero_and_validate(torch_input, ttnn_input, device):
    """Run ttnn.nonzero, extract count + indices, compare to torch.nonzero() reference."""
    output = ttnn.nonzero(ttnn_input)

    # output_0: count tensor [1, 1, 1, 8]
    count_tensor = ttnn.to_torch(ttnn.from_device(output[0]))
    count = int(count_tensor[0, 0, 0, 0].item())

    # output_1: [1, 1, 1, N*4] where each non-zero has a (b,n,h,c) 4-tuple of uint32 values
    indices_tensor = ttnn.to_torch(ttnn.from_device(output[1]))
    tt_indices = indices_tensor[0, 0, 0, : count * 4].reshape(count, 4).int()

    ref_indices = torch.nonzero(torch_input, as_tuple=False).int()

    assert count == ref_indices.shape[0], f"Count mismatch: got {count}, expected {ref_indices.shape[0]}"
    assert_equal(tt_indices, ref_indices)


def make_row_distinct_zero_pattern(shape):
    """Dense data whose zero pattern is unique to each row.

    The other sharded tests zero every 2nd or 3rd flat element of an even-width
    tensor, which gives every row an identical zero pattern.  That is invariant
    under a row mix-up, so a reader that fetches the right column-shard of the
    wrong row still produces the expected indices.

    Row r zeroes the columns given by the set bits of r, which is unique per row
    for any row count up to 2**width.  Row 0 is left zero-free.  The assertion
    below is the helper's contract: pairwise-distinct patterns are what make a
    row mix-up — including a whole-batch swap — change the emitted (b, n, h, c)
    tuples, so a shape that cannot satisfy it must fail loudly rather than
    silently weaken every case built on it.
    """
    torch_input = torch.arange(1, math.prod(shape) + 1, dtype=torch.bfloat16).reshape(shape)
    rows = torch_input.reshape(-1, shape[-1])
    num_rows, width = rows.shape
    for r in range(num_rows):
        for k in range(width):
            if (r >> k) & 1:
                rows[r, k] = 0

    patterns = {tuple((rows[r] == 0).tolist()) for r in range(num_rows)}
    assert len(patterns) == num_rows, f"zero pattern repeats: {len(patterns)} distinct across {num_rows} rows"
    return torch_input


def make_ttnn_tensor(torch_tensor, layout, device, mem_config=None):
    dtype = ttnn.bfloat16
    if mem_config is None:
        return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device)
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=mem_config)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 32])),
        (torch.Size([1, 1, 1, 64])),
    ),
)
def test_non_zero_indices_ttnn(input_shapes, device):
    torch.manual_seed(0)

    torch_input_tensor = torch.ones(input_shapes)
    torch_input_tensor[..., ::2] = 0

    ref_indices = torch.nonzero(torch_input_tensor, as_tuple=False).int()  # [K, 4]

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.nonzero(input_tensor, queue_id=0)

    output_tensor1 = ttnn.to_layout(output_tensor[0], ttnn.ROW_MAJOR_LAYOUT)
    output_tensor1 = ttnn.from_device(output_tensor1)
    output_tensor1 = ttnn.to_torch(output_tensor1)
    no_of_non_zero_indices = int(output_tensor1[..., 0].item())

    output_tensor2 = ttnn.to_layout(output_tensor[1], ttnn.ROW_MAJOR_LAYOUT)
    output_tensor2 = ttnn.from_device(output_tensor2)
    output_tensor2 = ttnn.to_torch(output_tensor2)
    tt_indices = output_tensor2[0, 0, 0, : no_of_non_zero_indices * 4].reshape(no_of_non_zero_indices, 4).int()

    assert_equal(ref_indices, tt_indices)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_nonzero(
    device,
    reset_seeds,
):
    torch_input = torch.tensor([[[[0, 4, 0, 2, 4, 0, 3]]]])

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device)
    for i in range(6):
        output_indices, output_tensor = ttnn.nonzero(ttnn_input)
        ttnn.deallocate(output_indices)
        ttnn.deallocate(output_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 4, 8],
        [1, 2, 4, 8],
        [2, 1, 4, 8],
        [2, 3, 4, 5],
    ],
)
def test_nonzero_multi_dim_row_major(shape, device):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::3] = 0

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 32],
        [1, 1, 1, 64],
        [1, 1, 32, 32],
        [1, 1, 32, 64],
        [2, 1, 32, 64],
    ],
)
def test_nonzero_tile_layout(shape, device):
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::4] = 0

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,num_cores",
    [
        ([1, 1, 4, 8], 4),  # 4 rows sharded across 4 cores
        ([1, 1, 8, 16], 4),  # 8 rows across 4 cores (2 rows each)
    ],
)
def test_nonzero_height_sharded_row_major(shape, num_cores, device):
    torch.manual_seed(1)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::2] = 0

    total_rows = shape[0] * shape[1] * shape[2]
    shard_height = total_rows // num_cores
    shard_width = shape[-1]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,num_cores",
    [
        ([1, 1, 64, 32], 2),  # 64 rows → 2 tile-rows per shard (32 rows each)
        ([1, 1, 64, 64], 2),
    ],
)
def test_nonzero_height_sharded_tile(shape, num_cores, device):
    torch.manual_seed(2)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::3] = 0

    shard_height = shape[-2] // num_cores
    shard_width = shape[-1]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.TILE_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,num_cores",
    [
        ([1, 1, 1, 32], 2),
        ([1, 1, 1, 64], 4),
    ],
)
def test_nonzero_width_sharded_row_major(shape, num_cores, device):
    torch.manual_seed(3)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::2] = 0

    total_rows = shape[0] * shape[1] * shape[2]
    shard_height = total_rows
    shard_width = shape[-1] // num_cores

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,num_cores",
    [
        ([1, 1, 32, 64], 2),
        ([1, 1, 32, 128], 4),
    ],
)
def test_nonzero_width_sharded_tile(shape, num_cores, device):
    torch.manual_seed(4)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::3] = 0

    shard_height = shape[-2]
    shard_width = shape[-1] // num_cores

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.TILE_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,grid_shape",
    [
        ([1, 1, 4, 8], (2, 2)),  # 4 rows × 8 cols → 2×2 grid, shard [2, 4]
        ([1, 1, 8, 16], (2, 2)),  # shard [4, 8]
    ],
)
def test_nonzero_block_sharded_row_major(shape, grid_shape, device):
    torch.manual_seed(5)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::2] = 0

    grid_h, grid_w = grid_shape
    total_rows = shape[0] * shape[1] * shape[2]
    shard_height = total_rows // grid_h
    shard_width = shape[-1] // grid_w

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,grid_shape",
    [
        ([1, 1, 64, 64], (2, 2)),  # shard [32, 32] (one tile per core)
        ([1, 1, 64, 128], (2, 2)),  # shard [32, 64]
    ],
)
def test_nonzero_block_sharded_tile(shape, grid_shape, device):
    torch.manual_seed(6)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input.flatten()[::4] = 0

    grid_h, grid_w = grid_shape
    shard_height = shape[-2] // grid_h
    shard_width = shape[-1] // grid_w

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.TILE_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,grid_shape",
    [
        ([1, 1, 4, 8], (2, 2)),
        ([1, 1, 8, 16], (2, 2)),
    ],
)
def test_nonzero_block_sharded_col_major_row_major(shape, grid_shape, device):
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    # bfloat16 -0.0 (0x8000) is mathematically zero so torch.nonzero skips it,
    # but the kernel's bitwise != 0 would count it. Remove all exact zeros from
    # the random data so only the explicitly masked positions are zero.
    torch_input[torch_input == 0] = 1.0
    torch_input.flatten()[::2] = 0

    grid_h, grid_w = grid_shape
    total_rows = shape[0] * shape[1] * shape[2]
    shard_height = total_rows // grid_h
    shard_width = shape[-1] // grid_w

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.COL_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,num_cores",
    [
        ([1, 1, 1, 32], 2),
        ([1, 1, 1, 64], 4),
    ],
)
def test_nonzero_width_sharded_col_major_row_major(shape, num_cores, device):
    torch.manual_seed(8)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input[torch_input == 0] = 1.0
    torch_input.flatten()[::2] = 0

    total_rows = shape[0] * shape[1] * shape[2]
    shard_height = total_rows
    shard_width = shape[-1] // num_cores

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.COL_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize(
    "shape,grid_shape",
    [
        ([1, 1, 64, 64], (2, 2)),
        ([1, 1, 64, 128], (2, 2)),
    ],
)
def test_nonzero_block_sharded_col_major_tile_layout(shape, grid_shape, device):
    torch.manual_seed(9)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_input[torch_input == 0] = 1.0
    torch_input.flatten()[::3] = 0

    grid_h, grid_w = grid_shape
    shard_height = shape[-2] // grid_h
    shard_width = shape[-1] // grid_w

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))})
    shard_spec = ttnn.ShardSpec(grid, [shard_height, shard_width], ttnn.ShardOrientation.COL_MAJOR)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.TILE_LAYOUT, device, mem_config)
    run_nonzero_and_validate(torch_input, ttnn_input, device)


@pytest.mark.parametrize("size", [32, 64, 128])
def test_nonzero_backward_compat_1d(size, device):
    """1D-like [1, 1, 1, X] ROW_MAJOR INTERLEAVED: output matches torch.nonzero() [K, 4] format."""
    torch.manual_seed(0)
    torch_input = torch.ones([1, 1, 1, size], dtype=torch.bfloat16)
    torch_input[..., ::2] = 0

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output = ttnn.nonzero(ttnn_input)
    count_tensor = ttnn.to_torch(ttnn.from_device(output[0]))
    count = int(count_tensor[0, 0, 0, 0].item())
    indices_tensor = ttnn.to_torch(ttnn.from_device(output[1]))
    tt_indices = indices_tensor[0, 0, 0, : count * 4].reshape(count, 4).int()

    ref_indices = torch.nonzero(torch_input, as_tuple=False).int()
    assert count == ref_indices.shape[0]
    assert_equal(tt_indices, ref_indices)


@pytest.mark.parametrize(
    "shape,grid_shape,shard,memory_layout,orientation",
    [
        # Shard shapes are given explicitly rather than derived: under COL_MAJOR the shard grid
        # is transposed onto the core grid (TensorSpec requires shards_h <= grid_w and
        # shards_w <= grid_h), so the same core grid needs a different shard than ROW_MAJOR.
        #
        # [1, 1, 4, 8] on 2x2 / shard [2, 4]: the configuration from the bug report. grid_h,
        # grid_w, shard_height and pages-per-shard-row are all 2 here, so it pins the reported
        # case but cannot tell those four quantities apart.
        ([1, 1, 4, 8], (2, 2), [2, 4], ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.ROW_MAJOR),
        ([1, 1, 4, 8], (2, 2), [2, 4], ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.COL_MAJOR),
        # Non-square core grids in both aspect ratios, so a transposed row/column index changes
        # the answer instead of cancelling out.
        ([1, 1, 8, 16], (4, 2), [2, 8], ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.ROW_MAJOR),
        ([1, 1, 8, 16], (2, 4), [4, 4], ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.ROW_MAJOR),
        # Non-square grid with grid_h > 1 under COL_MAJOR: 4x2 shards tiling a 2-row x 4-col
        # grid. Here the orientation genuinely decides which bank holds each page, which is
        # what substantiates keeping shard orientation out of the page-id computation.
        ([1, 1, 8, 16], (2, 4), [2, 8], ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.COL_MAJOR),
        # WIDTH_SHARDED with H > 1: one core row, but still several page-rows per shard.
        # ROW_MAJOR only — with grid_h == 1 both orientations enumerate the single core row
        # identically, so a COL_MAJOR case here would be a duplicate.
        ([1, 1, 4, 8], (1, 2), [4, 4], ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ShardOrientation.ROW_MAJOR),
        # Multi-batch: exercises the (b, n) decomposition alongside the sharded page walk.
        ([2, 2, 4, 8], (2, 2), [8, 4], ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.ROW_MAJOR),
    ],
)
def test_nonzero_sharded_row_major_row_distinct_pattern(shape, grid_shape, shard, memory_layout, orientation, device):
    """Row-major-layout WIDTH/BLOCK-sharded input, both shard orientations, with more than
    one page-row per shard.

    Note the two senses of "row major" here: the tensor layout is always ttnn.ROW_MAJOR_LAYOUT,
    while `orientation` is the shard orientation and is COL_MAJOR in some cases below.

    Regression test for the reader computing a bank-major device page index instead of the
    logical row-major page index that TensorAccessor expects.  Both formulas agree only when
    there is one column shard per row or one page-row per shard, so this needs H > 1 and more
    than one shard per row.
    """
    torch_input = make_row_distinct_zero_pattern(shape)

    grid_h, grid_w = grid_shape
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))})
    shard_spec = ttnn.ShardSpec(grid, shard, orientation)
    mem_config = ttnn.MemoryConfig(
        memory_layout=memory_layout,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    ttnn_input = make_ttnn_tensor(torch_input, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)

    # Guard the guard: if the input ever lands interleaved, shards_per_row collapses to 1 and
    # page_id == r, so every case here would pass while exercising none of the sharded path.
    # Shape and orientation are checked too: shards_per_row is derived from the shard width,
    # and a normalised orientation would silently turn the COL_MAJOR cases into duplicates of
    # their ROW_MAJOR twins — those cases are the whole justification for keeping orientation
    # out of the page-id computation.
    actual = ttnn_input.memory_config()
    assert actual.memory_layout == memory_layout, f"input landed as {actual.memory_layout}, expected {memory_layout}"
    assert actual.shard_spec is not None, "input landed without a shard spec"
    assert actual.shard_spec.shape == shard, f"shard landed as {actual.shard_spec.shape}, expected {shard}"
    assert (
        actual.shard_spec.orientation == orientation
    ), f"orientation landed as {actual.shard_spec.orientation}, expected {orientation}"

    run_nonzero_and_validate(torch_input, ttnn_input, device)
