# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

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
