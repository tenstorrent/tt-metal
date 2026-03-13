# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch

import ttnn

TILE_SIZE = 32


def pad_reference(input_torch, padded_shape, input_tensor_start, pad_value):
    """PyTorch reference for ttnn.pad using shape-based API."""
    rank = len(padded_shape)
    torch_padding = []
    for i in reversed(range(rank)):
        front = input_tensor_start[i]
        end = padded_shape[i] - input_torch.shape[i] - input_tensor_start[i]
        torch_padding.extend([front, end])
    return torch.nn.functional.pad(input_torch, torch_padding, mode="constant", value=pad_value)


def skip_if_tile_layout_incompatible(
    layout, tensor_shape, padded_shape, input_shard_shape=None, output_shard_shape=None
):
    """Skip test if tile layout is used but shapes aren't tile-aligned."""
    if layout != ttnn.TILE_LAYOUT:
        return
    for shard_shape in (input_shard_shape, output_shard_shape):
        if shard_shape is None:
            continue
        if shard_shape[-1] % TILE_SIZE != 0 or shard_shape[-2] % TILE_SIZE != 0:
            pytest.skip(f"TILE_LAYOUT requires shard shape last 2 dims to be multiples of TILE_SIZE, got {shard_shape}")


def assert_equal(expected, actual):
    assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
    assert torch.equal(expected, actual), "Data mismatch"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_shard_shape, output_shard_shape,shard_core_grid",
    [
        # Height-only end padding, same shard shape
        (
            [1, 1, 16, 64],
            [1, 1, 32, 64],
            [1, 1, 8, 64],
            None,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Width-only end padding, same shard shape
        (
            [1, 1, 16, 32],
            [1, 1, 16, 64],
            [1, 1, 8, 32],
            [1, 1, 8, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # N-dim padding
        (
            [2, 1, 16, 64],
            [4, 1, 16, 64],
            [1, 1, 16, 64],
            None,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Uneven input sharding (dim 0: 3 % 2 != 0)
        (
            [3, 1, 16, 64],
            [3, 1, 32, 64],
            [2, 1, 8, 64],
            [2, 1, 16, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Different input/output shard specs
        (
            [1, 1, 16, 64],
            [1, 1, 32, 64],
            [1, 1, 8, 64],
            [1, 1, 16, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Combined H and W padding, different output shard
        (
            [1, 1, 16, 32],
            [1, 1, 32, 64],
            [1, 1, 8, 32],
            [1, 1, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        # Uneven ND sharding, odd padding
        (
            [5, 4, 160, 160],
            [5, 4, 191, 187],
            [2, 3, 64, 96],
            [2, 3, 60, 40],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))}),
        ),
        # Uneven ND sharding, tile layout
        (
            [5, 4, 160, 160],
            [5, 4, 192, 192],
            [2, 3, 64, 96],
            [2, 3, 128, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))}),
        ),
    ],
)
@pytest.mark.parametrize("input_is_nd_sharded", [True, False])
@pytest.mark.parametrize("output_is_nd_sharded", [True, False])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_nd_sharded(
    device,
    dtype,
    tensor_shape,
    padded_shape,
    input_shard_shape,
    output_shard_shape,
    shard_core_grid,
    input_is_nd_sharded,
    output_is_nd_sharded,
    layout,
):
    effective_output_shard_shape = output_shard_shape if output_shard_shape is not None else input_shard_shape
    skip_if_tile_layout_incompatible(
        layout, tensor_shape, padded_shape, input_shard_shape, effective_output_shard_shape
    )
    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=effective_output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    if input_is_nd_sharded:
        input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    else:
        input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    if output_is_nd_sharded:
        output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    else:
        output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 10, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 10)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_shard_shape",
    [
        # ND-sharded input, height-only padding, interleaved output
        ([2, 1, 16, 64], [2, 1, 32, 64], [1, 1, 8, 64]),
        # ND-sharded input, combined padding, interleaved output
        ([1, 1, 16, 32], [1, 1, 32, 64], [1, 1, 8, 32]),
        # Uneven input sharding
        ([3, 1, 16, 64], [3, 1, 32, 64], [2, 1, 8, 64]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_nd_sharded_to_interleaved(
    device, dtype, tensor_shape, padded_shape, input_shard_shape, shard_core_grid, layout
):
    skip_if_tile_layout_incompatible(layout, tensor_shape, padded_shape, input_shard_shape=input_shard_shape)
    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, output_shard_shape",
    [
        # Interleaved input, height-only padding, ND-sharded output
        ([1, 1, 16, 64], [1, 1, 32, 64], [1, 1, 16, 64]),
        # Interleaved input, combined padding, ND-sharded output
        ([1, 1, 16, 32], [1, 1, 32, 64], [1, 1, 16, 32]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_interleaved_to_nd_sharded(
    device, dtype, tensor_shape, padded_shape, output_shard_shape, shard_core_grid, layout
):
    skip_if_tile_layout_incompatible(layout, tensor_shape, padded_shape, output_shard_shape=output_shard_shape)
    torch.manual_seed(42)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, input_shard_shape",
    [
        ([1, 1, 16, 64], [1, 1, 32, 64], [1, 1, 8, 64]),
        ([2, 1, 16, 64], [2, 1, 32, 64], [1, 1, 16, 64]),
        # Uneven ND sharding: dim 0 has 3 % 2 != 0
        ([3, 1, 16, 64], [3, 1, 32, 64], [2, 1, 8, 64]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_nd_sharded_to_legacy_sharded(
    device, dtype, tensor_shape, padded_shape, input_shard_shape, shard_core_grid, output_memory_layout, layout
):
    """pad: ND-sharded input -> legacy-sharded output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(padded_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= padded_shape[i]
    tensor_width = padded_shape[-1]

    height_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )
    shard_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": height_shard_shape},
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": width_shard_shape},
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": block_shard_shape},
    }
    layout_info = shard_layout_map[output_memory_layout]

    if any(d == 0 for d in layout_info["shard_shape"]):
        pytest.skip("Shard shape has a zero dimension")
    if layout_info["shard_shape"][0] * layout_info["shard_shape"][1] == 0:
        pytest.skip("Invalid shard shape")
    skip_if_tile_layout_incompatible(
        layout,
        tensor_shape,
        padded_shape,
        input_shard_shape=input_shard_shape,
        output_shard_shape=layout_info["shard_shape"],
    )

    output_shard_spec = ttnn.ShardSpec(
        layout_info["shard_grid"], layout_info["shard_shape"], ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape, output_shard_shape",
    [
        ([1, 1, 16, 64], [1, 1, 32, 64], [1, 1, 16, 64]),
        ([2, 1, 16, 64], [2, 1, 32, 64], [1, 1, 16, 64]),
        # Uneven ND sharding
        ([5, 4, 160, 160], [5, 4, 192, 192], [2, 3, 64, 96]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_legacy_sharded_to_nd_sharded(
    device, dtype, tensor_shape, padded_shape, output_shard_shape, shard_core_grid, input_memory_layout, layout
):
    """pad: legacy-sharded input -> ND-sharded output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    input_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    skip_if_tile_layout_incompatible(
        layout, tensor_shape, padded_shape, input_shard_shape=input_shard_shape, output_shard_shape=output_shard_shape
    )
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape",
    [
        ([1, 1, 16, 64], [1, 1, 32, 64]),
        ([2, 1, 16, 64], [2, 1, 32, 64]),
        ([3, 1, 128, 128], [3, 1, 150, 150]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_legacy_sharded_to_interleaved(device, dtype, tensor_shape, padded_shape, shard_core_grid, layout):
    """pad: legacy HEIGHT_SHARDED input -> interleaved output via default factory."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    input_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    skip_if_tile_layout_incompatible(layout, tensor_shape, padded_shape, input_shard_shape=input_shard_shape)
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, padded_shape",
    [
        ([3, 1, 128, 128], [3, 1, 192, 192]),
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))})],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_pad_interleaved_to_legacy_sharded(device, dtype, tensor_shape, padded_shape, shard_core_grid, layout):
    """pad: interleaved input -> legacy HEIGHT_SHARDED output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(padded_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= padded_shape[i]
    tensor_width = padded_shape[-1]

    output_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    skip_if_tile_layout_incompatible(layout, tensor_shape, padded_shape, output_shard_shape=output_shard_shape)
    output_shard_spec = ttnn.ShardSpec(shard_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_tensor_start = [0] * len(tensor_shape)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )
    output_ttnn_tensor = ttnn.pad(
        input_ttnn_tensor, padded_shape, input_tensor_start, 0, use_multicore=True, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    expected_torch = pad_reference(input_torch_tensor, padded_shape, input_tensor_start, 0)
    assert_equal(expected_torch, output_torch_tensor)
