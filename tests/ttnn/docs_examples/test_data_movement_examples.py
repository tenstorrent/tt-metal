# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


def test_concat(device):
    # Create two tensors to concatenate
    tensor1 = ttnn.rand([1, 1, 64, 32], dtype=ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand([1, 1, 64, 32], dtype=ttnn.bfloat16, device=device)

    # Concatenate along dimension 3
    concatenated_tensor = ttnn.concat([tensor1, tensor2], dim=3)
    logger.info(
        "Concatenated Tensor Shape:", concatenated_tensor.shape
    )  # Concatenated Tensor Shape: Shape([1, 1, 64, 64])


def test_nonzero(device):
    # Create a tensor with some zero and non-zero elements
    input_tensor = torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)
    input_tensor_ttnn = ttnn.from_torch(input_tensor, device=device)

    # Find indices of non-zero elements
    nonzero_indices = ttnn.nonzero(input_tensor_ttnn)
    logger.info("Non-zero Indices:", nonzero_indices)


def test_pad(device):
    # Create a tensor to pad
    input_tensor = ttnn.rand((1, 1, 4, 4), dtype=ttnn.bfloat16, device=device)

    # Pad the tensor with 1 zero on all sides
    padded_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 12), (0, 12)], 0)
    logger.info("Padded Tensor Shape:", padded_tensor.shape)  # Padded Tensor Shape: Shape([1, 1, 16, 16])


def test_permute(device):
    # Create a tensor to permute
    input_tensor = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, device=device)

    # Permute the tensor dimensions
    permuted_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    logger.info("Permuted Tensor Shape:", permuted_tensor.shape)  # Permuted Tensor Shape: Shape([1, 1, 32, 64])


def test_reshape(device):
    # Create a tensor to reshape
    input_tensor = torch.arange(4, dtype=torch.bfloat16)
    input_tensor_tt = ttnn.from_torch(input_tensor, device=device)

    # Reshape the tensor
    reshaped_tensor = ttnn.reshape(input_tensor_tt, (1, 1, 2, 2))
    logger.info("Reshaped Tensor Shape:", reshaped_tensor.shape)  # Reshaped Tensor Shape: Shape([1, 1, 2, 2])


def test_scatter(device):
    # Create input, index, and source tensors
    input_torch = torch.randn([10, 20, 30, 20, 10], dtype=torch.float32)
    index_torch = torch.randint(0, 10, [10, 20, 30, 20, 5], dtype=torch.int64)
    source_torch = torch.randn([10, 20, 30, 20, 10], dtype=input_torch.dtype)

    input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    index_ttnn = ttnn.from_torch(index_torch, dtype=ttnn.int32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    source_ttnn = ttnn.from_torch(source_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    dim = -1

    # Perform scatter operation
    output = ttnn.scatter(input_ttnn, dim, index_ttnn, source_ttnn)
    logger.info(f"Scatter operation result: {output}")


def test_scatter_add(device):
    # Create input, index, and source tensors for scatter_add
    input_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    index_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    source_torch = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

    input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.float32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    index_ttnn = ttnn.from_torch(index_torch, dtype=ttnn.int32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    source_ttnn = ttnn.from_torch(source_torch, dtype=ttnn.float32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    dim = 1

    # Perform scatter_add operation (adds source values to input at specified indices)
    output = ttnn.scatter_add(input_ttnn, dim, index_ttnn, source_ttnn)
    logger.info(f"Scatter add operation result: {output}")


def test_repeat(device):
    # Create a tensor to repeat
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    input_tensor_tt = ttnn.from_torch(input_tensor, device=device)

    # Repeat the tensor along specified dimensions
    repeated_tensor = ttnn.repeat(input_tensor_tt, (1, 2))
    logger.info("Repeated Tensor Shape:", repeated_tensor.shape)  # Repeated Tensor Shape: Shape([2, 4])


def test_repeat_interleave(device):
    # Create a tensor to repeat interleave
    input_tensor = ttnn.rand((10, 10), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Repeat interleave the tensor
    repeated_tensor = ttnn.repeat_interleave(input_tensor, repeats=2, dim=0)
    logger.info(
        "Repeat Interleave Tensor Shape:", repeated_tensor.shape
    )  # Repeat Interleave Tensor Shape: Shape([20, 10])


def test_slice(device):
    # Create a tensor to slice
    input_tensor = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Slice the tensor
    sliced_tensor = ttnn.slice(input_tensor, [0, 0, 0, 0], [1, 1, 64, 16], [1, 1, 2, 1])
    logger.info("Sliced Tensor Shape:", sliced_tensor.shape)  # Sliced Tensor Shape: Shape([1, 1, 32, 16])

    # Create a tensor to slice without step
    input_tensor = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    output = ttnn.slice(input_tensor, [0, 0, 0, 0], [1, 1, 32, 32])
    logger.info("Sliced Tensor Shape:", output.shape)  # Sliced Tensor Shape: Shape([1, 1, 32, 32])


def test_tilize(device):
    # Create a tensor to tilize
    input_tensor = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Tilize the tensor
    tilized_tensor = ttnn.tilize(input_tensor)
    logger.info("Tilized Tensor Shape:", tilized_tensor.shape)


def test_tilize_with_val_padding(device):
    # Create a tensor to tilize with value padding
    input_tensor = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Tilize the tensor with value padding
    tilized_tensor = ttnn.tilize_with_val_padding(input_tensor, output_tensor_shape=[1, 1, 64, 64], pad_value=0.0)
    logger.info("Tilized Tensor with Value Padding Shape:", tilized_tensor.shape)


def test_fill_rm(device):
    # Define tensor dimensions
    N = 2
    C = 3
    H = 64
    W = 96

    # Define fill dimensions
    hOnes = 33
    wOnes = 31

    # Define fill values
    val_hi = 1.0
    val_lo = 0.0

    # Create input tensor
    input = torch.zeros((N, C, H, W))
    xt = ttnn.Tensor(
        input.reshape(-1).tolist(),
        input.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ).to(device)

    # Fill the tensor
    output = ttnn.fill_rm(N, C, H, W, hOnes, wOnes, xt, val_hi, val_lo)
    logger.info("Filled Tensor Shape:", output.shape)


def test_fill_ones_rm(device):
    # Define tensor dimensions
    N = 2
    C = 3
    H = 64
    W = 96

    # Define fill dimensions
    hOnes = 33
    wOnes = 31

    # Create input tensor
    input = torch.zeros((N, C, H, W))
    xt = ttnn.Tensor(
        input.reshape(-1).tolist(),
        input.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ).to(device)

    # Fill the tensor
    output = ttnn.fill_ones_rm(N, C, H, W, hOnes, wOnes, xt)
    logger.info("Filled Ones Tensor Shape:", output.shape)


def test_untilize(device):
    # Create a tilized tensor
    tilized_tensor = ttnn.rand((1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Untilize the tensor
    untilized_tensor = ttnn.untilize(tilized_tensor)
    logger.info("Untilized Tensor Shape:", untilized_tensor.shape)


def test_untilize_with_unpadding(device):
    # Create a tilized tensor with padding
    tilized_tensor = ttnn.rand((1, 1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Untilize the tensor with unpadding
    untilized_tensor = ttnn.untilize_with_unpadding(tilized_tensor, output_tensor_end=(1, 1, 64, 32))
    logger.info("Untilized Tensor with Unpadding Shape:", untilized_tensor.shape)


def test_indexed_fill(device):
    # Define shapes for input tensors
    input_a_shape = (32, 1, 1, 4)
    input_b_shape = (6, 1, 1, 4)

    # Create a tensors to perform indexed fill
    batch_id = torch.randint(0, (32 - 1), (1, 1, 1, 6))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )
    input_tensor_a = ttnn.rand(input_a_shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.rand(input_b_shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Perform indexed fill
    output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b)
    logger.info("Indexed Fill Output Tensor Shape:", output_tensor.shape)


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
    input_tensor_a = ttnn.rand(input_a_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.rand(input_b_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Perform indexed fill - output preserves TILE layout.
    output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b)
    logger.info("Indexed Fill (TILE) Output Tensor Shape:", output_tensor.shape)
    logger.info("Indexed Fill (TILE) Output Tensor Layout:", output_tensor.layout)


def test_indexed_fill_sharded(device):
    # HEIGHT_SHARDED L1 example using the native CB-aliased fast path: input_a and the
    # output share the same shard geometry (one batch slab per core) so the kernel writes
    # the result directly into the output's per-core L1 shard with zero copy.
    B, b, D = 8, 3, 64
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


def test_indexed_fill_block_sharded_tile(device):
    # BLOCK_SHARDED + TILE layout exercises the SHARD_LOCAL_INTERLEAVED_B path with
    # 2D shard geometry: each core owns a (shard_H × shard_W) tile block of input_a
    # and reads its replacement rows from interleaved input_b.
    B, H, W = 4, 64, 64
    b = 2

    batch_id = torch.randint(0, B, (1, 1, 1, b))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    # 2×2 core grid → each core owns (B*H/2) rows × (W/2) cols of tiles.
    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(B, 1, H, W),
        core_grid=ttnn.CoreGrid(y=2, x=2),
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

    input_tensor_a = ttnn.rand(shape_a, dtype=ttnn.bfloat16, layout=layout, device=device)
    input_tensor_b = ttnn.rand(tuple(shape_b), dtype=ttnn.bfloat16, layout=layout, device=device)

    # Indexed fill along the requested dim.
    output_tensor = ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, dim=dim)

    # Output preserves input_a's shape and layout.
    assert tuple(output_tensor.shape) == shape_a
    assert output_tensor.layout == layout

    logger.info(
        f"Indexed Fill (dim={dim}, layout={layout}) Output Shape: {output_tensor.shape}, "
        f"Layout: {output_tensor.layout}"
    )


def test_indexed_fill_dim_out_of_bounds(device):
    # Verify that a dim outside [-rank, rank) raises a fatal error.
    input_tensor_a = ttnn.rand((4, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.rand((2, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    batch_id = torch.randint(0, 4, (1, 1, 1, 2))
    batch_id_ttnn = ttnn.Tensor(batch_id, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    with pytest.raises(Exception):
        ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, dim=4)  # rank=4, so dim=4 is out of bounds

    with pytest.raises(Exception):
        ttnn.indexed_fill(batch_id_ttnn, input_tensor_a, input_tensor_b, dim=-5)  # -5 < -rank=-4


def test_gather(device):
    # Create a tensor and an index tensor
    ttnn_input = ttnn.rand((4, 4), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.rand((4, 2), dtype=ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    # Gather elements along dimension 1 using the index tensor
    ttnn.gather(ttnn_input, 1, index=ttnn_index)


def test_sort(device):
    # Create a tensor
    input_tensor = torch.Tensor([[3, 1, 2], [3, 1, 2]])

    # Convert tensor to ttnn format
    input_tensor_ttnn = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Sort the tensor in ascending order
    sorted_tensor, indices = ttnn.sort(input_tensor_ttnn)
    logger.info(f"Sorted Tensor: {sorted_tensor}, Indices: {indices}")

    # Sort the tensor in descending order
    sorted_tensor_desc, indices_desc = ttnn.sort(input_tensor_ttnn, descending=True)
    logger.info(f"Sorted Tensor Descending: {sorted_tensor_desc}, Indices: {indices_desc}")

    # Sort along a specific dimension
    input_tensor_2d = torch.Tensor([[3, 1, 2], [6, 5, 4]])
    input_tensor_2d_ttnn = ttnn.from_torch(input_tensor_2d, dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    sorted_tensor_dim, indices_dim = ttnn.sort(input_tensor_2d_ttnn, dim=1)
    logger.info(f"Sorted Tensor along dim 1: {sorted_tensor_dim}, Indices: {indices_dim}")


def test_narrow(device):
    input_tensor = ttnn.rand((32, 16, 16, 4), dtype=ttnn.bfloat16, device=device)
    narrowed_tensor = ttnn.narrow(input_tensor, 0, 12, 8)
    logger.info("Narrowed Tensor Shape:", narrowed_tensor.shape)
