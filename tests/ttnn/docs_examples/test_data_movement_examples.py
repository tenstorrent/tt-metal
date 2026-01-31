# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
