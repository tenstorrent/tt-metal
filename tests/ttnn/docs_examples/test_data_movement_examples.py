# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


def test_gather(device):
    # Create a tensor and an index tensor
    ttnn_input = ttnn.rand([4, 4], ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.rand([4, 2], ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    # Gather elements along dimension 1 using the index tensor
    ttnn.gather(ttnn_input, 1, index=ttnn_index)


def test_sort(device):
    # Create a tensor
    input_tensor = torch.Tensor([3, 1, 2])

    # Convert tensor to ttnn format
    input_tensor_ttnn = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Sort the tensor in ascending order
    sorted_tensor, indices = ttnn.sort(input_tensor_ttnn)

    # Sort the tensor in descending order
    sorted_tensor_desc, indices_desc = ttnn.sort(input_tensor_ttnn, descending=True)

    # Sort along a specific dimension
    input_tensor_2d = torch.Tensor([[3, 1, 2], [6, 5, 4]])
    input_tensor_2d_ttnn = ttnn.from_torch(input_tensor_2d, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    sorted_tensor_dim, indices_dim = ttnn.sort(input_tensor_2d_ttnn, dim=1)


def test_concat(device):
    # Create two tensors to concatenate
    tensor1 = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, device=device)

    # Concatenate along dimension 3
    concatenated_tensor = ttnn.concat([tensor1, tensor2], dim=3)
    print("Concatenated Tensor Shape:", concatenated_tensor.shape)  # [1, 1, 64, 64]


def test_nonzero(device):
    # Create a tensor with some zero and non-zero elements
    input_tensor = torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)
    input_tensor_ttnn = ttnn.from_torch(input_tensor, device=device)

    # Find indices of non-zero elements
    nonzero_indices = ttnn.nonzero(input_tensor_ttnn)


def test_pad(device):
    # Create a tensor to pad
    input_tensor = ttnn.rand([1, 1, 4, 4], ttnn.bfloat16, device=device)

    # Pad the tensor with 1 zero on all sides
    padded_tensor = ttnn.pad(input_tensor, (1, 1, 1, 1), 0)
    print("Padded Tensor Shape:", padded_tensor.shape)  # [1, 1, 6, 6]


def test_permute(device):
    # Create a tensor to permute
    input_tensor = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, device=device)

    # Permute the tensor dimensions
    permuted_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    print("Permuted Tensor Shape:", permuted_tensor.shape)  # [1, 1, 32, 64]


def test_reshape(device):
    # Create a tensor to reshape
    input_tensor = torch.arange(4, dtype=torch.bfloat16)
    input_tensor_tt = ttnn.from_torch(input_tensor, device=device)

    # Reshape the tensor
    reshaped_tensor = ttnn.reshape(input_tensor_tt, (1, 1, 2, 2))
    print("Reshaped Tensor Shape:", reshaped_tensor.shape)  # [1, 1, 2, 2]


def test_repeat(device):
    # Create a tensor to repeat
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    input_tensor_tt = ttnn.from_torch(input_tensor, device=device)

    # Repeat the tensor along specified dimensions
    repeated_tensor = ttnn.repeat(input_tensor_tt, (1, 2))
    print("Repeated Tensor Shape:", repeated_tensor.shape)  # [1, 2], [1, 2], [3, 4], [3, 4]]


def test_repeat_interleave(device):
    # Create a tensor to repeat interleave
    input_tensor = torch.tensor([1, 2, 3])
    input_tensor_tt = ttnn.from_torch(input_tensor, device=device)

    # Repeat interleave the tensor
    repeated_tensor = ttnn.repeat_interleave(input_tensor_tt, repeats=2, dim=0)
    print("Repeat Interleave Tensor Shape:", repeated_tensor.shape)


def test_slice(device):
    # Create a tensor to slice
    input_tensor = ttnn.rand((1, 1, 64, 32), dtype=torch.bfloat16, device=device)

    # Slice the tensor
    sliced_tensor = ttnn.slice(input_tensor, [0, 0, 0, 0], [1, 1, 64, 16], [1, 1, 2, 1])
    print("Sliced Tensor Shape:", sliced_tensor.shape)

    # Create a tensor to slice without step
    input_tensor = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, device=device)
    output = ttnn.slice(input_tensor, [0, 0, 0, 0], [1, 1, 32, 32])
    print("Sliced Tensor Shape:", output.shape)


def test_tilize(device):
    # Create a tensor to tilize
    input_tensor = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Tilize the tensor
    tilized_tensor = ttnn.tilize(input_tensor)


def test_tilize_with_val_padding(device):
    # Create a tensor to tilize with value padding
    input_tensor = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Tilize the tensor with value padding
    tilized_tensor = ttnn.tilize_with_val_padding(input_tensor, output_tensor_shape=[1, 1, 64, 64], pad_value=0.0)


def test_fill_rm(device):
    # TODO: implement example
    pass


def test_fill_ones_rm(device):
    # TODO: implement example
    pass


def test_untilize(device):
    # Create a tilized tensor
    tilized_tensor = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, layout=ttnn.TILE, device=device)

    # Untilize the tensor
    untilized_tensor = ttnn.untilize(tilized_tensor)


def test_untilize_with_unpadding(device):
    # Create a tilized tensor with padding
    tilized_tensor = ttnn.rand([1, 1, 64, 64], ttnn.bfloat16, layout=ttnn.TILE, device=device)

    # Untilize the tensor with unpadding
    untilized_tensor = ttnn.untilize_with_unpadding(tilized_tensor, output_tensor_end=[1, 1, 64, 32])


def test_indexed_fill(device):
    # Create a tensor to perform indexed fill
    batch_id = ttnn.rand([1, 2], ttnn.uint32, layout=ttnn.Layout.TILE, device=device)
    input_tensor_a = ttnn.rand([1, 2], ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    input_tensor_b = ttnn.rand([0, 1], ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Perform indexed fill
    output_tensor = ttnn.indexed_fill(batch_id, input_tensor_a, input_tensor_b)
