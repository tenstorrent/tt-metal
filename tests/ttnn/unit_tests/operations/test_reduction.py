# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
def test_std(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.std(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.std(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [None, [], -1, -2])
@pytest.mark.parametrize("keepdim", [True])
def test_var(device, batch_size, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.var(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.var(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert len(torch_output_tensor.shape) == len(output_tensor.shape)
    assert torch_output_tensor.shape == output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("c", [11])
@pytest.mark.parametrize("h", [67])
@pytest.mark.parametrize("w", [77])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("keepdim", [True, False])
def test_prod(device, batch_size, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.prod(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output_tensor = ttnn.prod(input_tensor, dim=dim, keepdim=keepdim, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert len(output_tensor.shape) == len(torch_output_tensor.shape)
    assert output_tensor.shape == torch_output_tensor.shape
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("c", [32])
@pytest.mark.parametrize("h", [37])
@pytest.mark.parametrize("w", [63])
@pytest.mark.parametrize("dim", [None, [], 0, 2, [0, 1], [1, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_4d_tensor_dims(device, batch_size, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [[0, 2], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_3d_tensor_dims(device, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("h", [41])
@pytest.mark.parametrize("w", [31])
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_2d_tensor_dims(device, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("c", [5])
@pytest.mark.parametrize("h", [37])
@pytest.mark.parametrize("w", [63])
@pytest.mark.parametrize("dim", [None, [], 0, 2, [0, 1], [1, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]])
@pytest.mark.parametrize("keepdim", [True])
def test_mean_4d_tensor_dims(device, batch_size, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [[0, 2], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True])
def test_mean_3d_tensor_dims(device, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("h", [41])
@pytest.mark.parametrize("w", [31])
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [True])
def test_mean_2d_tensor_dims(device, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [1])
@pytest.mark.parametrize("dim3", [8])
@pytest.mark.parametrize("dim4", [256])
@pytest.mark.parametrize("dim5", [64])
# @pytest.mark.parametrize("dim", [0, 1, 2, 3, 4]) transpose cannot handle N-D tensor for all dims
@pytest.mark.parametrize("dim", [3, 4])
@pytest.mark.parametrize("k", [17, 32, 64])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_5d_topk(device, dim1, dim2, dim3, dim4, dim5, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2, dim3, dim4, dim5]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2, dim3, dim4, dim5]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices).to(torch.int64)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # rounding may also cause more ties than expected
    # the bigger we get, the tighter the distribution of the top K elements, so the pcc will be worse as stability/rounding will cause more ties
    # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"
    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)


@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [1])
@pytest.mark.parametrize("dim3", [8])
@pytest.mark.parametrize("dim4", [1])
@pytest.mark.parametrize("dim5", [128])
@pytest.mark.parametrize("dim6", [64])
# @pytest.mark.parametrize("dim", [0, 1, 2, 3, 4, 5]) transpose cannot handle N-D tensor for all dims
@pytest.mark.parametrize("dim", [4, 5])
@pytest.mark.parametrize("k", [50, 64])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_6d_topk(device, dim1, dim2, dim3, dim4, dim5, dim6, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2, dim3, dim4, dim5, dim6]
    torch_dtype = torch.bfloat
    16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2, dim3, dim4, dim5, dim6]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices).to(torch.int64)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # rounding may also cause more ties than expected
    # the bigger we get, the tighter the distribution of the top K elements, so the pcc will be worse as stability/rounding will cause more ties
    # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"
    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)
