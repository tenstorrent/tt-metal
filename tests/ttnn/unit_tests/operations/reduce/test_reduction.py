# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn
import sys

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull, skip_for_blackhole, is_blackhole, torch_random


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("correction", [True, False])
@pytest.mark.parametrize("keepdim", [True, False])
def test_std(device, batch_size, h, w, dim, correction, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.std(torch_input_tensor, dim=dim, keepdim=keepdim, correction=correction)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.std(input_tensor, dim=dim, keepdim=keepdim, correction=correction)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [None, [], -1, -2])
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("correction", [True, False])
def test_var(device, batch_size, h, w, dim, keepdim, correction):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.var(torch_input_tensor, dim=dim, keepdim=keepdim, correction=correction)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.var(input_tensor, dim=dim, keepdim=keepdim, correction=correction)
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
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_prod(device, batch_size, c, h, w, dim, keepdim, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.prod(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype
    )

    output_tensor = ttnn.prod(input_tensor, dim=dim, keepdim=keepdim, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.bfloat16)
    assert len(output_tensor.shape) == len(torch_output_tensor.shape)
    assert output_tensor.shape == torch_output_tensor.shape
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@skip_for_grayskull("Not a tile size multiple, may fail on GS if run all tests. #17084")
@pytest.mark.parametrize("dim_1", [1])
@pytest.mark.parametrize("dim_2", [2])
@pytest.mark.parametrize("dim_3", [3])
@pytest.mark.parametrize("dim_4", [4])
@pytest.mark.parametrize("dim_5", [4])
@pytest.mark.parametrize("dim_6", [6])
@pytest.mark.parametrize("dim_7", [7])
@pytest.mark.parametrize("dim_8", [8])
@pytest.mark.parametrize("dim", [[3, 7]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_8d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@skip_for_grayskull("Not a tile size multiple, may fail on GS if run all tests. #17084")
@pytest.mark.parametrize("dim_1", [1])
@pytest.mark.parametrize("dim_2", [2])
@pytest.mark.parametrize("dim_3", [3])
@pytest.mark.parametrize("dim_4", [4])
@pytest.mark.parametrize("dim_5", [5])
@pytest.mark.parametrize("dim_6", [6])
@pytest.mark.parametrize("dim_7", [7])
@pytest.mark.parametrize("dim", [[2, 5]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_7d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@skip_for_grayskull("Not a tile size multiple, may fail on GS if run all tests. #17084")
@pytest.mark.parametrize("dim_1", [1])
@pytest.mark.parametrize("dim_2", [2])
@pytest.mark.parametrize("dim_3", [3])
@pytest.mark.parametrize("dim_4", [4])
@pytest.mark.parametrize("dim_5", [5])
@pytest.mark.parametrize("dim_6", [6])
@pytest.mark.parametrize("dim", [[1, 4], -1, None])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_6d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5, dim_6), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@skip_for_grayskull("Not a tile size multiple, may fail on GS if run all tests. #17084")
@pytest.mark.parametrize("dim_1", [33])
@pytest.mark.parametrize("dim_2", [5])
@pytest.mark.parametrize("dim_3", [7])
@pytest.mark.parametrize("dim_4", [2])
@pytest.mark.parametrize("dim_5", [59])
@pytest.mark.parametrize("dim", [[1, 4], -1, None])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_5d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


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


@pytest.mark.parametrize("dim1", [1])
# This test picks the maximum dim2 that will pick the singlecore implementation.
# TopK multicore uses 8 cores in blackhole, so we need to add support for bitonic sort with 8 cores
# and non power of 2 dims as compared to wormhole. Issue #23465.
@pytest.mark.parametrize(
    "dim2",
    [8192 - 64, pytest.param(50257, marks=pytest.mark.xfail(condition=is_blackhole(), reason="Issue #23465"))],
)  # Need to resolve issue #20294 to verify 128256 for OXMIQ <- will need topk_local_sort to handle uint32_t
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("k", [50, 3200])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_2d_topk(device, dim1, dim2, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype) * 0.9

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices)

    # Add 2^16 to negative values
    ttnn_torch_indices = ttnn_torch_indices.to(dtype=torch.int32)
    ttnn_torch_indices = torch.where(ttnn_torch_indices < 0, ttnn_torch_indices + 65536, ttnn_torch_indices)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # Convert to int64 only for torch.gather which requires signed indices
    ttnn_torch_gather_from_indices = torch.gather(
        input, dim, ttnn_torch_indices.to(torch.int64)  # Convert to signed only for PyTorch API compatibility
    )

    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))
    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)


@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [1])
@pytest.mark.parametrize("dim3", [8])
@pytest.mark.parametrize("dim4", [256])
@pytest.mark.parametrize("dim5", [64])
# @pytest.mark.parametrize("dim", [0, 1, 2, 3, 4]) #transpose cannot handle N-D tensor for all dims
@pytest.mark.parametrize("dim", [3, 4])
@pytest.mark.parametrize("k", [17, 32, 64])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, pytest.param(ttnn.bfloat8_b, marks=pytest.mark.xfail(reason="Issue #21438"))]
)
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

    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)


# returns larger padded tensor instead of desired shape
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
    torch_dtype = torch.bfloat16

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

    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)


@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [[0, 2], [0, 1, 2], None])
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
@pytest.mark.parametrize("dim", [0, 1, [0, 1], None])
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
@pytest.mark.parametrize("keepdim", [True, False])
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
@pytest.mark.parametrize("keepdim", [True, False])
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
@pytest.mark.parametrize("keepdim", [True, False])
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


def run_maxpool(device, input_shape, kernel_size, stride, padding, dilation):
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    batch_size, in_c, in_h, in_w = input_shape
    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.max_pool2d(
        input_tensor,
        batch_size,
        in_h,
        in_w,
        in_c,
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_output_tensor = torch.nn.functional.max_pool2d(torch_input, kernel_size, stride, padding)

    output_tensor = ttnn.to_torch(output_tensor)
    _, out_c, out_h, out_w = torch_output_tensor.shape
    output_tensor = torch.reshape(output_tensor, (batch_size, out_h, out_w, out_c))
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    assert_with_pcc(output_tensor, torch_output_tensor)


def run_reduce_sum_h(device, batch_size, h, w, dim):
    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.mean(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_grayskull("Not a tile size multiple, will fail on GS. #17132")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4096}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 192, 56, 56),  # Multi core face height not default
    ],
)
@pytest.mark.parametrize(
    "kernel_size",
    [
        (2, 2),  # Small kernel
        (5, 5),  # Large kernel
    ],
)
def test_run_reduce_sum_h_after_max_pool(device, input_shape, kernel_size):
    run_maxpool(device, input_shape, kernel_size, kernel_size, (0, 0), (1, 1))
    run_reduce_sum_h(device, 1, 32, 32, -2)


@pytest.mark.parametrize(
    argnames="tensor_shape, keepdim, dim, op",
    argvalues=[
        ([], True, None, "mean"),
        ([], True, None, "std"),
        ([32], False, -1, "sum"),
        ([32, 0], True, 0, "max"),
        ([0, 0, 0], True, 2, "min"),
        ([0, 32, 0], False, -2, "std"),
        ([32, 32, 32, 0], False, 3, "var"),
    ],
)
def test_torch_compatibility(device, tensor_shape, keepdim, dim, op):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = torch.randn(*tensor_shape) if rank > 0 else torch.randn(())
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except IndexError:
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim) if dim is not None else ttnn_op(ttnn_tensor)
    except RuntimeError:
        ttnn_errored = True

    assert torch_errored == ttnn_errored, f"torch: {torch_errored}, ttnn: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    # torch's min/max double as argmin/argmax, so we need to extract the values only
    torch_result = (
        torch_result.values
        if isinstance(torch_result, (torch.return_types.min, torch.return_types.max))
        else torch_result
    )

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    assert_with_pcc(torch_result, ttnn_result, 0.99)

    atol = rtol = 0.1
    # There is a scale factor difference between torch and ttnn for std and var
    # But for other operations, it should be close. Issue #19478
    if op == "std":
        atol, rtol = sys.maxsize, 0.1 + math.sqrt(2)
    elif op == "var":
        atol, rtol = sys.maxsize, 0.1 + 2

    assert torch.allclose(
        torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True
    ), f"torch: {torch_result}, ttnn: {ttnn_result}"
