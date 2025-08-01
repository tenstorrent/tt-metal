# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids=None, pass_indices_tensor=False):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype) * 0.9
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)
    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)

    if pass_indices_tensor:
        indices_tensor_torch = torch.zeros(shape, dtype=torch.int32)
        for i in range(W):
            indices_tensor_torch[:, :, :, i] = i
        indices_tensor = ttnn.from_torch(indices_tensor_torch, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    else:
        indices_tensor = None

    try:
        ttnn_topk_values, ttnn_topk_indices = ttnn.topk(
            ttnn_input,
            k,
            dim=dim,
            largest=largest,
            sorted=sorted,
            sub_core_grids=sub_core_grids,
            indices_tensor=indices_tensor,
        )
    except Exception as e:
        raise e

    desired_shape = [N, C, H, W]
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


@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        # ttnn.float32, top bits in float32 get cut off somewhere, LLK does not work for this
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        # "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 8192, 3, 50),  # passed
        (1, 1, 64, 64, 2, 32),  # passed
        (1, 1, 64, 64, 2, 64),  # passed
        (1, 2048, 1, 64, 1, 32),  # skipped
        (1, 1, 32, 64, 3, 2),  # passed
        (1, 1, 32, 64, 3, 4),  # passed
        (1, 1, 32, 8192, 3, 6),  # passed
        (1, 2048, 1, 64, 1, 8),  # passed
        (1, 1, 32, 32768, 3, 3000),  # passed
    ),
)
@pytest.mark.parametrize(
    "sorted",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "largest",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        None,
    ],
)
def test_topk(N, C, H, W, dim, k, dtype, sorted, largest, device, sub_core_grids):
    if dim == 0 or dim == 1:
        # As of now, when we try to get top-k for dim = 0 or 1, we get following error from transpose_op.cpp's validate():
        # input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32
        # this is because, transpose.cpp always typecasts bf8 to bf16
        # and when dim = 0 or 1, transpose converts it into TransposeOpDim::HC & this dim doesnt support bf16 or fp32
        pytest.skip()
    run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "BFLOAT16_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    ((1, 1, 32, 16 * 1024, 3, 32),),
)
@pytest.mark.parametrize(
    "sorted",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "largest",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "pass_indices_tensor",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7)
                ),  # Note: for TG llama we use 1,0 to 3,9 but this requires TGs (non-harvested) and "dispatch_core_axis": ttnn.DispatchCoreAxis.COL
            ]
        ),
    ],
)
def test_topk_sub_core_grids(N, C, H, W, dim, k, dtype, sorted, largest, device, sub_core_grids, pass_indices_tensor):
    if dim == 0 or dim == 1:
        # As of now, when we try to get top-k for dim = 0 or 1, we get following error from transpose_op.cpp's validate():
        # input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32
        # this is because, transpose.cpp always typecasts bf8 to bf16
        # and when dim = 0 or 1, transpose converts it into TransposeOpDim::HC & this dim doesnt support bf16 or fp32
        pytest.skip()
    run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids, pass_indices_tensor)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "BFLOAT16_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 151936, 3, 50),  # passed  - customer shape 2
        (1, 1, 32, 128256, 3, 50),  # passed  - customer shape 1
    ),
)
@pytest.mark.parametrize(
    "sorted",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "largest",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "pass_indices_tensor",
    [
        True,
        False,
    ],
)
def test_topk_large_2d_shapes(N, C, H, W, dim, k, dtype, sorted, largest, device, sub_core_grids, pass_indices_tensor):
    if dim == 0 or dim == 1:
        pytest.skip()
    run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids, pass_indices_tensor)
