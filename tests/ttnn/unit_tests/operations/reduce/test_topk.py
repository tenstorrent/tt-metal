# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal

UINT16_MAX = 65535


def save_tensor(tensor, filename):
    import pandas as pd

    # Write tensor to CSV file
    # Tensor data should be
    # _,0,1,2,3,4, ...
    # 0,y,y,y,y,y
    # 1,y,y,y,y,y
    # 2,y,y,y,y,y
    # 3,y,y,y,y,y
    #
    # First row -> column indices
    # First column -> row indices
    # y = data value

    # Convert to numpy 2D (handle torch and ttnn tensors; squeeze batch/extra dims)
    import numpy as np

    data = np.asarray(tensor.to(torch.float32))

    print("saving tensor to csv {filename}")
    while data.ndim > 2:
        data = data[0]
    df = pd.DataFrame(data)
    df.index.name = ""
    df.to_csv(filename, index=True)


def run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids=None, pass_indices_tensor=False):
    torch.manual_seed(2005)
    torch.set_printoptions(profile="full")
    torch.set_printoptions(sci_mode=False)
    # Input tensor
    shape = [N, C, H, W]
    # ttnn_indices_dtype = ttnn.uint16 if W <= UINT16_MAX else ttnn.uint32
    # torch_indices_dtype = torch.uint16 if W <= UINT16_MAX else torch.uint32
    torch_dtype = torch.bfloat16
    input = torch.randint(0, 100, shape, dtype=torch_dtype)

    # Build an indices tensor with increasing integer values along the 'dim' dimension:
    indices_shape = list(shape)
    axis_len = shape[dim]
    # Create a 1D increasing range and reshape to broadcast along the required axis
    axis_input = torch.arange(1, axis_len + 1, dtype=torch.int64)
    # Create a shape with 1 in all dims, except axis which gets axis_len
    idx_shape = [1] * len(shape)
    idx_shape[dim] = axis_len
    axis_input = axis_input.view(*idx_shape)
    # Broadcast to full shape
    input = axis_input.expand(*shape)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    # if pass_indices_tensor:
    #     indices_tensor_torch = torch.zeros(shape, dtype=torch_indices_dtype)
    #     for i in range(W):
    #         indices_tensor_torch[:, :, :, i] = i
    #     indices_tensor = ttnn.from_torch(
    #         indices_tensor_torch, ttnn_indices_dtype, layout=ttnn.Layout.TILE, device=device
    #     )
    # else:
    # indices_tensor = None

    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(
        ttnn_input,
        k,
        dim=dim,
        largest=largest,
        sorted=sorted,
        sub_core_grids=sub_core_grids,
        # indices_tensor=indices_tensor,
    )
    # Convert TTNN outputs to Torch for comparison
    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    print("Input Tensor:")
    # print(input[0, 0, 2048, :])
    print("TTNN TopK Values:")
    # print(ttnn_torch_values[0, 0, 2048, :])
    print("PyTorch TopK Values:")
    # print(pyt_topk_values[0, 0, 2048, :])
    # for n in range(N):
    #     for c in range(C):
    #             for i in range(H):
    #                 ttnn_tensor = ttnn_torch_values[n, c, i, :]
    #                 torch_tensor = pyt_topk_values[n, c, i, :]
    #                 print(i, end=', ')
    #                 assert_equal(ttnn_tensor, torch_tensor)
    torch.set_printoptions(profile="default")
    # ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices, dtype=torch_indices_dtype)

    # # Assert output shapes
    # desired_shape = [N, C, H, W]
    # desired_shape[dim] = k
    # assert list(ttnn_topk_values.shape) == desired_shape
    # assert list(ttnn_topk_indices.shape) == desired_shape

    # # Assert values correctness
    # if dtype == ttnn.bfloat8_b:
    #     assert_allclose(ttnn_torch_values, pyt_topk_values, rtol=1e-1, atol=1e-1)
    # else:
    # save_tensor(ttnn_torch_values, "ttnn_torch_values.csv")
    # save_tensor(pyt_topk_values, "torch_topk_values.csv")

    assert_equal(ttnn_torch_values, pyt_topk_values)

    # # Assert indices correctness using gather
    # # pcc is not a good measure for the raw indices
    # # if index 49 and index 8 are tied, the order of the indices can be different
    # # but the values associated with the indices should be the same
    # # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # # rounding may also cause more ties than expected
    # # the bigger we get, the tighter the distribution of the top K elements, so the pcc will be worse as stability/rounding will cause more ties
    # # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    # ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
    # cosine = torch.nn.CosineSimilarity(dim=dim)
    # ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    # assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"


@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
        # ttnn.float32, top bits in float32 get cut off somewhere, LLK does not work for this
    ),
    ids=[
        "BFLOAT16_B",
        # "BFLOAT8_B",
        # "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 8192, 3, 50),
        (1, 1, 64, 64, 2, 32),
        (1, 1, 32, 32 * 512, 3, 32),
        (1, 1, 64, 64, 2, 64),
        (1, 2048, 1, 64, 1, 32),
        (1, 1, 32, 64, 3, 2),
        (1, 1, 32, 64, 3, 4),
        (1, 1, 32, 8192, 3, 6),
        (1, 2048, 1, 64, 1, 8),
        (1, 1, 32, 32768, 3, 3000),
        (1, 1, 32, 18992, 3, 3000),
        (1, 1, 32, 18992, 3, 32),
        (1, 1, 32, 10000, 3, 32),
        (1, 1, 32, 64128, 3, 32),
        (1, 1, 65 * 32, 32 * 3, 3, 32),
        (1, 1, 65 * 32, 32 * 3, 3, 32),
        (1, 10, 32, 512, 2, 32),
        (5, 9, 96, 1024, 2, 32),
        (5, 9, 1024, 96, 3, 32),
        (3, 2, 160, 960, 2, 32),
    ),
)
@pytest.mark.parametrize(
    "sorted",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize(
    "largest",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        None,
    ],
)
def test_topk(N, C, H, W, dim, k, dtype, sorted, largest, device, sub_core_grids):
    run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids)


"""
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
        (1, 1, 32, 151936, 3, 50),
        (1, 1, 32, 128256, 3, 50),
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


@pytest.mark.parametrize(
    "torch_input_tenosr_dtype, ttnn_input_tenosr_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.uint32, ttnn.uint32),
        (torch.int32, ttnn.int32),
    ],
)
def test_topk_input_dtypes_raise(torch_input_tenosr_dtype, ttnn_input_tenosr_dtype, device):
    torch.manual_seed(0)
    shape = [1, 1, 32, 64]

    if torch_input_tenosr_dtype == torch.float32:
        input_torch = torch.randn(shape, dtype=torch_input_tenosr_dtype)
    else:
        input_torch = torch.randint(0, 100, shape, dtype=torch_input_tenosr_dtype)

    ttnn_input = ttnn.from_torch(input_torch, ttnn_input_tenosr_dtype, layout=ttnn.Layout.TILE, device=device)

    with pytest.raises(Exception):
        ttnn.topk(ttnn_input, k=32, dim=-1, largest=True, sorted=True)


@pytest.mark.parametrize(
    "value_dtype, index_dtype",
    [
        (ttnn.float32, ttnn.uint16),
        (ttnn.uint32, ttnn.uint16),
        (ttnn.int32, ttnn.uint16),
        (ttnn.bfloat16, ttnn.int32),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat16),
    ],
)
def test_topk_preallocated_dtype_raise(value_dtype, index_dtype, device):
    torch.manual_seed(0)
    shape = [1, 1, 32, 64]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    value_tensor = ttnn.empty_like(ttnn_input, dtype=value_dtype)
    index_tensor = ttnn.empty_like(ttnn_input, dtype=index_dtype)

    with pytest.raises(Exception):
        ttnn.topk(ttnn_input, k=32, dim=-1, largest=True, sorted=True, out=(value_tensor, index_tensor))
"""
