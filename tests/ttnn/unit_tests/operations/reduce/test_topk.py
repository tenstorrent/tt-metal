# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_numeric_metrics

UINT16_MAX = 65535


def run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids=None, pass_indices_tensor=False):
    torch.manual_seed(2005)

    if dtype == ttnn.bfloat8_b:
        pytest.xfail("BFLOAT8_B not supported by pad operation in topk")

    # Input tensor
    shape = [N, C, H, W]
    ttnn_indices_dtype = ttnn.uint16 if W <= UINT16_MAX else ttnn.uint32
    torch_indices_dtype = torch.uint16 if W <= UINT16_MAX else torch.uint32
    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype) * 0.9
    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    if pass_indices_tensor:
        indices_tensor_torch = torch.zeros(shape, dtype=torch_indices_dtype)
        for i in range(W):
            indices_tensor_torch[:, :, :, i] = i
        indices_tensor = ttnn.from_torch(
            indices_tensor_torch, ttnn_indices_dtype, layout=ttnn.Layout.TILE, device=device
        )
    else:
        indices_tensor = None

    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(
        ttnn_input,
        k,
        dim=dim,
        largest=largest,
        sorted=sorted,
        sub_core_grids=sub_core_grids,
        indices_tensor=indices_tensor,
    )

    # Convert TTNN outputs to Torch for comparison
    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices, dtype=torch_indices_dtype)

    # Assert output shapes
    desired_shape = [N, C, H, W]
    desired_shape[dim] = k
    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    # test for equivalance
    assert_numeric_metrics(
        pyt_topk_values,
        ttnn_torch_values,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )
    assert_equal(ttnn_torch_values, pyt_topk_values)

    # Assert indices correctness using gather
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


def run_topk_bfloat8_inf_test(N, C, H, W, k, dim, sub_core_grids, device):
    assert W % 32 == 0, "W must be a multiple of 32 to avoid the pad path"
    assert H >= 2, "H must be >= 2 to have both finite and all-inf rows"

    torch.manual_seed(2005)
    shape = [N, C, H, W]
    input_torch = torch.randn(shape, dtype=torch.bfloat16) * 0.9
    # Set all rows except the first to +inf to trigger the shared-exponent bug
    # on the intermediate transposed tiles.
    input_torch[:, :, 1:, :] = float("inf")

    pyt_values, _ = torch.topk(input_torch, k, dim=dim, largest=True, sorted=True)

    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat8_b, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(
        ttnn_input, k, dim=dim, largest=True, sorted=True, sub_core_grids=sub_core_grids
    )

    desired_shape = list(shape)
    desired_shape[dim] = k
    assert list(ttnn_values.shape) == desired_shape
    assert list(ttnn_indices.shape) == desired_shape

    ttnn_values_torch = ttnn.to_torch(ttnn_values)
    ttnn_indices_torch = ttnn.to_torch(ttnn_indices).to(torch.int64)

    # Only compare the finite (H=0) rows; the all-inf rows are uninteresting and
    # their exact ordering is undefined when all values are equal (+inf).
    pyt_values_finite = pyt_values[:, :, :1, :]
    ttnn_values_finite = ttnn_values_torch[:, :, :1, :]
    ttnn_gather_finite = torch.gather(input_torch, dim, ttnn_indices_torch)[:, :, :1, :]

    cosine = torch.nn.CosineSimilarity(dim=dim)
    cosine_sim = torch.mean(cosine(pyt_values_finite, ttnn_gather_finite))
    assert cosine_sim > 0.99, (
        f"Cosine similarity between bfloat8_b topk values and gather-from-indices "
        f"is {cosine_sim:.4f} (expected > 0.99).  "
        f"This is the bfp8 shared-exponent/inf regression."
    )

    # bfloat8_b has 2 mantissa bits, so the maximum relative quantization error per
    # value is 2^-2 = 25%.  Two quantization steps occur (input bf16→bfp8 and output
    # bf16→bfp8), but they are correlated so the combined worst-case stays near 25%.
    # rtol=0.1 is therefore stricter than the format's theoretical maximum, meaning it
    # catches genuine corruption (e.g. values becoming 0 due to the inf/shared-exponent
    # bug) while tolerating legitimate bfp8 rounding.
    # Cast to float32 first: pyt_values_finite is bfloat16 while ttnn_values_finite is
    # float32 (ttnn.to_torch upcasts bfloat8_b), and torch.allclose raises on dtype mismatch.
    assert torch.allclose(pyt_values_finite.float(), ttnn_values_finite.float(), rtol=0.1, atol=0.1), (
        f"bfloat8_b TopK values exceed 10 % relative error vs PyTorch reference:\n"
        f"  PyTorch:  {pyt_values_finite}\n"
        f"  TTNN:     {ttnn_values_finite}"
    )


@pytest.mark.parametrize(
    "N, C, H, W, dim, k, sub_core_grids",
    [
        (1, 1, 32, 256, 3, 32, None),
        (
            1,
            1,
            32,
            16 * 1024,
            3,
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7)),
                ]
            ),
        ),
    ],
    ids=["single_core_bfp8_inf", "multi_core_bfp8_inf"],
)
def test_topk_bfloat8_with_inf(N, C, H, W, dim, k, sub_core_grids, device):
    """bfloat8_b TopK correctness when an entire H-row contains +inf values."""
    if dim == 0 or dim == 1:
        pytest.skip("dim=0/1 not supported for bfloat8_b (transpose path requires bfloat16 or float32)")
    run_topk_bfloat8_inf_test(N, C, H, W, k, dim, sub_core_grids, device)


@pytest.mark.parametrize(
    "torch_input_tensor_dtype, ttnn_input_tensor_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.uint32, ttnn.uint32),
        (torch.int32, ttnn.int32),
    ],
)
def test_topk_input_dtypes_raise(torch_input_tensor_dtype, ttnn_input_tensor_dtype, device):
    torch.manual_seed(0)
    shape = [1, 1, 32, 64]

    if torch_input_tensor_dtype == torch.float32:
        input_torch = torch.randn(shape, dtype=torch_input_tensor_dtype)
    else:
        input_torch = torch.randint(0, 100, shape, dtype=torch_input_tensor_dtype)

    ttnn_input = ttnn.from_torch(input_torch, ttnn_input_tensor_dtype, layout=ttnn.Layout.TILE, device=device)

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
        ttnn.topk(ttnn_input, k=32, dim=-1, largest=True, sorted=True, output_tensor=(value_tensor, index_tensor))
