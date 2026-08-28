# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_numeric_metrics

UINT16_MAX = 65535
TEST_PADDING_VALUE = -42


def run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids=None, pass_indices_tensor=False):
    torch.manual_seed(2005)

    if dtype == ttnn.bfloat8_b:
        pytest.xfail("BFLOAT8_B not supported by pad operation in topk")

    # float32 sorts at full 32-bit precision (no bf16 downcast), so its single-core compute
    # buffers are ~2x the bf16 size. The result-prep (2*Kt) + output (Kt) tiles dominate L1, so
    # the ceiling is a tile count, not a width: measured max is 54 output tiles (k <= 1728) in
    # Wormhole's ~1.5MB per-core L1; beyond that the buffers overflow. Multi-core can't help
    # here (it requires k <= 64). Kt = ceil(k/32).
    if dtype == ttnn.float32 and (k + 31) // 32 > 54:
        pytest.skip("exact float32 top-K exceeds single-core L1 beyond k=1728 (54 output tiles)")

    # Input tensor
    shape = [N, C, H, W]
    ttnn_indices_dtype = ttnn.uint16 if W <= UINT16_MAX else ttnn.uint32
    torch_indices_dtype = torch.uint16 if W <= UINT16_MAX else torch.uint32
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype) * 0.9
    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    if pass_indices_tensor:
        # The payload must differ from the iota topk generates on its own, or the gather below
        # cannot tell a read from a regeneration. Column i is labelled W - 1 - i.
        indices_tensor_torch = (W - 1 - torch.arange(W)).expand(shape).to(torch_indices_dtype)
        indices_tensor = ttnn.from_torch(
            indices_tensor_torch, ttnn_indices_dtype, layout=ttnn.Layout.TILE, device=device
        )
        indices_tensor = ttnn.fill_implicit_tile_padding(indices_tensor, TEST_PADDING_VALUE)
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
    # topk returns the labels it is given, so turn each label back into the column it names.
    # The labelling above is its own inverse: label L names column W - 1 - L.
    # Without a payload the op generates plain column indices, so no conversion is needed.
    ttnn_torch_columns = ttnn_torch_indices.to(torch.int64)
    if pass_indices_tensor:
        ttnn_torch_columns = W - 1 - ttnn_torch_columns
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_columns)
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"


@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        "FLOAT32",
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
        (8, 16, 18, 20, 3, 18),
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
    ((1, 1, 32, 16 * 1024, 3, 32), (8, 16, 18, 22, 3, 22)),
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
        (1, 1, 16, 20, 3, 16),
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
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)
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
        (torch.uint32, ttnn.uint32),
        (torch.int32, ttnn.int32),
    ],
)
def test_topk_input_dtypes_raise(torch_input_tensor_dtype, ttnn_input_tensor_dtype, device, expect_error):
    torch.manual_seed(0)
    shape = [1, 1, 32, 64]

    input_torch = torch.randint(0, 100, shape, dtype=torch_input_tensor_dtype)

    ttnn_input = ttnn.from_torch(input_torch, ttnn_input_tensor_dtype, layout=ttnn.Layout.TILE, device=device)

    with expect_error(RuntimeError, "Input tensor must be BFLOAT16, BFLOAT8_B, or FLOAT32"):
        ttnn.topk(ttnn_input, k=32, dim=-1, largest=True, sorted=True)


@pytest.mark.parametrize(
    "value_dtype, index_dtype",
    [
        (ttnn.float32, ttnn.uint16),
        (ttnn.uint32, ttnn.uint16),
        (ttnn.int32, ttnn.uint16),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat16),
    ],
)
def test_topk_preallocated_dtype_raise(value_dtype, index_dtype, device, expect_error):
    torch.manual_seed(0)
    k = 32
    shape = [1, 1, 32, 64]
    output_shape = [1, 1, 32, k]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Preallocated outputs must carry the topk output shape ([..., k]); allocating at the input shape would
    # trip the shape check in topk() before dtype validation is reached and defeat the purpose of this test.
    output_torch = torch.zeros(output_shape, dtype=torch.bfloat16)
    value_tensor = ttnn.from_torch(output_torch, value_dtype, layout=ttnn.Layout.TILE, device=device)
    index_tensor = ttnn.from_torch(output_torch, index_dtype, layout=ttnn.Layout.TILE, device=device)

    with expect_error(RuntimeError, "Preallocated"):
        ttnn.topk(ttnn_input, k=k, dim=-1, largest=True, sorted=True, output_tensor=(value_tensor, index_tensor))


@pytest.mark.parametrize(
    "W, input_dtype, index_dtype, raises",
    [
        (64, ttnn.bfloat16, ttnn.uint16, False),
        (64, ttnn.bfloat16, ttnn.uint32, False),
        (UINT16_MAX + 1, ttnn.bfloat16, ttnn.uint32, False),
        (UINT16_MAX + 1, ttnn.bfloat16, ttnn.uint16, True),
        (64, ttnn.float32, ttnn.uint16, True),
    ],
    ids=["w64_u16", "w64_u32", "w65536_u32", "w65536_u16_raises", "fp32_u16_raises"],
)
def test_topk_preallocated_indices_width(W, input_dtype, index_dtype, raises, device, expect_error):
    # index_dtype is the preallocated output indices tensor. A 16-bit one on an input that needs
    # 32 bits is rejected; wider than needed is legal. fp32 needs 32 bits at any width.
    torch.manual_seed(0)
    k = 32
    shape = [1, 1, 32, W]
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16

    torch_input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, input_dtype, layout=ttnn.Layout.TILE, device=device)
    value_tensor = ttnn.from_torch(
        torch.zeros([1, 1, 32, k], dtype=torch_dtype), input_dtype, layout=ttnn.Layout.TILE, device=device
    )
    index_tensor = ttnn.from_torch(
        torch.zeros([1, 1, 32, k], dtype=torch.int32), index_dtype, layout=ttnn.Layout.TILE, device=device
    )

    if raises:
        with expect_error(RuntimeError, "must be 32-bit"):
            ttnn.topk(ttnn_input, k=k, dim=-1, largest=True, sorted=True, output_tensor=(value_tensor, index_tensor))
        return

    values, indices = ttnn.topk(
        ttnn_input, k=k, dim=-1, largest=True, sorted=True, output_tensor=(value_tensor, index_tensor)
    )

    assert indices.dtype == index_dtype
    gathered = torch.gather(torch_input, -1, ttnn.to_torch(indices).to(torch.int64))
    assert_equal(gathered, ttnn.to_torch(values))


def test_topk_fp32_input_with_uint16_indices_tensor_raise(device, expect_error):
    # fp32 input forces UINT32 index CBs; a UINT16 indices_tensor would silently produce wrong indices.
    torch.manual_seed(0)
    k = 32
    shape = [1, 1, 32, 8192]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.float32, layout=ttnn.Layout.TILE, device=device)

    indices_torch = torch.zeros(shape, dtype=torch.uint16)
    indices_tensor = ttnn.from_torch(indices_torch, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    with expect_error(RuntimeError, "Optional indices tensor must be UINT32 when input tensor is FLOAT32"):
        ttnn.topk(ttnn_input, k=k, dim=-1, largest=True, sorted=True, indices_tensor=indices_tensor)


def test_topk_narrower_indices_tensor_raise(device, expect_error):
    # The indices are streamed with the input's page stride, so a narrower indices tensor is read at
    # the wrong pages and produces wrong indices rather than an error.
    torch.manual_seed(0)
    k = 32
    shape = [1, 1, 32, 8192]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    indices_torch = torch.zeros([1, 1, 32, shape[3] // 2], dtype=torch.uint16)
    indices_tensor = ttnn.from_torch(indices_torch, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    with expect_error(RuntimeError, "Indices tensor has incorrect shape"):
        ttnn.topk(ttnn_input, k=k, dim=-1, largest=True, sorted=True, indices_tensor=indices_tensor)


def test_topk_indices_tensor_on_non_last_dim_raise(device, expect_error):
    # The front end transposes the reduced dim to last and leaves the indices tensor alone, so the
    # indices are paged in the input's post-transpose layout and come back rounded to a tile.
    torch.manual_seed(0)
    k = 32
    shape = [1, 1, 8192, 32]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    indices_torch = torch.zeros(shape, dtype=torch.uint16)
    for i in range(shape[2]):
        indices_torch[:, :, i, :] = i
    indices_tensor = ttnn.from_torch(indices_torch, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    with expect_error(RuntimeError, "only supported for a reduction on the last dimension"):
        ttnn.topk(ttnn_input, k=k, dim=2, largest=True, sorted=True, indices_tensor=indices_tensor)


@pytest.mark.parametrize("W", (64, 16384), ids=["single_core", "multi_core"])
def test_topk_indices_tensor_payload_is_used(W, device):
    # Column i is labelled i + label_offset. A label below the offset means the payload was ignored;
    # subtracting it recovers the column, so a wrong column is caught too.
    # W=64 takes the single-core factory, W=16384 the multi-core one.
    torch.manual_seed(0)
    k = 32
    label_offset = 20000
    shape = [1, 1, 32, W]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    labels = (torch.arange(W, dtype=torch.int32) + label_offset).expand(shape)

    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    indices_tensor = ttnn.from_torch(labels, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True, indices_tensor=indices_tensor)

    returned = ttnn.to_torch(ttnn_indices, dtype=torch.uint16).to(torch.int64)
    assert torch.all(returned >= label_offset), "indices_tensor was ignored; the op generated its own iota"
    # Each label must name the column whose value was selected.
    assert_equal(torch.gather(torch_input, -1, returned - label_offset), ttnn.to_torch(values))


@pytest.mark.parametrize("W", (64, 16384), ids=["single_core", "multi_core"])
@pytest.mark.parametrize("index_dtype", (ttnn.uint16, ttnn.uint32, ttnn.int32), ids=["uint16", "uint32", "int32"])
def test_topk_indices_tensor_dtype(W, index_dtype, device):
    torch.manual_seed(0)

    k = 32
    shape = [1, 1, 32, W]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    iota = torch.arange(W, dtype=torch.int32).expand(shape)

    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    indices_tensor = ttnn.from_torch(iota, index_dtype, layout=ttnn.Layout.TILE, device=device)

    values, indices = ttnn.topk(ttnn_input, k, dim=-1, indices_tensor=indices_tensor)

    assert indices.dtype == index_dtype
    indices = ttnn.to_torch(indices).to(torch.int64)

    # Indices are the iota, so gathering from the input must reproduce the returned values.
    assert_equal(torch.gather(torch_input, -1, indices), ttnn.to_torch(values))


def test_topk_indices_tensor_labels_above_uint16_max(device):
    k, W = 32, 16384
    offset = 100_000
    shape = [1, 1, 32, W]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    labels = (torch.arange(W, dtype=torch.int64) + offset).expand(shape)

    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    indices_tensor = ttnn.from_torch(labels, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    values, indices = ttnn.topk(ttnn_input, k, dim=-1, indices_tensor=indices_tensor)
    indices = ttnn.to_torch(indices).to(torch.int64)

    assert torch.all(indices > UINT16_MAX), "labels were truncated to 16 bits"
    assert_equal(torch.gather(torch_input, -1, indices - offset), ttnn.to_torch(values))


def test_topk_indices_tensor_too_narrow_raises(device, expect_error):
    # W is past 65535, so the op resolves the index dtype to UINT32 and sizes the index CB 32-bit,
    # but the payload here is UINT16. Reject rather than read a 16-bit tensor at a 32-bit stride.
    k = 32
    W = UINT16_MAX + 1  # smallest width that forces UINT32
    shape = [1, 1, 32, W]

    ttnn_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
    )
    indices_tensor = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.int32), ttnn.uint16, layout=ttnn.Layout.TILE, device=device
    )

    with expect_error(RuntimeError, "must be the same width as the output indices dtype"):
        ttnn.topk(ttnn_input, k, dim=-1, indices_tensor=indices_tensor)


@pytest.mark.parametrize("tensor_under_test", ("indices_tensor", "output_tensor"))
def test_topk_row_major_tensor_raises(tensor_under_test, device, expect_error):
    # topk is tile-only: the compute kernel sorts tiles and compute_output_specs hardcodes TILE.
    k = 32
    W = 64
    shape = [1, 1, 32, W]

    ttnn_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
    )

    if tensor_under_test == "indices_tensor":
        kwargs = {
            "indices_tensor": ttnn.from_torch(
                torch.arange(W, dtype=torch.int32).expand(shape),
                ttnn.uint16,
                layout=ttnn.Layout.ROW_MAJOR,
                device=device,
            )
        }
    else:
        kwargs = {
            "output_tensor": (
                ttnn.from_torch(
                    torch.zeros([1, 1, 32, k], dtype=torch.bfloat16),
                    ttnn.bfloat16,
                    layout=ttnn.Layout.ROW_MAJOR,
                    device=device,
                ),
                ttnn.from_torch(
                    torch.zeros([1, 1, 32, k], dtype=torch.int32),
                    ttnn.uint16,
                    layout=ttnn.Layout.ROW_MAJOR,
                    device=device,
                ),
            )
        }

    with expect_error(RuntimeError, "must be in tiled format"):
        ttnn.topk(ttnn_input, k, dim=-1, **kwargs)


@pytest.mark.parametrize("largest", [True, False])
def test_topk_multicore_local_write_correctness(largest, device):
    """
    Correctness guard for the multi-core topk local-writer path: an input width >= multi_core_min_width
    (8192) routes to TopKMultiCoreProgramFactory + writer_local_topk, which NoC-writes each local-topk
    tile from its CB slot to the final core and then cb_pop_front releases the slot back to the compute
    producer. Correct aggregation at the final core relies on each write being drained before its slot is
    reused; a regression there -- e.g. a WAR hazard where the producer's next pack_tile overwrites the
    slot while the NoC write's source-read is still in flight -- would corrupt the landed values and make
    this check fail.

    This is a value-correctness guard, not a deterministic race reproducer: such a WAR is latent (masked
    by compute-pack latency), so it would not necessarily surface on every run.
    """
    torch.manual_seed(2005)
    W, k = 8192, 32  # W >= 8192 -> multi-core path; k=32 -> Kt=1
    t = torch.randn((1, 1, 32, W), dtype=torch.bfloat16)
    x = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v, i = ttnn.topk(x, k, dim=-1, largest=largest, sorted=True)
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(v).float()
    ref, _ = torch.topk(t.float(), k, dim=-1, largest=largest, sorted=True)
    # Compare the (order-insensitive) set of top-k values per row.
    got_s = got.sort(dim=-1, descending=True).values
    ref_s = ref.sort(dim=-1, descending=True).values
    assert torch.allclose(
        got_s, ref_s, atol=1e-2
    ), f"multi-core topk values mismatch (WAR regression?): max_diff={(got_s - ref_s).abs().max():.4f}"


@pytest.mark.parametrize(
    "H, W, k",
    (
        (32, 1024, 32),  # smallest newly-eligible width; crashed pre-fix on grids with lp2(cores) > W/32
        (32, 1024, 64),
        (32, 2048, 32),  # the measured ~4x cell class
        (32, 2048, 64),
        (32, 4096, 32),  # composite-routed by default on BH; sub_core_grids pins the stock path here
        (64, 4096, 64),  # two tile rows: upper edge of the Ht-aware gate
    ),
)
@pytest.mark.parametrize("largest", (True, False))
def test_topk_low_tile_row_multicore(H, W, k, largest, device):
    """
    Device coverage for the Ht-aware multi-core eligibility domain: pow2 widths in
    [multi_core_low_ht_min_width=1024, multi_core_min_width=8192) with <= 2 tile rows route to
    TopKMultiCoreProgramFactory (select_program_factory requirement #1's low-tile-row arm). Before
    the eligibility change these shapes ran single-core; W=1024 additionally used to crash inside
    find_topk_core_config on grids where largest_power_of_two(max_cores) > W/32 truncated the
    starting split size to zero (e.g. Blackhole 13x10). Checks both the top-k value set per row and
    that the returned indices really address the returned values in the input.

    sub_core_grids (the device's full compute grid, so the device op sees the same grid as the
    default) is passed to decline the Blackhole composite router: at pow2 W in [4096, 8192) with
    largest=True the router otherwise takes the cell (the composite measured faster there — see
    should_route_to_topk_large_indices in topk.cpp), and this test targets the stock
    multi-core factory specifically.
    """
    torch.manual_seed(2005)
    grid = device.compute_with_storage_grid_size()
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    t = torch.randn((1, 1, H, W), dtype=torch.bfloat16)
    x = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v, i = ttnn.topk(x, k, dim=-1, largest=largest, sorted=True, sub_core_grids=full_grid)
    ttnn.synchronize_device(device)

    got_v = ttnn.to_torch(v).float()
    got_i = ttnn.to_torch(i, dtype=torch.int64)
    ref_v, _ = torch.topk(t.float(), k, dim=-1, largest=largest, sorted=True)

    # Order-insensitive top-k value set per row (bf16 ties may be permuted).
    got_s = got_v.sort(dim=-1, descending=True).values
    ref_s = ref_v.sort(dim=-1, descending=True).values
    assert torch.allclose(
        got_s, ref_s, atol=1e-2
    ), f"low-Ht multicore topk values mismatch: max_diff={(got_s - ref_s).abs().max():.4f}"

    # Indices must address the returned values exactly (both come from the input untouched).
    gathered = torch.gather(t.float(), -1, got_i)
    assert torch.equal(gathered, got_v), "low-Ht multicore topk indices do not point at returned values"


@pytest.mark.parametrize(
    "H, W, k",
    (
        (32, 4096, 32),  # 1 tile row
        (64, 4096, 64),  # 2 tile rows
    ),
)
def test_topk_w4096_low_ht_default_routes_to_composite(H, W, k, device):
    """
    Correctness lock for the pow2 [small_k_route_min_padded_width=4096, multi_core_min_width=8192)
    low-tile-row cell with DEFAULT args on Blackhole: the composite router takes it (measured faster
    than the stock multi-core bitonic — see should_route_to_topk_large_indices in topk.cpp), even
    though the cell is structurally eligible for stock multi-core since the Ht-aware gate. On
    non-Blackhole archs the same call exercises the stock multi-core path instead; the assertions
    hold either way.
    """
    torch.manual_seed(2005)
    t = torch.randn((1, 1, H, W), dtype=torch.bfloat16)
    x = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v, i = ttnn.topk(x, k, dim=-1, largest=True, sorted=True)
    ttnn.synchronize_device(device)

    got_v = ttnn.to_torch(v).float()
    got_i = ttnn.to_torch(i, dtype=torch.int64)
    ref_v, _ = torch.topk(t.float(), k, dim=-1, largest=True, sorted=True)

    # Order-insensitive top-k value set per row (bf16 ties may be permuted).
    got_s = got_v.sort(dim=-1, descending=True).values
    ref_s = ref_v.sort(dim=-1, descending=True).values
    assert torch.allclose(
        got_s, ref_s, atol=1e-2
    ), f"routed W=4096 low-Ht topk values mismatch: max_diff={(got_s - ref_s).abs().max():.4f}"

    # Indices must address the returned values exactly.
    gathered = torch.gather(t.float(), -1, got_i)
    assert torch.equal(gathered, got_v), "routed W=4096 low-Ht topk indices do not point at returned values"


@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 64, 3, 32),  # small dim -> single-core path
        (1, 1, 32, 4224, 3, 32),  # larger dim, non-pow2 -> genuinely single-core
        (1, 1, 32, 4096, 3, 32),  # pow2 low-tile-row dim -> multi-core path (Ht-aware gate)
        (1, 1, 32, 8192, 3, 50),  # power-of-2 dim, k<=64 -> multi-core path (32-bit indices)
    ),
)
@pytest.mark.parametrize("index_dtype", (ttnn.int32, ttnn.uint32))
@pytest.mark.parametrize("largest", (True, False))
def test_topk_int32_indices(N, C, H, W, dim, k, index_dtype, largest, device):
    # Exercises 32-bit (UINT32/INT32) index outputs on both the single-core and multi-core paths.
    torch.manual_seed(2005)
    shape = [N, C, H, W]

    input = torch.randn(shape, dtype=torch.bfloat16) * 0.9
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    pyt_topk_values, _ = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    # Preallocate the value and (int32/uint32) index output tensors.
    out_shape = shape.copy()
    out_shape[dim] = k
    value_tensor = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
    )
    index_tensor = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.int32), index_dtype, layout=ttnn.Layout.TILE, device=device
    )

    ttnn_values, ttnn_indices = ttnn.topk(
        ttnn_input,
        k,
        dim=dim,
        largest=largest,
        sorted=True,
        output_tensor=(value_tensor, index_tensor),
    )

    assert ttnn_indices.dtype == index_dtype

    # The index dtype does not change which elements are selected; validate that the returned
    # indices point at the correct top-k values (gather + cosine similarity, as in run_topk_test).
    ttnn_torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.int32)
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))
    assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"


@pytest.mark.parametrize("num_rows", [64, 96])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_multicore_values_beyond_first_tile_row(num_rows, largest, device):
    """
    Regression test for the multi-core final-gather values corruption: topk_final.cpp's values gather
    ran without re-establishing datacopy unpack state at ht >= 1, so the state left by the previous
    iteration's index transpose (TRANSPOSE mode, UInt16/UInt32 SRCA format) made the bare copy_tile
    unpack the bf16 gathered values as garbage. Observable as fabricated ~1e38 values in EVERY row past
    the first 32 flattened rows (indices unaffected) on any multi-core call with > 32 flattened rows.
    """
    torch.manual_seed(2006)
    W, k = 8192, 32  # multi-core path; > 32 flattened rows => ht >= 1 iterations in the final gather
    t = torch.randn((1, 1, num_rows, W), dtype=torch.bfloat16)
    x = ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v, i = ttnn.topk(x, k, dim=-1, largest=largest, sorted=True)
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(v).float()
    ref, _ = torch.topk(t.float(), k, dim=-1, largest=largest, sorted=True)
    got_s = got.sort(dim=-1, descending=True).values
    ref_s = ref.sort(dim=-1, descending=True).values
    # Pre-fix this fails catastrophically (~1e38 magnitudes), not marginally.
    assert torch.allclose(
        got_s, ref_s, atol=1e-2
    ), f"values fabricated past first tile row: max_diff={(got_s - ref_s).abs().max():.4f}"


# ---------------------------------------------------------------------------
# Large-k Blackhole routing: ttnn.topk with bf16, dim=-1, largest=True,
# stable=False and 64 < k <= 2048 is routed at the composite level through
# ttnn.experimental.topk_large_indices (fused untilize+clamp to lowest-finite
# bf16 — the private topk_route_prep device op, exercised by EVERY routed
# call below -> op -> fused gather of the ORIGINAL values straight from the
# TILE source + TILE assembly of both outputs + index dtype emit — the
# private topk_route_finish device op, also exercised by EVERY routed call;
# a slice pair only when k is not a multiple of 16), bypassing the
# single-core cliff (the device op's own multi-core path is gated at
# k <= 64 + pow2 width + width < 65536).
#
# Tie semantics on the routed path: the returned index SET is a correct top-k
# set with deterministic-but-unspecified tie order, so these tests assert
# value-exactness (both sides sorted descending) and index validity
# (input[index] == value, in-range, unique), never index-order equality.
# ---------------------------------------------------------------------------

from models.common.utility_functions import is_blackhole


def run_topk_large_k_routed_test(N, C, H, W, k, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.9
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    pyt_values, _ = torch.topk(torch_input, k, dim=-1, largest=True, sorted=True)

    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True)

    # Index dtype contract must match the stock device op: UINT16 iff the
    # tile-padded width fits 16 bits, else UINT32.
    padded_w = 32 * ((W + 31) // 32)
    uint16_expected = padded_w <= UINT16_MAX
    assert ttnn_indices.dtype == (ttnn.uint16 if uint16_expected else ttnn.uint32)

    desired_shape = [N, C, H, k]
    assert list(ttnn_values.shape) == desired_shape
    assert list(ttnn_indices.shape) == desired_shape

    torch_values = ttnn.to_torch(ttnn_values)
    torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.uint16 if uint16_expected else torch.uint32).to(torch.int64)

    # Values: exact, order included -- both sides are sorted descending, so
    # this holds even under ties.
    assert_equal(torch_values, pyt_values)

    # Indices: value-based validation (tie order is unspecified).
    assert torch_indices.min() >= 0
    assert torch_indices.max() < W
    gathered = torch.gather(torch_input, -1, torch_indices)
    assert_equal(gathered, torch_values)
    for row_indices in torch_indices.reshape(-1, k):
        assert row_indices.unique().numel() == k


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
@pytest.mark.parametrize("k", [96, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("W", [8192, 32768, 65536, 131072])
def test_topk_large_k_routed(k, W, device):
    # All combos take the routed (topk_large_indices) path: bf16, dim=-1,
    # largest=True, stable=False, 64 < k <= 2048, W <= 2^19.
    run_topk_large_k_routed_test(1, 1, 32, W, k, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
@pytest.mark.parametrize("k", [512, 2048])
def test_topk_large_k_routed_non_pow2_width(k, device):
    # W=100000: neither a power of two nor 16-bit -- the old multi-core gates
    # excluded it entirely; the routed path supports it natively.
    run_topk_large_k_routed_test(1, 1, 32, 100000, k, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
@pytest.mark.parametrize("W", [262144, 524288])
@pytest.mark.parametrize("k", [1536, 2048])
def test_topk_large_k_routed_wide(k, W, device):
    # The routed envelope extends to 2^19 — kept from the gather-era L1
    # constraint as the silicon-validated ceiling even though the fused
    # topk_route_finish tail no longer stages full rows (see
    # large_k_route_max_width in topk.cpp). Fewer rows keep host-side
    # reference cost and transfer size sane at these widths.
    run_topk_large_k_routed_test(1, 1, 8, W, k, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
def test_topk_large_k_routed_ragged_shape(device):
    # H=30 and W=4999, neither a multiple of 32: the fused topk_route_prep
    # writer must emit ONLY the logical sticks/columns (tile padding dropped
    # on both axes), and W=4999 -> 157 width-tiles (prime) forces its
    # bw_last=5 remainder blocks on every tile-row. Every other routed cell
    # is 32-aligned, so this is the lone cover for both clamps.
    run_topk_large_k_routed_test(1, 1, 30, 4999, 96, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
def test_topk_large_k_routed_ragged_tall(device):
    # TALL ragged: enough tile-rows that cores own MULTIPLE prep blocks
    # crossing tile-row boundaries (nblocks = 7 * ceil(157/8) = 140 > grid),
    # which desynchronizes the prep input-CB read pointer on the bw_last
    # remainder blocks -- the wrap hazard the flat 30-row ragged cell is
    # structurally blind to (1 block/core there).
    run_topk_large_k_routed_test(1, 1, 224, 4999, 96, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
def test_topk_large_k_routed_single_row(device):
    # A single logical row engages topk_large_indices' column-parallel
    # (intra-row multi-core) factory underneath the routed composite.
    run_topk_large_k_routed_test(1, 1, 1, 65536, 2048, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
def test_topk_large_k_routed_neginf_lanes(device):
    # Rows whose top-k contains exact -inf: the routed path clamps the op's
    # input to the lowest finite bf16 (the op then sees no -inf and stamps a
    # REAL source position for every lane — its 0xFFFFFFFF sentinel never
    # fires) and gathers values from the ORIGINAL tensor, so -inf values are
    # bit-exact and -inf lanes carry real positions (stock/torch parity).
    # W=65536 is tile-aligned, so every returned index must be a real
    # in-range column.
    torch.manual_seed(2005)
    W, k, finite_count = 65536, 512, 100  # W=65536 -> tile-padded > 65535 -> uint32 indices
    torch_input = torch.full((1, 1, 32, W), -float("inf"), dtype=torch.bfloat16)
    finite = (torch.randn(1, 1, 32, finite_count) * 0.9).to(torch.bfloat16)
    torch_input[..., :finite_count] = finite

    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True)
    assert ttnn_indices.dtype == ttnn.uint32

    torch_values = ttnn.to_torch(ttnn_values)
    torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.uint32).to(torch.int64)
    pyt_values, _ = torch.topk(torch_input, k, dim=-1, largest=True, sorted=True)

    # Values match torch exactly, including the -inf tail.
    assert_equal(torch_values, pyt_values)

    # ALL lanes (finite and -inf alike) carry real, in-range, unique,
    # self-consistent indices -- no sentinel anywhere.
    assert torch_indices.min() >= 0
    assert torch_indices.max() < W
    gathered = torch.gather(torch_input, -1, torch_indices)
    assert_equal(gathered, torch_values)
    for row_indices in torch_indices.reshape(-1, k):
        assert row_indices.unique().numel() == k
    # The finite lanes specifically must point into the finite prefix.
    assert torch_indices[..., :finite_count].max() < finite_count


@pytest.mark.skipif(not is_blackhole(), reason="fallback-path guards mirror the Blackhole routing predicate")
def test_topk_large_k_routing_fallbacks(device):
    # Shapes/args just OUTSIDE the routing predicate must keep the stock
    # (device-op) path and stay correct. Kept at W=8192 so the stock
    # single-core runs are fast.
    W = 8192

    # k=2049 exceeds the topk_large_indices ceiling (2048) -> stock path.
    run_topk_test(1, 1, 32, W, 2049, ttnn.bfloat16, 3, True, True, device)

    # largest=False is not supported by topk_large_indices -> stock path.
    run_topk_test(1, 1, 32, W, 512, ttnn.bfloat16, 3, True, False, device)

    # stable=True promises lowest-index tie-breaking -> stock path. A
    # constant row makes this a real guard: stable top-96 of an all-equal
    # row MUST return indices 0..95 in order, which the routed (non-stable)
    # engine does not promise -- so this asserts INDICES, not just values.
    # (Value-only asserts pass on either engine and would not catch the
    # stable guard being deleted from the routing predicate.)
    torch_input = torch.ones(1, 1, 32, W, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, 96, dim=-1, largest=True, sorted=True, stable=True)
    assert_equal(ttnn.to_torch(ttnn_values), torch.ones(1, 1, 32, 96, dtype=torch.bfloat16))
    expected_indices = torch.arange(96, dtype=torch.int64).expand(1, 1, 32, 96)
    assert_equal(ttnn.to_torch(ttnn_indices).to(torch.int64), expected_indices)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
def test_topk_routed_k_not_multiple_of_16(device):
    # k=100 exercises the routed k % 16 != 0 tail: topk_large_indices
    # requires a multiple of 16, so the op runs at k_rounded=112 and
    # post_topk_transform slices back down to 100. k_rounded=112 is also the
    # suite's only k_rounded % 32 == 16 cell: topk_route_finish's last output
    # tile is half k-padding (its right face pair must stay zero-filled).
    run_topk_large_k_routed_test(1, 1, 32, 8192, 100, device)


@pytest.mark.skipif(
    not is_blackhole(), reason="large-k routing is Blackhole-only; stock single-core takes minutes at these shapes"
)
def test_topk_large_k_routed_multi_tile_rows(device):
    # H=96 (3 row-tiles) x C=2 batches: topk_route_finish decomposes its work
    # units over GLOBAL tile rows (batch * R_p/32 + rt) and pages the source
    # gather at row_tile * width_tiles + idx/32 — every other routed cell has
    # H <= 32 and a single batch (row_tile == 0, batch == 0 throughout), which
    # would leave that decomposition entirely uncovered.
    run_topk_large_k_routed_test(1, 2, 96, 8192, 96, device)


@pytest.mark.skipif(not is_blackhole(), reason="routing predicate is Blackhole-only")
def test_topk_large_k_routing_engages(device):
    # Guards the routing predicate itself: outputs alone are also satisfied
    # by the stock path, so a silently narrowed predicate would demote the
    # whole routed suite to stock with everything still green. Graph capture
    # shows which ops actually ran.
    torch.manual_seed(2005)
    torch_input = torch.randn(1, 1, 32, 8192, dtype=torch.bfloat16) * 0.9
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    def ran_large_indices(captured) -> bool:
        # The device op surfaces as TopkLargeIndices*/topk_large_indices
        # depending on node type; normalize before matching.
        return "largeindices" in str(captured).lower().replace("_", "")

    # Routed shape (k=96 > 64): topk_large_indices must appear in the trace.
    ttnn.graph.begin_graph_capture()
    ttnn.topk(ttnn_input, 96, dim=-1, largest=True, sorted=True)
    assert ran_large_indices(ttnn.graph.end_graph_capture())

    # Stock shape (k=32, pow2 width >= 8192 -> multi-core bitonic eligible):
    # the routed op must NOT appear.
    ttnn.graph.begin_graph_capture()
    ttnn.topk(ttnn_input, 32, dim=-1, largest=True, sorted=True)
    assert not ran_large_indices(ttnn.graph.end_graph_capture())
