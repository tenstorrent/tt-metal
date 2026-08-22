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
        indices_tensor_torch = torch.zeros(shape, dtype=torch_indices_dtype)
        for i in range(W):
            indices_tensor_torch[:, :, :, i] = i
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
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
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
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 64, 3, 32),  # small dim -> single-core path
        (1, 1, 32, 4096, 3, 32),  # larger dim, still single-core
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


@pytest.mark.parametrize(
    "W, k",
    (
        (64, 4),
        (64, 8),
        (64, 16),
        (64, 32),
        (64, 64),
        (8192, 32),  # W >= 8192 -> multi-core path
        (8192, 64),
    ),
)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_stable_index_parity(W, k, largest, device):
    """stable=True must break exact-value ties by lowest original index, matching torch's
    stable ordering bit-exactly for both values and indices (no tie tolerance)."""
    torch.manual_seed(0)
    shape = [1, 1, 32, W]
    # Only 8 distinct values across W elements per row -> many exact ties in every row.
    input = torch.randint(0, 8, shape).to(torch.bfloat16)

    # torch.topk is NOT stable, so derive the golden from a stable argsort instead.
    order = torch.argsort(input, dim=-1, descending=largest, stable=True)[..., :k]
    golden_values = torch.gather(input, -1, order)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    # k < 32 is padded up to a tile-aligned k internally and sliced back to k on output.
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)

    ttnn_torch_values = ttnn.to_torch(ttnn_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)

    assert_equal(golden_values, ttnn_torch_values)
    assert_equal(order, ttnn_torch_indices)


def _stable_topk_golden(input, k, largest):
    # torch.topk is NOT stable, so derive the golden from a stable argsort instead.
    order = torch.argsort(input, dim=-1, descending=largest, stable=True)[..., :k]
    values = torch.gather(input, -1, order)
    return values, order


@pytest.mark.parametrize("W, k", ((8192, 32),))  # multi-core band, 32-bit index CBs
@pytest.mark.parametrize("index_dtype", (ttnn.uint32, ttnn.int32))
@pytest.mark.parametrize("largest", (True, False))
def test_topk_stable_index_parity_32bit_preallocated(W, k, index_dtype, largest, device):
    """stable=True with preallocated 32-bit index outputs must keep the torch-stable
    tie order. Routes 32-bit index CBs through the multi-core band (W in [8192, 65535),
    pow2, k <= 64), i.e. the comparator-stable network with UInt32 index transport."""
    torch.manual_seed(1)
    shape = [1, 1, 32, W]
    # Few distinct values (negative and positive) -> many exact ties in every row.
    input = torch.randint(-4, 4, shape).to(torch.bfloat16)
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    out_shape = shape.copy()
    out_shape[-1] = k
    value_tensor = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
    )
    index_tensor = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.int32), index_dtype, layout=ttnn.Layout.TILE, device=device
    )

    ttnn_values, ttnn_indices = ttnn.topk(
        ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True, output_tensor=(value_tensor, index_tensor)
    )
    assert ttnn_indices.dtype == index_dtype

    ttnn_torch_values = ttnn.to_torch(ttnn_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64)

    assert_equal(golden_values, ttnn_torch_values)
    assert_equal(order, ttnn_torch_indices)


@pytest.mark.parametrize("W, k", ((64, 32), (8192, 32)))
@pytest.mark.parametrize("largest", (True, False))
def test_topk_stable_index_parity_float32(W, k, largest, device):
    """stable=True with FLOAT32 values must keep the torch-stable tie order. FLOAT32
    forces 32-bit index CBs and fp32 dest, exercising the INT32 index load/store arms
    of the comparator-stable network (single-core at W=64; W=8192 takes whatever the
    factory selects for fp32 in the multi-core band)."""
    torch.manual_seed(2)
    shape = [1, 1, 32, W]
    # Small integers, exactly representable in fp32 -> exact ties, both signs.
    input = torch.randint(-4, 4, shape).to(torch.float32)
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.float32, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)

    ttnn_torch_values = ttnn.to_torch(ttnn_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64)

    assert_equal(golden_values, ttnn_torch_values)
    assert_equal(order, ttnn_torch_indices)


@pytest.mark.parametrize("W, k", ((65536, 32), (65536, 16), (131072, 32)))
@pytest.mark.parametrize("largest", (True, False))
def test_topk_stable_index_parity_wide_u32(W, k, largest, device):
    """stable=True at W >= 65536: indices no longer fit 16 bits, so the auto-selected
    index dtype is UINT32 and the single-core path runs the RANK-STAMPED fast engine
    (sign-conditioned local-rank tags on the unstable network, true u32 indices riding
    index tracking) — bf16-family only; fp32 keeps the comparator. k > 32 runs the
    chain-rank stamped insertion cascade."""
    torch.manual_seed(3)
    shape = [1, 1, 32, W]
    input = torch.randint(-4, 4, shape).to(torch.bfloat16)
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)
    assert ttnn_indices.dtype == ttnn.uint32

    ttnn_torch_values = ttnn.to_torch(ttnn_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64)

    assert_equal(golden_values, ttnn_torch_values)
    assert_equal(order, ttnn_torch_indices)


@pytest.mark.parametrize("largest", (True, False))
def test_topk_stable_wide_tie_saturation(largest, device):
    """Tie-heavy rank-stamped coverage at W=65536: rows bulk-filled from the two middle
    levels (+-0.5), plus 12 scattered occurrences each of +-2.0 and +-1.0 at strided
    columns. In each direction the top-32 therefore spans THREE exact-tie groups —
    two extreme groups (12 elements each) entirely inside the top-k, and the k=32 cut
    landing inside the third (dominant ~32k-element) group — so every tie must break by
    ascending original index across the full 2048-tile insertion pipeline (accumulator
    vs incoming chunk at every step). Strict torch-stable parity on values and indices."""
    torch.manual_seed(5)
    W, k = 65536, 32
    shape = [1, 1, 32, W]
    levels = torch.tensor([-0.5, 0.5], dtype=torch.bfloat16)
    input = levels[torch.randint(0, 2, shape)]
    # 48 distinct strided columns (1291 * 47 < W, so no modular collisions), interleaved
    # across the four extreme levels -> 12 scattered occurrences each, spanning many
    # insertion chunks. Same columns in every row; ties still break per-row by index.
    extreme_cols = torch.arange(48) * 1291
    input[..., extreme_cols[0::4]] = 2.0
    input[..., extreme_cols[1::4]] = 1.0
    input[..., extreme_cols[2::4]] = -2.0
    input[..., extreme_cols[3::4]] = -1.0
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)
    assert ttnn_indices.dtype == ttnn.uint32

    assert_equal(golden_values, ttnn.to_torch(ttnn_values))
    assert_equal(order, ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64))


@pytest.mark.parametrize("largest", (True, False))
def test_topk_stable_wide_signed_zero(largest, device):
    """bf16 +-0.0 tie class at W=65536 on the rank-stamped engine: the local-position
    stamp folds -0.0 into +0.0 before tagging, so the whole zero group breaks by index
    like torch (which treats +-0 as one tie class). Zeros straddle the k cut for the
    smallest direction; normals cover the largest direction."""
    torch.manual_seed(6)
    W, k = 65536, 32
    shape = [1, 1, 32, W]
    input = torch.randn(shape, dtype=torch.bfloat16).abs() + 0.5  # positive normals
    # 48 zeros per row, alternating +0.0 / -0.0, scattered across the row (so the zero
    # tie group spans many insertion chunks and straddles k=32 for smallest).
    zero_cols = torch.arange(48) * 1291 % W
    input[..., zero_cols[0::2]] = 0.0
    input[..., zero_cols[1::2]] = -0.0
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)

    # torch.eq treats -0.0 == 0.0, so the values check is sign-insensitive by design
    # (the device canonicalizes -0.0 to +0.0 inside the compute).
    assert_equal(golden_values, ttnn.to_torch(ttnn_values))
    assert_equal(order, ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64))


@pytest.mark.parametrize("largest", (True, False))
@pytest.mark.parametrize("k", (64, 128))
def test_topk_stable_wide_cascade_parity(k, largest, device):
    """k > 32 at W=65536 runs the multi-tile insertion CASCADE (output_tiles = k/32)
    on the rank-stamped engine with CHAIN-RANK stamps: each level re-stamps its
    accumulator tile with that tile's round-start chain-position range while the
    loser tile's tags ride to the next level. Plain data-random parity."""
    torch.manual_seed(7)
    W = 65536
    shape = [1, 1, 32, W]
    input = torch.randint(-4, 4, shape).to(torch.bfloat16)
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)

    assert_equal(golden_values, ttnn.to_torch(ttnn_values))
    assert_equal(order, ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64))


@pytest.mark.parametrize("largest", (True, False))
@pytest.mark.parametrize("k", (64, 128))
def test_topk_stable_wide_cascade_displacement_ties(k, largest, device):
    """The cascade's adversarial tie pattern: rows dominated by the two middle tie
    levels (+-0.5) with the k cut landing INSIDE the dominant tie group, plus rare
    extreme spikes (+-2.0 / +-1.0) concentrated at HIGH columns. The late spikes
    displace long-accumulated low-index ties out of upper accumulator tiles mid-round,
    so displaced OLD elements (lower original index) meet NEWER accumulator entries at
    every lower cascade level — exactly the case naive per-level position stamps break
    (the displaced element would be ranked after the accumulator ties it must precede).
    Chain-rank stamps keep those ties in true index order; strict torch-stable parity."""
    torch.manual_seed(9)
    W = 65536
    shape = [1, 1, 32, W]
    levels = torch.tensor([-0.5, 0.5], dtype=torch.bfloat16)
    input = levels[torch.randint(0, 2, shape)]
    # 48 spike columns clustered in the top eighth of the row (all >= 7*W/8), strided to
    # stay distinct, interleaved across the four extreme levels. They arrive in the last
    # rounds of the insertion pipeline, after the accumulator chain is full of ties.
    spike_cols = W - 1 - torch.arange(48) * 157
    input[..., spike_cols[0::4]] = 2.0
    input[..., spike_cols[1::4]] = 1.0
    input[..., spike_cols[2::4]] = -2.0
    input[..., spike_cols[3::4]] = -1.0
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)
    assert ttnn_indices.dtype == ttnn.uint32

    assert_equal(golden_values, ttnn.to_torch(ttnn_values))
    assert_equal(order, ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64))


def test_topk_stable_wide_program_cache(device):
    """Program-cache behavior across the rank-stamp variants: W=65536 stable k=32
    (rank-stamped, single tile), W=65536 unstable (plain network) and W=65536
    stable k=64 (rank-stamped CASCADE, two output tiles) must be three DISTINCT
    cache entries — the gate inputs (stable flag, k) are covered by the program
    hash — and a cache-hit rerun of the rank-stamped program on fresh data must
    stay torch-stable-correct (guards the classic works-first-time /
    wrong-on-second-run failure mode)."""
    torch.manual_seed(8)
    W = 65536
    # 64 rows: a shape no other test in this module uses at this width, so the three
    # programs below are guaranteed fresh entries regardless of what ran earlier in
    # the same device session.
    shape = [1, 1, 64, W]

    def run(stable, k, seed):
        torch.manual_seed(seed)
        input = torch.randint(-4, 4, shape).to(torch.bfloat16)
        ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
        v, i = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True, stable=stable)
        return input, ttnn.to_torch(v), ttnn.to_torch(i, dtype=torch.int32).to(torch.int64)

    base_entries = device.num_program_cache_entries()

    inp, v, i = run(stable=True, k=32, seed=100)  # rank-stamped
    golden_v, golden_i = _stable_topk_golden(inp, 32, True)
    assert_equal(golden_v, v)
    assert_equal(golden_i, i)
    after_first = device.num_program_cache_entries()
    assert after_first > base_entries, "first rank-stamped run must MISS the program cache"

    run(stable=False, k=32, seed=101)  # unstable network: different program
    after_unstable = device.num_program_cache_entries()
    assert after_unstable > after_first, "stable and unstable topk must not share a cache entry"

    run(stable=True, k=64, seed=102)  # rank-stamped cascade (Ktiles=2): different program
    after_k64 = device.num_program_cache_entries()
    assert after_k64 > after_unstable, "k=32 and k=64 rank-stamped programs must not share a cache entry"

    # Cache-hit rerun of the rank-stamped program on fresh data.
    inp2, v2, i2 = run(stable=True, k=32, seed=103)
    assert device.num_program_cache_entries() == after_k64, "rerun must HIT the program cache"
    golden_v2, golden_i2 = _stable_topk_golden(inp2, 32, True)
    assert_equal(golden_v2, v2)
    assert_equal(golden_i2, i2)


@pytest.mark.parametrize("largest", (True, False))
def test_topk_stable_float32_signed_zero(largest, device):
    """FLOAT32 rows mixing +0.0 and -0.0: torch treats them as one tie class (broken by
    index); the sign-magnitude comparator must not split them into two classes."""
    torch.manual_seed(4)
    W, k = 64, 32
    shape = [1, 1, 32, W]
    input = torch.randn(shape, dtype=torch.float32)
    # Interleave +0.0 / -0.0 across half of each row, spanning the k boundary for both
    # directions (the rest are +/- normals).
    input[..., 0:32:2] = 0.0
    input[..., 1:32:2] = -0.0
    golden_values, order = _stable_topk_golden(input, k, largest)

    ttnn_input = ttnn.from_torch(input, ttnn.float32, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True, stable=True)

    ttnn_torch_values = ttnn.to_torch(ttnn_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_indices, dtype=torch.int32).to(torch.int64)

    # torch.eq treats -0.0 == 0.0, so the values check is sign-insensitive by design.
    assert_equal(golden_values, ttnn_torch_values)
    assert_equal(order, ttnn_torch_indices)


def test_topk_indices_tensor_dtype_mismatch_raise(device, expect_error):
    # A 32-bit indices_tensor with a bf16 input at W <= 65535 does not match the resolved
    # UInt16 index CB format: the reader would stream the UINT32 DRAM tiles as 16-bit pages
    # and produce garbage indices (silently feeding wrong index halves to the sort engines).
    # The op must reject the mismatch instead.
    torch.manual_seed(5)
    shape = [1, 1, 32, 8192]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_torch, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    indices_torch = torch.zeros(shape, dtype=torch.int32)
    for i in range(shape[-1]):
        indices_torch[..., i] = i
    indices_tensor = ttnn.from_torch(indices_torch, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    with expect_error(RuntimeError, "indices tensor"):
        ttnn.topk(ttnn_input, k=32, dim=-1, largest=True, sorted=True, indices_tensor=indices_tensor)
