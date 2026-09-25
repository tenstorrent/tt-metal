# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose


def select_torch_dtype(ttnn_dtype):
    if ttnn_dtype is ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype is ttnn.float32:
        return torch.float32
    if ttnn_dtype is ttnn.uint8:
        return torch.uint8
    if ttnn_dtype is ttnn.uint16:
        return torch.int64
    if ttnn_dtype is ttnn.int32:
        return torch.int64
    if ttnn_dtype is ttnn.uint32:
        return (
            torch.int64
        )  # !!! there is a strict requirement for the index tensor in Torch to be int64, and there is no int64 in ttnn


def rand_scatter_index(index_shape, dim, input_dim_size, dtype):
    """Indices unique along `dim`, drawn from the whole `[0, input_dim_size)` range.

    Uniqueness matters for any test that compares values: when two entries of the same index row
    point at the same output element, torch leaves the winning source element unspecified, so the
    comparison is only asserting that ttnn and torch happen to walk the row in the same order.
    Neither promises that.

    Permuting `index_shape[dim]` is not enough by itself. Where the index is shorter than the input
    along `dim` that only ever reaches the input's first `index_shape[dim]` positions, leaving the
    rest of the scatter axis untouched by the test. Permute the *input* extent instead and keep the
    first `index_shape[dim]` entries, which is unique and spans the full range.

    When `index_shape[dim] > input_dim_size` uniqueness is impossible by pigeonhole and the values
    wrap; callers detect that with `scatter_duplicates_unavoidable` and make the winner moot.
    """
    permuted_shape = list(index_shape)
    permuted_shape[dim] = max(input_dim_size, index_shape[dim])
    permuted = torch.argsort(torch.rand(*permuted_shape), dim=dim) % input_dim_size
    return permuted.narrow(dim, 0, index_shape[dim]).to(dtype).contiguous()


def scatter_duplicates_unavoidable(index_shape, dim, input_dim_size):
    """True when the index is longer than the input along `dim`, so no index can be unique there."""
    return index_shape[dim] > input_dim_size


def rand_scatter_source(source_shape, index_shape, dim, input_dim_size, dtype, index):
    """Random source values, or index-derived ones where duplicate indices cannot be avoided.

    With duplicates forced, several source elements compete for one output slot and torch leaves
    the winner unspecified. Deriving the value from the index makes every competitor for a slot
    carry the same value, so any winner gives the same answer - and unlike a single fill value it
    keeps distinct slots apart *within one lane along `dim`*, so a value landing in the wrong slot
    of its own lane is still caught.

    It does not tell one lane from another: the value depends on the index, not on where the index
    sits. The widest case here is the extreme of that - test_scatter_forge's [1, 1, 320, 320]
    against index [1, 1, 320, 384] gives every lane an index covering the whole 320-wide axis, so
    all 320 output rows come out identical (verified) and a lane-to-lane mapping error would be
    invisible. A positional term would not buy much, because bfloat16 already cannot keep 320
    values apart (289 distinct). Lane mapping is what the rank 5/6/8 fold tests cover: duplicates
    cannot occur there, so those keep a fully random source and every lane differs.

    Only the index-shaped prefix is overwritten, because that is all either side reads: ttnn slices
    the source down to the index's shape and torch indexes it at the index's coordinates.

    None of the above affects the agreement property, which holds for any mapping of index to value.
    """
    source = torch.randn(source_shape, dtype=dtype)
    if scatter_duplicates_unavoidable(index_shape, dim, input_dim_size):
        source[tuple(slice(0, extent) for extent in index_shape)] = index.to(dtype) * 0.5
    return source


@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout",
    [
        ([1], 0, [1], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([100], 0, [80], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 1, 20, 20, 200], -1, [1, 1, 20, 20, 20], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
        ([2, 2, 2, 2, 2, 2, 2, 2], -1, [2, 2, 2, 2, 2, 2, 2, 2], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([10, 1, 10, 1, 10], 0, [10, 1, 10, 1, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([1, 128256], -1, [1, 128256], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([50, 200], 0, [50, 200], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([10, 10, 10, 10, 10], 2, [10, 10, 10, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        ([10, 10, 10, 10, 10], 2, [10, 10, 10, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        ([50, 200], 0, [50, 200], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        ##################
        # these cases fail due to the to_layout precision issue (fp32 tiled <-> row-major) : #23405
        # ([10, 50, 10, 50, 100], -1, [10, 50, 10, 50, 100], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([10, 50, 10, 50, 100], 0, [10, 50, 10, 50, 100], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([2, 30, 200], 0, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        ##################
        # these cases fail due to the to_layout integer issue (integer dtype size>256 tiled -> row-major): #23407
        # ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([100, 151936], -1, [100, 151936], ttnn.float32, ttnn.int32, ttnn.Layout.TILE),
        # ([2, 10, 151936], -1, [2, 10, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([1, 151936], -1, [1, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.TILE),
        # ([100, 151936], -1, [100, 151936], ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE),
        # ([2, 10, 151936], -1, [2, 10, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.TILE),
    ],
)
def test_scatter_spec(input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout, device):
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)
    torch_index_dtype = select_torch_dtype(index_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    # Unique indices spanning the input's full extent along dim: this matrix is value-checked
    # below, and a repeated index leaves the winning source element unspecified in torch.
    torch_index = rand_scatter_index(index_and_source_shape, dim, input_shape[dim], torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    # Values too, not just the spec: the rank-5/6/8 shapes above are the ones a fold-style
    # regression would hit, and asserting only shape/dtype is what let #56876 through.
    if torch_dtype is torch.float32:
        assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
    else:
        assert_allclose(torch_result_from_ttnn, torch_result)


@pytest.mark.parametrize(
    "input_shape, dim, index_shape, source_shape, input_dtype, index_dtype, layout, expected_num_cache_entries",
    [
        ([100], -1, [80], [90], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, 8),
        ([6, 8, 200], -1, [2, 5, 100], [3, 40, 1000], ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR, 2),
        ([1, 3 * 151936], -1, [1, 2 * 151936], [2, 5 * 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, 2),
        # ([1, 3 * 151936], -1, [1, 3 * 151936], [2, 4 * 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, 2),
        ([2, 2, 100000], 0, [1, 2, 80000], [4, 4, 80001], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, 6),
        (
            [2, 2, 100000],
            1,
            [1, 2, 79000],
            [4, 4, 180001],
            ttnn.bfloat16,
            ttnn.int32,
            ttnn.Layout.ROW_MAJOR,
            6,
        ),
        ([50, 20], 0, [50, 20], [200, 80], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, 5),
        ([10, 10, 10], 1, [2, 30, 10], [2, 30, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, 8),
        ([10, 30, 6, 10], -1, [2, 30, 6, 5], [2, 30, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, 2),
        ([10, 30, 6, 10], 2, [2, 30, 6, 5], [2, 30, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, 6),
        ([50, 200], 0, [49, 199], [51, 201], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, 10),
        ([10, 20], 0, [9, 19], [11, 21], ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE, 10),
    ],
)
def test_scatter_partial(
    input_shape, dim, index_shape, source_shape, input_dtype, index_dtype, layout, expected_num_cache_entries, device
):
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    torch_index = rand_scatter_index(index_shape, dim, input_shape[dim], torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    # [10, 10, 10] / dim=1 / index [2, 30, 10] asks for 30 unique indices along an axis of 10.
    torch_src = rand_scatter_source(source_shape, index_shape, dim, input_shape[dim], torch_dtype, torch_index)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    if torch_dtype is torch.float32:
        assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
    else:
        assert_allclose(torch_result_from_ttnn, torch_result)
    assert device.num_program_cache_entries() == expected_num_cache_entries


# A rank > 4 input used to be collapsed to 4D by merging its leading (rank - 3) dims, and the index
# tensor was collapsed the same way but with its own extents. The reader maps an input stick
# coordinate straight onto the index tensor's axes, so once the leading dims were fused into one
# linear id the two no longer referred to the same element whenever any interior leading dim
# differed between input and index - scatter then wrote the wrong source row and skipped a valid one,
# silently. Legal input: scatter only requires index_shape[d] <= input_shape[d] for d != dim.
# See issue #56876.
@pytest.mark.parametrize(
    "input_shape, dim, index_shape, source_shape",
    [
        # dim == -1, so no transpose: the leading dims reach the kernel exactly as given.
        ([2, 3, 4, 5, 6], -1, [2, 2, 4, 5, 6], [2, 2, 4, 5, 6]),  # one interior leading dim differs
        ([2, 3, 4, 5, 6], -1, [2, 2, 3, 5, 6], [2, 2, 3, 5, 6]),  # two interior leading dims differ
        ([4, 5, 6, 7, 8], -1, [3, 2, 3, 7, 8], [3, 2, 3, 7, 8]),  # every leading dim differs
        ([2, 3, 4, 5, 6], -1, [2, 2, 4, 5, 6], [3, 4, 5, 6, 7]),  # source larger than index
        ([2, 3, 4, 5, 6, 7], -1, [2, 3, 2, 5, 6, 7], [2, 3, 2, 5, 6, 7]),  # rank 6
        ([2, 3, 2, 3, 2, 3, 2, 3], -1, [2, 2, 2, 2, 2, 2, 2, 3], [2, 2, 2, 2, 2, 2, 2, 3]),  # rank 8
        # dim != -1, so the op transposes dim to the last axis first - the mismatched leading dim
        # ends up in a different position than it started in. Covered above rank 5 too, since the
        # transpose interacts with how many leading axes there are to walk.
        ([2, 3, 4, 5, 6], 2, [2, 2, 4, 5, 6], [2, 2, 4, 5, 6]),
        ([2, 3, 4, 5, 6], 0, [2, 2, 4, 5, 6], [2, 2, 4, 5, 6]),
        ([2, 3, 4, 5, 6], -2, [2, 2, 4, 5, 6], [2, 2, 4, 5, 6]),
        # The mismatched dim has to sit somewhere other than the axis `dim` names: dim=2 against an
        # index that differs at axis 2 transposes the mismatch into the scatter axis, which is
        # excluded from the leading-dim comparison, so the fold lines up again and the case proves
        # nothing (verified - it passes unfixed). Mismatch at axis 1 for dim=2.
        ([2, 3, 4, 5, 6, 7], 2, [2, 2, 4, 5, 6, 7], [2, 2, 4, 5, 6, 7]),  # rank 6, transposed
        ([2, 3, 4, 5, 6, 7], 0, [2, 3, 2, 5, 6, 7], [2, 3, 2, 5, 6, 7]),  # rank 6, transposed
        ([2, 3, 2, 3, 2, 3, 2, 3], 3, [2, 2, 2, 2, 2, 2, 2, 3], [2, 2, 2, 2, 2, 2, 2, 3]),  # rank 8, transposed
        # Interior leading dim of 1 on the index side: broadcast-looking but not broadcast.
        ([2, 3, 4, 5, 6], -1, [2, 1, 4, 5, 6], [2, 1, 4, 5, 6]),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, index_dtype, layout",
    [
        (ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR),
        (ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR),
        (ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE),
    ],
)
def test_scatter_high_rank_unequal_leading_dims(
    input_shape, dim, index_shape, source_shape, input_dtype, index_dtype, layout, device
):
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)
    torch_index_dtype = select_torch_dtype(index_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    torch_index = rand_scatter_index(index_shape, dim, input_shape[dim], torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randn(source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    if torch_dtype is torch.float32:
        assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
    else:
        assert_allclose(torch_result_from_ttnn, torch_result)


# pre_scatter_transform_tensor runs once per operand, so a Shape{1} operand used to return early
# and reach the device op at rank 1 while its siblings were padded to rank 4. The reader sizes its
# shape-vararg reads from the input rank alone, so it then read index_dims past the end of the block
# the factory wrote, in_bounds() failed against the stale values and the scatter was skipped for
# every stick - the op returned the input unchanged. Legal input: with dim == 0 on a rank-1 tensor
# there is no d != dim, so a size-1 index against a size-100 input passes every check.
@pytest.mark.parametrize(
    "input_shape, dim, index_shape, source_shape",
    [
        ([100], 0, [1], [1]),  # index and source are Shape{1}, input is not
        ([100], -1, [1], [1]),  # same, negative dim
        ([100], 0, [1], [5]),  # only index is Shape{1}
        ([1], 0, [5], [5]),  # only input is Shape{1}
        ([1], 0, [1], [1]),  # every operand is Shape{1}
        ([1], -1, [1], [1]),  # same, negative dim
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.float32])
def test_scatter_singleton_operand(input_shape, dim, index_shape, source_shape, input_dtype, device):
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.Layout.ROW_MAJOR, device=device)

    torch_index = rand_scatter_index(index_shape, dim, input_shape[dim], torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.Layout.ROW_MAJOR, device=device)

    # [1] / index [5] asks for 5 unique indices along an axis of 1: every source element targets
    # output element 0, so the winner has to stop mattering. With a single slot the index-derived
    # values collapse to one value on their own.
    torch_src = rand_scatter_source(source_shape, index_shape, dim, input_shape[dim], torch_dtype, torch_index)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=ttnn.Layout.ROW_MAJOR, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    if torch_dtype is torch.float32:
        assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
    else:
        assert_allclose(torch_result_from_ttnn, torch_result)


# Same defect, exercised through the separate bfloat16 reduction program factory / reader kernel,
# which carries its own copy of the coordinate walk. See issue #56876.
@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape",
    [
        ([2, 3, 4, 5, 6], -1, [2, 2, 4, 5, 6]),
        ([2, 3, 4, 5, 6], 2, [2, 2, 4, 5, 6]),
        ([2, 3, 4, 5, 6, 7], -1, [2, 3, 2, 5, 6, 7]),
        # mismatch at axis 1, not axis 2 - see the note in the non-reduction test above
        ([2, 3, 4, 5, 6, 7], 2, [2, 2, 4, 5, 6, 7]),
    ],
)
@pytest.mark.parametrize("reduction", ["add", "multiply"])
def test_scatter_reduction_high_rank_unequal_leading_dims(input_shape, dim, index_and_source_shape, reduction, device):
    # Deliberately randint rather than rand_scatter_index, unlike every non-reduction test in this
    # file. Duplicate indices are the whole point here: they are what makes the reduce path combine
    # anything at all. Unique indices write each output slot exactly once, which turns add and
    # multiply into plain assignment - measured over these four cases, randint reduces 125, 128,
    # 678 and 883 slots respectively while unique indices reduce zero. Duplicates are safe to keep
    # because add and multiply are order-independent, so there is no unspecified winner to depend
    # on the way a plain scatter would have.
    torch.manual_seed(0)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR, device=device)

    torch_index = torch.randint(0, input_shape[dim], index_and_source_shape, dtype=torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.Layout.ROW_MAJOR, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch.bfloat16)
    ttnn_src = ttnn.from_torch(torch_src, dtype=ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src, reduce=reduction)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src, reduce=reduction)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    assert_allclose(torch_result_from_ttnn, torch_result)


@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout, expected_num_cache_entries",
    [
        ([100], -1, [80], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, 5),
        ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR, 1),
        ([1, 1, 20, 20, 200], -1, [1, 1, 20, 20, 20], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, 5),
        ([2, 2, 2, 2, 2, 2, 2, 2], -1, [2, 2, 2, 2, 2, 2, 2, 2], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR, 1),
        ([10, 1, 10, 1, 10], 0, [10, 1, 10, 1, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR, 3),
        ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, 1),
        ([50, 20], 0, [50, 20], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, 4),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, 6),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, 3),
        ([10, 10, 10, 10, 10], 2, [10, 10, 10, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, 6),
        ([10, 10, 10, 10, 10], 2, [10, 10, 10, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, 3),
        ([50, 200], 0, [50, 200], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, 7),
        ##################
        # these cases fail due to the to_layout precision issue (fp32 tiled <-> row-major) : #23405
        # ([10, 50, 10, 50, 100], -1, [10, 50, 10, 50, 100], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([10, 50, 10, 50, 100], 0, [10, 50, 10, 50, 100], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        # ([2, 30, 200], 0, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.TILE),
        ##################
        # these cases fail due to the to_layout integer issue (integer dtype size>256 tiled -> row-major): #23407
        # ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([100, 151936], -1, [100, 151936], ttnn.float32, ttnn.int32, ttnn.Layout.TILE),
        # ([2, 10, 151936], -1, [2, 10, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        # ([1, 151936], -1, [1, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.TILE),
        # ([100, 151936], -1, [100, 151936], ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE),
        # ([2, 10, 151936], -1, [2, 10, 151936], ttnn.float32, ttnn.uint32, ttnn.Layout.TILE),
    ],
)
def test_scatter_normal_with_callback(
    input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout, expected_num_cache_entries, device
):
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)
    torch_index_dtype = select_torch_dtype(index_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    torch_index = rand_scatter_index(index_and_source_shape, dim, input_shape[dim], torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    for _ in range(2):
        torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
        ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

        torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
        assert torch_result_from_ttnn.shape == torch_result.shape
        assert torch_result_from_ttnn.dtype == torch_result.dtype
        if torch_dtype is torch.float32:
            assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
        else:
            assert_allclose(torch_result_from_ttnn, torch_result)
    assert device.num_program_cache_entries() == expected_num_cache_entries


##### !!!! WARNING !!!! #####
##### DO NOT FEED CORE RANGE SETS CONTAINING ONLY **ONE** CORE INSIDE - split_work_to_cores DOES NOT HANDLE THAT GRACEFULLY!!!
@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout, sub_core_grids, expected_num_cache_entries",
    [
        ([100], -1, [80], ttnn.int32, ttnn.uint16, ttnn.Layout.ROW_MAJOR, None, 1),
        (
            [2, 30, 200],
            -1,
            [2, 30, 200],
            ttnn.int32,
            ttnn.uint16,
            ttnn.Layout.ROW_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
            1,
        ),
        (
            [1, 1, 20, 20, 200],
            -1,
            [1, 1, 20, 20, 20],
            ttnn.int32,
            ttnn.uint16,
            ttnn.Layout.ROW_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 2), ttnn.CoreCoord(4, 3)),
                ]
            ),
            1,
        ),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.int32, ttnn.int32, ttnn.Layout.ROW_MAJOR, None, 2),
        (
            [10, 10, 10, 10, 10],
            2,
            [10, 10, 10, 10, 10],
            ttnn.int32,
            ttnn.int32,
            ttnn.Layout.ROW_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 4), ttnn.CoreCoord(3, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(6, 5)),
                ]
            ),
            2,
        ),
        (
            [50, 200],
            0,
            [50, 200],
            ttnn.int32,
            ttnn.int32,
            ttnn.Layout.ROW_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 4), ttnn.CoreCoord(3, 5)),
                ]
            ),
            3,
        ),
        (
            [32, 128 * 1024],
            1,
            [32, 128 * 1024],
            ttnn.int32,
            ttnn.int32,
            ttnn.Layout.ROW_MAJOR,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
            1,
        ),
    ],
)
def test_scatter_reduction_row_major_int32_with_callback_and_sub_cores(
    input_shape,
    dim,
    index_and_source_shape,
    input_dtype,
    index_dtype,
    layout,
    sub_core_grids,
    expected_num_cache_entries,
    device,
):
    torch.manual_seed(0)

    # randint, not rand_scatter_index: scatter_add needs duplicate indices to exercise the
    # accumulate path - see test_scatter_reduction_high_rank_unequal_leading_dims.
    torch_dtype = torch.float32

    torch_input = torch.randint(0, input_shape[dim], input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    torch_index = torch.randint(0, input_shape[dim], index_and_source_shape, dtype=torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randint(0, input_shape[dim], index_and_source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    for _ in range(2):
        torch_result = torch.scatter_add(torch_input, dim, index=torch_index, src=torch_src)
        ttnn_result = ttnn.scatter_add(ttnn_input, dim, ttnn_index, ttnn_src, sub_core_grids=sub_core_grids)

        torch_result_from_ttnn = ttnn.to_torch(ttnn_result).to(torch.int64)
        assert torch_result_from_ttnn.shape == torch_result.shape
        if torch_dtype is torch.float32:
            assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
        else:
            assert_allclose(torch_result_from_ttnn, torch_result)
    assert device.num_program_cache_entries() == expected_num_cache_entries


@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape, input_dtype, index_dtype, layout, reduction, expected_num_cache_entries",
    [
        ([2, 30, 200], -1, [2, 30, 200], ttnn.float32, ttnn.uint16, ttnn.Layout.ROW_MAJOR, "add", 1),
        (
            [2, 2, 2, 2, 2, 2, 2, 2],
            -1,
            [2, 2, 2, 2, 2, 2, 2, 2],
            ttnn.float32,
            ttnn.uint16,
            ttnn.Layout.ROW_MAJOR,
            "multiply",
            1,
        ),
        ([50, 20], 0, [50, 20], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, "add", 4),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, "add", 3),
        ([10, 10, 10, 10, 10], 2, [10, 10, 10, 10, 10], ttnn.float32, ttnn.int32, ttnn.Layout.ROW_MAJOR, "multiply", 3),
        ([100], -1, [80], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, "add", 5),
        ([1, 1, 20, 20, 200], -1, [1, 1, 20, 20, 20], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, "add", 5),
        ([10, 1, 10, 1, 10], 0, [10, 1, 10, 1, 10], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.ROW_MAJOR, "multiply", 3),
        ([1, 151936], -1, [1, 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, "add", 1),
        ([10, 10, 10, 10, 10], 0, [10, 10, 10, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, "multiply", 6),
        ([10, 10, 10, 10, 10], 2, [10, 10, 10, 10, 10], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, "add", 6),
        ([50, 200], 0, [50, 200], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, "add", 7),
    ],
)
def test_scatter_reduction(
    input_shape,
    dim,
    index_and_source_shape,
    input_dtype,
    index_dtype,
    layout,
    reduction,
    expected_num_cache_entries,
    device,
):
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=layout, device=device)

    # randint, not rand_scatter_index: reduce=add/multiply needs duplicates to combine anything -
    # see test_scatter_reduction_high_rank_unequal_leading_dims.
    torch_index = torch.randint(0, input_shape[dim], index_and_source_shape)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src, reduce=reduction)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src, reduce=reduction)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    if torch_dtype is torch.float32:
        assert_allclose(torch_result_from_ttnn, torch_result, atol=0.1, rtol=1e-2)
    else:
        assert_allclose(torch_result_from_ttnn, torch_result)
    assert device.num_program_cache_entries() == expected_num_cache_entries


@pytest.mark.parametrize("index_dtype, max_index", [(ttnn.uint16, 2**16), (ttnn.uint8, 2**8)])
def test_scatter_reduction_bf16_narrow_index_multi_chunk(index_dtype, max_index, device):
    # Regression: the bf16 reduction reader typed the chunk offset as the index dtype, so on sticks
    # longer than one chunk a narrow index truncated it and re-reduced indices from earlier chunks.
    torch.manual_seed(0)
    input_shape, index_shape = [1, 200000], [1, 50000]

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, max_index, index_shape)
    torch_src = torch.randn(index_shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_src = ttnn.from_torch(torch_src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_result = torch.scatter(torch_input, -1, index=torch_index, src=torch_src, reduce="add")
    ttnn_result = ttnn.scatter(ttnn_input, -1, ttnn_index, ttnn_src, reduce="add")

    assert_allclose(ttnn.to_torch(ttnn_result), torch_result)


@pytest.mark.parametrize(
    "dim, input_shape, index_shape, source_shape, torch_dtype, input_dtype, index_dtype, source_dtype, expected_message",
    [
        (
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.uint16,
            ttnn.bfloat16,
            "is out of range for tensor rank",
        ),  # input_rank vs dim
        (
            -10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.uint16,
            ttnn.bfloat16,
            "is out of range for tensor rank",
        ),  # input_rank vs dim
        (
            0,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.uint16,
            ttnn.bfloat16,
            "input_rank must be equal to index_rank",
        ),  # index_shape vs source_shape
        (
            0,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            "index_dtype is not integer",
        ),  # index_dtype is integer
    ],
)
def test_scatter_failing_cases(
    dim,
    input_shape,
    index_shape,
    source_shape,
    torch_dtype,
    input_dtype,
    index_dtype,
    source_dtype,
    expected_message,
    device,
    expect_error,
):
    torch.manual_seed(0)
    torch_index_dtype = select_torch_dtype(index_dtype)
    torch_source_dtype = select_torch_dtype(source_dtype)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    max_range = input_shape[dim] if (-len(input_shape) <= dim and dim < len(input_shape)) else 1
    torch_index = torch.randint(0, max_range, index_shape, dtype=torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_src = torch.randn(source_shape, dtype=torch_source_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=source_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, expected_message):
        ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)


@pytest.mark.parametrize(
    "input_shape, index_and_source_shape",
    [
        ([1, 1, 32, 32], [1, 1, 32, 32]),
        ([1, 1, 320, 384], [1, 1, 320, 384]),
        ([1, 3, 32, 32], [1, 3, 32, 32]),
        ([1, 1, 32, 32], [1, 1, 64, 64]),
        ([1, 1, 320, 320], [1, 1, 320, 384]),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_scatter_forge(input_shape, index_and_source_shape, input_dtype, device):
    import math

    if math.prod(input_shape[:-1]) != math.prod(index_and_source_shape[:-1]):
        pytest.xfail(
            f"unsupported shapes configuration: input_shape has a non-last dimension of a different length than index_and_source_shape ({math.prod(input_shape[:-1])} vs {math.prod(index_and_source_shape[:-1])})"
        )
    torch.manual_seed(0)
    torch_dtype = select_torch_dtype(input_dtype)
    torch_index_dtype = select_torch_dtype(ttnn.int32)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_index = rand_scatter_index(index_and_source_shape, -1, input_shape[-1], torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # [1, 1, 320, 320] against index [1, 1, 320, 384] asks for 384 unique indices along an axis
    # of 320, so the source must not depend on which duplicate wins.
    torch_src = rand_scatter_source(
        index_and_source_shape, index_and_source_shape, -1, input_shape[-1], torch_dtype, torch_index
    )
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_result = torch.scatter(torch_input, -1, index=torch_index, src=torch_src)
    ttnn_result = ttnn.scatter(ttnn_input, -1, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    if torch_dtype is torch.float32:
        assert_allclose(torch_result_from_ttnn, torch_result, rtol=1e-3)
    else:
        assert_allclose(torch_result_from_ttnn, torch_result)


@pytest.mark.parametrize("shape", [(100,), (4, 128), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_negative_dim(device, shape, dim, dtype):
    """
    Regression test for negative dimension support in ttnn.scatter.
    Verifies that negative dimension values work correctly and match PyTorch behavior.
    Includes 1D tensors to cover the logical_shape vs padded_shape bug (PR #41762).
    """
    # Skip invalid dim combinations (e.g., dim=-2 for 1D tensor)
    if abs(dim) > len(shape):
        pytest.skip(f"dim={dim} invalid for rank={len(shape)} tensor")

    torch_input = torch.rand(shape, dtype=dtype)

    # Create index and source tensors for scattering
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)  # Scatter subset
    torch_index = rand_scatter_index(index_shape, dim, shape[dim], torch.int32)
    torch_source = torch.rand(index_shape, dtype=dtype)

    # PyTorch reference with negative dim
    torch_output_neg = torch_input.clone()
    torch_output_neg.scatter_(dim, torch_index.long(), torch_source)

    # Convert to ttnn using the matching test dtype
    torch_to_ttnn_dtype = {
        torch.bfloat16: ttnn.bfloat16,
        torch.float32: ttnn.float32,
    }
    ttnn_dtype = torch_to_ttnn_dtype[dtype]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_source = ttnn.from_torch(torch_source, device=device, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with negative dim
    ttnn_output = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_source)
    output = ttnn.to_torch(ttnn_output)

    assert (
        output.shape == torch_output_neg.shape
    ), f"Output shape {output.shape} does not match expected {torch_output_neg.shape}"
    # Use tighter tolerance for float32 as per existing scatter tests
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    assert_allclose(torch_output_neg, output, rtol=rtol)


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((100,), -1),  # 1D tensor (regression for PR #41762)
        ((4, 128), -1),  # Last dimension
        ((4, 128), -2),  # First dimension
        ((2, 3, 4), -1),  # 3D tensor, last dim
        ((2, 3, 4), -2),  # 3D tensor, middle dim
        ((2, 3, 4), -3),  # 3D tensor, first dim
    ],
)
def test_scatter_negative_dim_equals_positive(device, shape, dim):
    """
    Verify that negative and positive dim produce identical results.
    Includes 1D tensors to cover the logical_shape vs padded_shape bug (PR #41762).
    """
    positive_dim = len(shape) + dim

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)
    torch_index = rand_scatter_index(index_shape, dim, shape[dim], torch.int32)
    torch_source = torch.rand(index_shape, dtype=torch.bfloat16)

    # Get results for both negative and positive dims
    torch_output_neg = torch_input.clone()
    torch_output_neg.scatter_(dim, torch_index.long(), torch_source)

    torch_output_pos = torch_input.clone()
    torch_output_pos.scatter_(positive_dim, torch_index.long(), torch_source)

    # Verify PyTorch results match
    assert torch.allclose(torch_output_neg, torch_output_pos)

    # Test ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_source = ttnn.from_torch(torch_source, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn_output_neg = ttnn.to_torch(ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_source))

    ttnn_input_pos = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output_pos = ttnn.to_torch(ttnn.scatter(ttnn_input_pos, positive_dim, ttnn_index, ttnn_source))

    assert_allclose(ttnn_output_neg, ttnn_output_pos, rtol=1e-3)
    assert_allclose(torch_output_neg, ttnn_output_neg, rtol=1e-2)


@pytest.mark.parametrize(
    "shape,index_shape",
    [
        ((100,), (80,)),  # 1D tensor
        ((50,), (50,)),  # 1D tensor, full scatter
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_scatter_1d_tile_layout_negative_dim(device, shape, index_shape, dtype):
    """
    Regression test for PR #41762: 1D tensors in TILE_LAYOUT with negative dim.

    The bug was that dim normalization used padded_shape().rank() instead of
    logical_shape().rank(). For a 1D tensor [100] in TILE_LAYOUT, the padded
    shape becomes [1, 128] (rank 2), causing dim=-1 to normalize to 1 instead
    of 0, which then fails validation.
    """
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    torch_input = torch.randn(shape, dtype=torch_dtype)
    torch_index = rand_scatter_index(index_shape, -1, shape[0], torch.int64)
    torch_src = torch.randn(index_shape, dtype=torch_dtype)

    # PyTorch reference with dim=-1 (should be equivalent to dim=0 for 1D)
    torch_result = torch.scatter(torch_input, dim=-1, index=torch_index, src=torch_src)

    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_src = ttnn.from_torch(torch_src, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # This would fail before PR #41762 with:
    # "dim must follow the condition -input_rank <= dim < input_rank (dim: 1, rank: 1)"
    ttnn_result = ttnn.scatter(ttnn_input, -1, ttnn_index, ttnn_src)
    result = ttnn.to_torch(ttnn_result)

    assert result.shape == torch_result.shape
    assert_allclose(result, torch_result)
