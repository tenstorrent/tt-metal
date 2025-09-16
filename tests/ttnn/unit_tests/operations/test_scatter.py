# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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


def rand_permutations(shape, dim, dtype):
    r = torch.rand(*shape)
    return torch.argsort(r, dim=dim).to(dtype)


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

    torch_index = torch.randint(0, input_shape[dim], index_and_source_shape, dtype=torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=layout, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
    ttnn_result = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype


@pytest.mark.parametrize(
    "input_shape, dim, index_shape, source_shape, input_dtype, index_dtype, layout, expected_num_cache_entries",
    [
        ([100], -1, [80], [90], ttnn.bfloat16, ttnn.uint16, ttnn.Layout.TILE, 8),
        ([6, 8, 200], -1, [2, 5, 100], [3, 40, 1000], ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR, 2),
        ([1, 3 * 151936], -1, [1, 2 * 151936], [2, 5 * 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.ROW_MAJOR, 2),
        # ([1, 3 * 151936], -1, [1, 3 * 151936], [2, 4 * 151936], ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE, 2),
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

    torch_index = torch.randint(0, input_shape[dim], index_shape)
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
    assert device.num_program_cache_entries() == expected_num_cache_entries


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

    torch_index = rand_permutations(index_and_source_shape, dim, torch_index_dtype)
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


@pytest.mark.parametrize(
    "dim, input_shape, index_shape, source_shape, torch_dtype, input_dtype, index_dtype, source_dtype",
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
    device,
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

    with pytest.raises(RuntimeError):
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

    torch_index = torch.randint(0, input_shape[-1], index_and_source_shape, dtype=torch_index_dtype)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
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
