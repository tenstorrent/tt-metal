# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from math import ceil

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

layouts = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]
dtypes = [torch.bfloat16, torch.float32]
shapes = [(1,), (2,), (1, 2, 3), (3, 2, 1), (10, 9, 8, 7, 6), (32, 64, 4096)]

chunksize_list = [1, 2, 4, 5]
dims = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("chunksize", chunksize_list)
@pytest.mark.parametrize("dim", dims)
def test_split(device, layout, dtype, shape, chunksize, dim):
    if dim > len(shape) - 1:
        pytest.skip("dim greater than rank")

    torch_input_tensor = torch.rand(shape)
    torch_results = torch.split(torch_input_tensor, chunksize, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)

    outputs = ttnn.split(input_tensor, chunksize, dim=dim)
    outputs = [ttnn.to_torch(t) for t in outputs]

    assert len(outputs) == len(torch_results)
    for output, torch_result in zip(outputs, torch_results):
        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        assert_with_pcc(torch_result, output, 0.9999)


# test the very special case where we invoke a specialized kernel rather than use `ttnn::slice`
@pytest.mark.parametrize("dtype", dtypes)
def test_split_last_dim_kernel(device, dtype):
    shape, chunksize, dim = (64, 64), 32, 1
    if dim > len(shape) - 1:
        pytest.skip("dim greater than rank")

    torch_input_tensor = torch.rand(shape)
    torch_results = torch.split(torch_input_tensor, chunksize, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    outputs = ttnn.split(input_tensor, chunksize, dim=dim)
    outputs = [ttnn.to_torch(t) for t in outputs]

    assert len(outputs) == len(torch_results)
    for output, torch_result in zip(outputs, torch_results):
        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        assert_with_pcc(torch_result, output, 0.9999)


# Regression test: splitting a tiled tensor whose padded last-dim tile count
# is odd (e.g. 720 -> padded 736 -> 23 tiles, prime).  Previously the
# SplitDeviceOperation kernel path was entered with num_cores_c=1, causing a
# cb_wait_front deadlock because the writer consumed 2x the tiles the reader
# produced.  The fix falls back to slice-based split when the padded tile
# count is not divisible by 2.
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "shape,chunksize",
    [
        ((1, 1, 128, 720), 360),  # 720/32 -> padded 736 -> 23 tiles (prime)
        ((1, 1, 64, 352), 176),  # 352/32 -> padded 352 -> 11 tiles (prime)
        ((1, 1, 96, 160), 80),  # 160/32 -> 5 tiles (odd)
    ],
)
def test_split_odd_padded_tiles(device, dtype, shape, chunksize):
    """Verify two-chunk split on last dim with odd padded tile count."""
    dim = len(shape) - 1
    torch_input_tensor = torch.rand(shape, dtype=dtype)
    torch_results = torch.split(torch_input_tensor, chunksize, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    outputs = ttnn.split(input_tensor, chunksize, dim=dim)
    outputs = [ttnn.to_torch(t) for t in outputs]

    assert len(outputs) == len(torch_results)
    for output, torch_result in zip(outputs, torch_results):
        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", [(1, 256, 512, 512)])
@pytest.mark.parametrize("chunksize", [256])
@pytest.mark.parametrize("dim", [2, 3])
def test_split_large_inner_dims(device, layout, dtype, shape, chunksize, dim):
    torch_input_tensor = torch.rand(shape)
    torch_results = torch.split(torch_input_tensor, chunksize, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)

    outputs = ttnn.split(input_tensor, chunksize, dim=dim)
    outputs = [ttnn.to_torch(t) for t in outputs]

    assert len(outputs) == len(torch_results)
    for output, torch_result in zip(outputs, torch_results):
        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        assert_with_pcc(torch_result, output, 0.9999)
