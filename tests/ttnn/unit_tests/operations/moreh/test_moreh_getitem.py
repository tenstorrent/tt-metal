# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc, skip_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_getitem import (
    run_getitem_RAW_MAJOR,
    run_moreh_geitem_tilized_one_index,
)

pytestmark = pytest.mark.use_module_device

# A ROW_MAJOR input runs the Rm factory, which can't index the last (W) dim. A TILE input runs the Tilized factory,
# which builds separate kernels when W is indexed, and a ROW_MAJOR_INDEX or TILIZE_INDEX variant per index layout.
# Every nightly getitem test is skipped on Blackhole (#12349); the Tilized cases here are too.


def run_moreh_getitem_test(
    input_shape, layout, device, index_dims=(0,), index_size=4, dtype=ttnn.bfloat16, row_major_index=True
):
    # Not the nightly helpers: they take one index dim only.
    torch_dtype = torch.int32 if dtype == ttnn.int32 else torch.bfloat16
    torch_input = torch.randint(0, 10, input_shape, dtype=torch_dtype)
    torch_indices = [torch.randint(-input_shape[dim], input_shape[dim] - 1, (index_size,)) for dim in index_dims]
    torch_output = torch_input[(slice(None),) * index_dims[0] + tuple(torch_indices)]

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, pad_value=float("nan"))
    if row_major_index:
        tt_indices = [ttnn.from_torch(index, dtype=ttnn.int32, device=device) for index in torch_indices]
    else:
        tt_indices = [
            ttnn.from_torch(index.reshape(1, index_size), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
            for index in torch_indices
        ]
    tt_output = ttnn.to_torch(ttnn.moreh_getitem(tt_input, tt_indices, list(index_dims)))

    if layout == ttnn.TILE_LAYOUT:
        # A TILE output keeps the input's rank: each indexed dim becomes 1 and the last one the index length, so
        # [10, 5, 7, 70] indexed on [2, 3] gives [10, 5, 1, 4] where torch gives [10, 5, 4].
        assert tt_output.numel() == torch_output.numel(), tt_output.shape
        tt_output = tt_output.reshape(torch_output.shape)
    assert tt_output.shape == torch_output.shape, tt_output.shape
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape_index_dim, dtype, index_size",
    [
        # A rank-5 input indexed on each dim the Rm factory allows: N, C, D, H.
        ([[10, 2, 5, 7, 70], 0], torch.bfloat16, 4),
        ([[10, 2, 5, 7, 70], 1], torch.bfloat16, 4),
        ([[10, 2, 5, 7, 70], 2], torch.bfloat16, 4),
        ([[10, 2, 5, 7, 70], 3], torch.bfloat16, 4),
        # A rank-2 input is padded to 5 dims on the host.
        ([[10, 70], 0], torch.bfloat16, 4),
        ([[10, 5, 70], 1], torch.int32, 4),
        ([[10, 70], 0], torch.bfloat16, 100),
    ],
    ids=["n", "c", "d", "h", "rank_2", "int32", "index_size_100"],
)
def test_moreh_getitem_row_major(shape_index_dim, dtype, index_size, device):
    torch.manual_seed(0)
    run_getitem_RAW_MAJOR(shape_index_dim, dtype, index_size, device)


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape_index_dim, row_major_index",
    [
        ([[10, 5, 64], 1], True),
        ([[7, 70], 0], False),
        # Indexing the last dim runs the *_tilize_w kernels, which pick elements out of 16-wide tile faces.
        ([[1, 5, 7, 3, 80], 4], True),
        ([[5, 64], 1], False),
    ],
    ids=["h_row_major_index", "h_tile_index", "w_row_major_index", "w_tile_index"],
)
def test_moreh_getitem_tilized(shape_index_dim, row_major_index, device):
    torch.manual_seed(0)
    run_moreh_geitem_tilized_one_index(shape_index_dim, torch.bfloat16, 4, row_major_index, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, layout, index_dims",
    [
        ([10, 3, 5, 7, 80], ttnn.ROW_MAJOR_LAYOUT, [2, 3]),
        ([10, 15, 7, 80], ttnn.ROW_MAJOR_LAYOUT, [0, 1, 2]),
        pytest.param(
            [10, 5, 7, 70],
            ttnn.TILE_LAYOUT,
            [2, 3],
            marks=skip_for_blackhole("Mismatching on Blackhole, see #12349"),
        ),
    ],
    ids=["row_major_two_indices", "row_major_three_indices", "tile_two_indices_with_w"],
)
def test_moreh_getitem_multi_index(input_shape, layout, index_dims, device):
    torch.manual_seed(0)
    run_moreh_getitem_test(input_shape, layout, device, index_dims=index_dims)
