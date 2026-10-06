# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc, skip_for_blackhole

pytestmark = pytest.mark.use_module_device


def run_moreh_getitem_test(
    input_shape, layout, device, index_dims=(0,), index_size=4, dtype=ttnn.bfloat16, row_major_index=True
):
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

    assert tt_output.shape == torch_output.shape, tt_output.shape
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, layout",
    [
        ([10, 70], ttnn.ROW_MAJOR_LAYOUT),
        pytest.param([7, 70], ttnn.TILE_LAYOUT, marks=skip_for_blackhole("Mismatching on Blackhole, see #12349")),
    ],
    ids=["row_major", "tile"],
)
def test_moreh_getitem(input_shape, layout, device):
    torch.manual_seed(0)
    run_moreh_getitem_test(input_shape, layout, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, layout, index_dims, index_size, dtype, row_major_index",
    [
        # ROW_MAJOR input can't be indexed on its last (W) dim.
        ([10, 5, 70], ttnn.ROW_MAJOR_LAYOUT, [1], 4, ttnn.bfloat16, True),
        ([10, 5, 7, 70], ttnn.ROW_MAJOR_LAYOUT, [2], 4, ttnn.bfloat16, True),
        ([10, 2, 5, 7, 70], ttnn.ROW_MAJOR_LAYOUT, [3], 4, ttnn.bfloat16, True),
        ([10, 3, 5, 7, 80], ttnn.ROW_MAJOR_LAYOUT, [2, 3], 4, ttnn.bfloat16, True),
        ([10, 15, 7, 80], ttnn.ROW_MAJOR_LAYOUT, [0, 1, 2], 4, ttnn.bfloat16, True),
        ([10, 70], ttnn.ROW_MAJOR_LAYOUT, [0], 100, ttnn.bfloat16, True),
        ([10, 5, 70], ttnn.ROW_MAJOR_LAYOUT, [1], 4, ttnn.int32, True),
        pytest.param(
            [10, 5, 64],
            ttnn.TILE_LAYOUT,
            [1],
            4,
            ttnn.bfloat16,
            True,
            marks=skip_for_blackhole("Mismatching on Blackhole, see #12349"),
        ),
        pytest.param(
            [7, 70],
            ttnn.TILE_LAYOUT,
            [0],
            4,
            ttnn.bfloat16,
            False,
            marks=skip_for_blackhole("Mismatching on Blackhole, see #12349"),
        ),
    ],
    ids=[
        "rank_3",
        "rank_4",
        "rank_5",
        "two_indices",
        "three_indices",
        "index_size_100",
        "int32",
        "tile_index_on_h",
        "tile_index_tensor",
    ],
)
def test_moreh_getitem_corner_cases(input_shape, layout, index_dims, index_size, dtype, row_major_index, device):
    torch.manual_seed(0)
    run_moreh_getitem_test(
        input_shape,
        layout,
        device,
        index_dims=index_dims,
        index_size=index_size,
        dtype=dtype,
        row_major_index=row_major_index,
    )


@pytest.mark.merge_gate
def test_moreh_getitem_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_getitem_test([10, 70], ttnn.ROW_MAJOR_LAYOUT, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = ttnn.from_torch(torch.zeros([10, 70]), dtype=ttnn.bfloat16, device=device)
    run_moreh_getitem_test([10, 70], ttnn.ROW_MAJOR_LAYOUT, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
