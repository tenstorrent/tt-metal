# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull


def slice_test(
    input_layout,
    input_tensor_shape,
    output_tensor_start,
    output_tensor_end,
    device,
    in_mem_config,
    out_mem_config,
    dtype,
):
    if dtype == ttnn.float32:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.float)
    else:
        torch_input_tensor = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_layout, device=device, memory_config=in_mem_config
    )

    tt_output_tensor = ttnn.slice(tt_input_tensor, output_tensor_start, output_tensor_end, memory_config=out_mem_config)

    a_pt = ttnn.to_torch(tt_output_tensor)

    # Pytorch reference
    a_ref = torch_input_tensor[
        output_tensor_start[0] : output_tensor_end[0] + 1,
        output_tensor_start[1] : output_tensor_end[1] + 1,
        output_tensor_start[2] : output_tensor_end[2] + 1,
        output_tensor_start[3] : output_tensor_end[3] + 1,
    ]

    return a_pt, a_ref, device.num_program_cache_entries()


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "input_tensor_shape_0, output_tensor_start_0, output_tensor_end_0",
    (
        ((4, 3, 64, 64), (0, 0, 0, 0), (3, 2, 31, 31)),
        ((1, 1, 64, 64), (0, 0, 0, 0), (0, 0, 31, 63)),
        ((1, 1, 128, 96), (0, 0, 64, 32), (0, 0, 95, 95)),
        ((1, 1, 128, 96), (0, 0, 64, 32), (0, 0, 95, 95)),
        ((1, 3, 32, 32), (0, 1, 0, 0), (0, 2, 31, 31)),
        ((1, 6, 32, 32), (0, 2, 0, 0), (0, 4, 31, 31)),
        ((1, 6, 128, 64), (0, 2, 64, 32), (0, 4, 95, 63)),
        ((4, 6, 128, 64), (1, 2, 64, 32), (2, 4, 95, 63)),
    ),
)
@pytest.mark.parametrize(
    "input_tensor_shape_1, output_tensor_start_1, output_tensor_end_1",
    (((9, 8, 128, 128), (0, 0, 0, 0), (8, 7, 31, 31)),),
)
def test_run_slice_test(
    input_tensor_shape_0,
    output_tensor_start_0,
    output_tensor_end_0,
    input_tensor_shape_1,
    output_tensor_start_1,
    output_tensor_end_1,
    device,
    in_mem_config,
    out_mem_config,
    dtype,
    use_program_cache,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor_shape_0,
        output_tensor_start_0,
        output_tensor_end_0,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
    )
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq
    assert num_cache_entries == 1

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor_shape_1,
        output_tensor_start_1,
        output_tensor_end_1,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
    )
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq
    # different width for row major
    assert num_cache_entries == 2

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.TILE_LAYOUT,
        input_tensor_shape_0,
        output_tensor_start_0,
        output_tensor_end_0,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
    )
    # change from RM to TILE
    assert num_cache_entries == 3
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq

    a_pt, a_ref, num_cache_entries = slice_test(
        ttnn.TILE_LAYOUT,
        input_tensor_shape_1,
        output_tensor_start_1,
        output_tensor_end_1,
        device,
        in_mem_config,
        out_mem_config,
        dtype,
    )
    # CACHE HIT
    assert num_cache_entries == 3
    assert a_pt.shape == a_ref.shape
    eq = torch.equal(a_pt, a_ref)
    assert eq
