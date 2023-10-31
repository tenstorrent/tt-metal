# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import torch

import tt_lib as ttl

def unpadding_test(input_layout, input_tensor_shape, output_tensor_start,
                   output_tensor_end, device, in_mem_config,
                   out_mem_config):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)


    a = (
        ttl.tensor.Tensor(
            inp.reshape(-1).tolist(),
            inp.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(input_layout)
        .to(device)
    )


    a_pt = ttl.tensor.unpad(a, output_tensor_start, output_tensor_end, output_mem_config=out_mem_config).cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Pytorch reference
    a_ref = inp[
        output_tensor_start[0] : output_tensor_end[0] + 1,
        output_tensor_start[1] : output_tensor_end[1] + 1,
        output_tensor_start[2] : output_tensor_end[2] + 1,
        output_tensor_start[3] : output_tensor_end[3] + 1,
    ]

    return a_pt, a_ref, ttl.program_cache.num_entries()

@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
    ),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
    ),
    ids=["in0_DRAM"],
)

@pytest.mark.parametrize(
    "input_tensor_shape_0, output_tensor_start_0, output_tensor_end_0",
    (
        ((4, 3, 64, 64), (0, 0, 0, 0), (3, 2, 31, 31)),
    ),
)

@pytest.mark.parametrize(
    "input_tensor_shape_1, output_tensor_start_1, output_tensor_end_1",
    (
        ((9, 8, 128, 128), (0, 0, 0, 0), (8, 7, 31, 31)),
    ),
)

def test_run_unpadding_test(input_tensor_shape_0, output_tensor_start_0, output_tensor_end_0,
                            input_tensor_shape_1, output_tensor_start_1, output_tensor_end_1,
                            device, in_mem_config, out_mem_config, use_program_cache):

    a_pt, a_ref, num_cache_entries = unpadding_test(ttl.tensor.Layout.ROW_MAJOR, input_tensor_shape_0, output_tensor_start_0,
                                 output_tensor_end_0, device,
                                 in_mem_config, out_mem_config)
    assert a_pt.shape == a_ref.shape
    assert torch.equal(a_pt, a_ref)
    assert (num_cache_entries == 1)

    a_pt, a_ref, num_cache_entries = unpadding_test(ttl.tensor.Layout.ROW_MAJOR, input_tensor_shape_1, output_tensor_start_1,
                                 output_tensor_end_1, device,
                                 in_mem_config, out_mem_config)
    assert a_pt.shape == a_ref.shape
    assert torch.equal(a_pt, a_ref)
    #different width for row major
    assert (num_cache_entries == 2)


    a_pt, a_ref, num_cache_entries = unpadding_test(ttl.tensor.Layout.TILE, input_tensor_shape_0, output_tensor_start_0,
                                 output_tensor_end_0, device,
                                 in_mem_config, out_mem_config)
    #change from RM to TILE
    assert (num_cache_entries == 3)
    assert a_pt.shape == a_ref.shape
    assert torch.equal(a_pt, a_ref)


    a_pt, a_ref, num_cache_entries = unpadding_test(ttl.tensor.Layout.TILE, input_tensor_shape_1, output_tensor_start_1,
                                 output_tensor_end_1, device,
                                 in_mem_config, out_mem_config)
    #CACHE HIT
    assert (num_cache_entries == 3)
    assert a_pt.shape == a_ref.shape
    assert torch.equal(a_pt, a_ref)
