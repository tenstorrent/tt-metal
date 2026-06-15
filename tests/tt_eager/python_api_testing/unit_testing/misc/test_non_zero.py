# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np
import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "seed",
    [
        0,
        42,
        13,
        9,
    ],
)
@pytest.mark.parametrize(
    "B, b",
    [
        (16, 3),
        (32, 6),
    ],
)
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
    ],
)
def test_indexed_slice(seed, B, b, tt_dtype, device):
    torch.manual_seed(seed)

    if tt_dtype == ttnn.uint32:
        dtype = torch.int32
    else:
        dtype = tt_dtype_to_torch_dtype[tt_dtype]

    zero_tensor = torch.zeros(1, 1, 1, b, dtype=dtype)
    if tt_dtype == ttnn.uint32:
        non_zero_tensor = torch.randint(1, 100, (1, 1, 1, B - b), dtype=dtype)
    else:
        non_zero_tensor = torch.rand((1, 1, 1, B - b), dtype=dtype)
        non_zero_tensor.add(1, alpha=1)
    torch_input_tensor = torch.concat((zero_tensor, non_zero_tensor), dim=3)
    torch_input_tensor = torch_input_tensor[:, :, :, torch.randperm(torch_input_tensor.size(3))]

    golden_output = torch.nonzero(torch_input_tensor, as_tuple=True)[3]
    golden_output = golden_output.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.int32)

    input_tt = ttnn.from_torch(torch_input_tensor, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    num_indices_tt, indices_tt = ttnn.nonzero(input_tt, memory_config=mem_config, queue_id=0)

    num_non_zeros = ttnn.to_torch(ttnn.to_layout(num_indices_tt, ttnn.ROW_MAJOR_LAYOUT))[0, 0, 0, 0].item()
    assert num_non_zeros == B - b

    a_pt = ttnn.to_torch(ttnn.to_layout(indices_tt, ttnn.ROW_MAJOR_LAYOUT))[:, :, :, :num_non_zeros].to(torch.int32)

    assert torch.equal(golden_output, a_pt)
