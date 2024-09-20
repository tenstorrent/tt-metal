# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np
import ttnn


tt_dtype_to_torch_dtype = {
    ttnn.uint16: torch.int16,
    ttnn.uint32: torch.int32,
    ttnn.float32: torch.float,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float,
}


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

    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    zero_tensor = torch.zeros(1, 1, 1, b, dtype=dtype)
    if tt_dtype == ttnn.uint32:
        non_zero_tensor = torch.randint(1, 100, (1, 1, 1, B - b), dtype=dtype)
    else:
        non_zero_tensor = torch.rand((1, 1, 1, B - b), dtype=dtype)
        non_zero_tensor.add(1, alpha=1)
    torch_input_tensor = torch.concat((zero_tensor, non_zero_tensor), dim=3)
    torch_input_tensor = torch_input_tensor[torch.randperm(torch_input_tensor.size()[2])]

    input_tt = ttnn.Tensor(torch_input_tensor, tt_dtype).to(device)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    num_indices_tt, indices_tt = ttnn.nonzero(input_tt, memory_config=mem_config, queue_id=0)
    torch_num_indices = num_indices_tt.cpu().to_torch()
    torch_indices = indices_tt.cpu().to_torch()

    num_non_zeros = torch_num_indices[0, 0, 0, 0].item()
    assert num_non_zeros == B - b

    a_pt = (
        ttnn.slice(indices_tt, (0, 0, 0, 0), (1, 1, 1, num_non_zeros), memory_config=mem_config)
        .cpu()
        .to(ttnn.ROW_MAJOR_LAYOUT)
        .to_torch()
    )

    golden_output = torch.arange(start=b, end=B, step=1, dtype=torch.int32)
    golden_output = torch.reshape(golden_output, (1, 1, 1, B - b))
    assert torch.allclose(golden_output, a_pt)
