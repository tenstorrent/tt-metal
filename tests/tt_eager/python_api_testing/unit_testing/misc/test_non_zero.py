# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import tt_lib as ttl


tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT16: torch.int16,
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.FLOAT32: torch.float,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
    ttl.tensor.DataType.BFLOAT8_B: torch.float,
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
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.UINT32,
    ],
)
def test_indexed_slice(seed, B, b, tt_dtype, device):
    torch.manual_seed(seed)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    zero_tensor = torch.zeros(1, 1, 1, b, dtype=dtype)
    if tt_dtype == ttl.tensor.DataType.UINT32:
        non_zero_tensor = torch.randint(1, 100, (1, 1, 1, B - b), dtype=dtype)
    else:
        non_zero_tensor = torch.rand((1, 1, 1, B - b), dtype=dtype)
        non_zero_tensor.add(1, alpha=1)
    torch_input_tensor = torch.concat((zero_tensor, non_zero_tensor), dim=3)
    torch_input_tensor = torch_input_tensor[torch.randperm(torch_input_tensor.size()[2])]

    input_tt = ttl.tensor.Tensor(torch_input_tensor, tt_dtype).to(device)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    num_indices_tt, indices_tt = ttl.tensor.nonzero(input_tt, mem_config)
    torch_num_indices = num_indices_tt.cpu().to_torch()
    torch_indices = indices_tt.cpu().to_torch()

    num_non_zeros = torch_num_indices[0, 0, 0, 0].item()
    assert num_non_zeros == B - b

    a_pt = (
        ttl.tensor.unpad(indices_tt, (0, 0, 0, 0), (0, 0, 0, num_non_zeros - 1), output_mem_config=mem_config)
        .cpu()
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .to_torch()
    )

    golden_output = torch.arange(start=b, end=B, step=1, dtype=torch.int32)
    golden_output = torch.reshape(golden_output, (1, 1, 1, B - b))
    assert torch.allclose(golden_output, a_pt)
