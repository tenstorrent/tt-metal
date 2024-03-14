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
        (32, 6),
        (16, 3),
    ],
)
@pytest.mark.parametrize(
    "D",
    [4, 16, 1024, 4096],
)
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_indexed_slice(seed, B, b, D, tt_dtype, device):
    torch.manual_seed(seed)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    input_a_shape = (B, 1, 1, D)
    input_b_shape = (b, 1, 1, D)
    torch_batch_ids = torch.randint(0, B - 1, (1, 1, 1, b))
    torch_input_a = torch.rand(input_a_shape, dtype=dtype)
    torch_input_b = torch.rand(input_b_shape, dtype=dtype)
    batch_ids = ttl.tensor.Tensor(torch_batch_ids, ttl.tensor.DataType.UINT32).to(
        device, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    )
    input_a = ttl.tensor.Tensor(torch_input_a, tt_dtype).to(device)
    input_b = ttl.tensor.Tensor(torch_input_b, tt_dtype).to(device)
    output = ttl.tensor.indexed_fill(batch_ids, input_a, input_b)
    torch_input_a[torch_batch_ids[-1]] = torch_input_b
    output_torch = output.cpu().to_torch()

    print(torch_batch_ids)
    assert torch.allclose(torch_input_a, output_torch)
