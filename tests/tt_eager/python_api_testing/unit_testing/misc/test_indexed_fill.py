# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib
import ttnn
import torch
import numpy as np


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
        ttnn.bfloat16,
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
    batch_ids = ttnn.Tensor(torch_batch_ids, ttnn.uint32).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )
    input_a = ttnn.Tensor(torch_input_a, tt_dtype).to(device)
    input_b = ttnn.Tensor(torch_input_b, tt_dtype).to(device)
    output = ttnn.indexed_fill(batch_ids, input_a, input_b)
    torch_input_a[torch_batch_ids[-1]] = torch_input_b
    output_torch = output.cpu().to_torch()

    print(torch_batch_ids)
    assert torch.allclose(torch_input_a, output_torch)
