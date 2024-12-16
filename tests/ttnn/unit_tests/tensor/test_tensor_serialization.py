# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
def test_serialization(tmp_path, shape, tt_dtype):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype)

    file_name = tmp_path / pathlib.Path("tensor.bin")
    ttnn.dump_tensor(str(file_name), tt_tensor)
    torch_tensor_from_file = ttnn.load_tensor(str(file_name)).to_torch()

    assert torch_tensor.dtype == torch_tensor_from_file.dtype
    assert torch_tensor.shape == torch_tensor_from_file.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing
