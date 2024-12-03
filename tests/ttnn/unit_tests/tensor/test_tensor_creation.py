# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
def test_tensor_creation(shape, tt_dtype, device):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.uint8, torch.int16, torch.int32}:
        py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        py_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(py_tensor, tt_dtype, device)

    tt_tensor = tt_tensor.cpu()

    py_tensor_after_round_trip = tt_tensor.to_torch()

    assert py_tensor.dtype == py_tensor_after_round_trip.dtype
    assert py_tensor.shape == py_tensor_after_round_trip.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(py_tensor, py_tensor_after_round_trip, **allclose_kwargs)
    assert passing
