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
    ttnn.bfloat4_b: torch.float,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint32,
        ttnn.uint16,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
def test_tensor_conversion_between_torch_and_tt(shape, tt_dtype, device):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype)
    if tt_dtype in {
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.uint32,
        ttnn.uint16,
    }:
        assert tt_tensor.storage_type() == ttnn.StorageType.BORROWED
    else:
        assert tt_tensor.storage_type() == ttnn.StorageType.OWNED

    if tt_dtype in {ttnn.bfloat8_b, ttnn.bfloat4_b}:
        tt_tensor = tt_tensor.to(ttnn.TILE_LAYOUT)

    if tt_dtype in {
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.uint32,
        ttnn.uint16,
    }:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    if tt_dtype in {
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    }:
        tt_tensor = tt_tensor.to(ttnn.ROW_MAJOR_LAYOUT)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip, **allclose_kwargs)
    assert passing


tt_dtype_to_np_dtype = {
    ttnn.uint16: np.int16,
    ttnn.uint32: np.int32,
    ttnn.float32: np.float32,
    ttnn.bfloat16: np.float32,
    ttnn.bfloat8_b: np.float32,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint32,
        ttnn.uint16,
        ttnn.float32,
        # ttnn.bfloat16,
    ],
)
def test_tensor_conversion_between_torch_and_np(shape, tt_dtype, device):
    dtype = tt_dtype_to_np_dtype[tt_dtype]

    if dtype in {np.int16, np.int32}:
        np_tensor = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype=dtype)
    else:
        np_tensor = np.random.random(shape).astype(dtype=dtype)

    tt_tensor = ttnn.Tensor(np_tensor, tt_dtype)
    if tt_dtype in {ttnn.float32, ttnn.uint32, ttnn.uint16}:
        assert tt_tensor.storage_type() == ttnn.StorageType.BORROWED

    if tt_dtype in {
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.uint32,
        ttnn.uint16,
    }:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    np_tensor_after_round_trip = tt_tensor.to_numpy()

    assert np_tensor.dtype == np_tensor_after_round_trip.dtype
    assert np_tensor.shape == np_tensor_after_round_trip.shape

    passing = np.allclose(np_tensor, np_tensor_after_round_trip)
    assert passing


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
