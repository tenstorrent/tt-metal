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
    ttl.tensor.DataType.BFLOAT4_B: torch.float,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.BFLOAT4_B,
    ],
)
def test_tensor_conversion_between_torch_and_tt(shape, tt_dtype, device):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)
    if tt_dtype in {
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
    }:
        assert tt_tensor.storage_type() == ttl.tensor.StorageType.BORROWED
    else:
        assert tt_tensor.storage_type() == ttl.tensor.StorageType.OWNED

    if tt_dtype in {ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT4_B}:
        tt_tensor = tt_tensor.to(ttl.tensor.Layout.TILE)

    if tt_dtype in {
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.BFLOAT4_B,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
    }:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    if tt_dtype in {
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.BFLOAT4_B,
    }:
        tt_tensor = tt_tensor.to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    allclose_kwargs = {}
    if tt_dtype == ttl.tensor.DataType.BFLOAT8_B:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttl.tensor.DataType.BFLOAT4_B:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip, **allclose_kwargs)
    assert passing


tt_dtype_to_np_dtype = {
    ttl.tensor.DataType.UINT16: np.int16,
    ttl.tensor.DataType.UINT32: np.int32,
    ttl.tensor.DataType.FLOAT32: np.float32,
    ttl.tensor.DataType.BFLOAT16: np.float32,
    ttl.tensor.DataType.BFLOAT8_B: np.float32,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.FLOAT32,
        # ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_tensor_conversion_between_torch_and_np(shape, tt_dtype, device):
    dtype = tt_dtype_to_np_dtype[tt_dtype]

    if dtype in {np.int16, np.int32}:
        np_tensor = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype=dtype)
    else:
        np_tensor = np.random.random(shape).astype(dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(np_tensor, tt_dtype)
    if tt_dtype in {ttl.tensor.DataType.FLOAT32, ttl.tensor.DataType.UINT32, ttl.tensor.DataType.UINT16}:
        assert tt_tensor.storage_type() == ttl.tensor.StorageType.BORROWED

    if tt_dtype in {
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
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
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.BFLOAT4_B,
    ],
)
def test_serialization(tmp_path, shape, tt_dtype):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)

    file_name = tmp_path / pathlib.Path("tensor.bin")
    ttl.tensor.dump_tensor(str(file_name), tt_tensor)
    torch_tensor_from_file = ttl.tensor.load_tensor(str(file_name)).to_torch()

    assert torch_tensor.dtype == torch_tensor_from_file.dtype
    assert torch_tensor.shape == torch_tensor_from_file.shape

    allclose_kwargs = {}
    if tt_dtype == ttl.tensor.DataType.BFLOAT8_B:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttl.tensor.DataType.BFLOAT4_B:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing
