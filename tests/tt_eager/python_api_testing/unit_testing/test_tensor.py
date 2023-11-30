# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import tt_lib as ttl


tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT16: torch.int16,
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.FLOAT32: torch.float,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
    ttl.tensor.DataType.BFLOAT8_B: torch.float,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_tensor_conversion_between_torch_and_tt(shape, tt_dtype, device):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)
    if tt_dtype in {
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
    }:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
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

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing


@pytest.mark.parametrize("shape", [(1, 2, 3, 4)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_print(shape, tt_dtype):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)
    if tt_dtype == ttl.tensor.DataType.UINT16:
        assert (
            str(tt_tensor)
            == "Tensor([ [[[684, 559, 629, 192],\n    [835, 763, 707, 359],\n    [9, 723, 277, 754]],\n\n   [[804, 599, 70, 472],\n    [600, 396, 314, 705],\n    [486, 551, 87, 174]]]], dtype=uint16 )\n"
        )
    elif tt_dtype == ttl.tensor.DataType.UINT32:
        assert (
            str(tt_tensor)
            == "Tensor([ [[[684, 559, 629, 192],\n    [835, 763, 707, 359],\n    [9, 723, 277, 754]],\n\n   [[804, 599, 70, 472],\n    [600, 396, 314, 705],\n    [486, 551, 87, 174]]]], dtype=uint32 )\n"
        )
    elif tt_dtype == ttl.tensor.DataType.FLOAT32:
        assert (
            str(tt_tensor)
            == "Tensor([ [[[0.496257, 0.768222, 0.0884774, 0.13203],\n    [0.307423, 0.634079, 0.490093, 0.896445],\n    [0.455628, 0.632306, 0.348893, 0.401717]],\n\n   [[0.0223258, 0.168859, 0.293888, 0.518522],\n    [0.697668, 0.800011, 0.161029, 0.282269],\n    [0.681609, 0.915194, 0.3971, 0.874156]]]], dtype=float32 )\n"
        )
    elif tt_dtype == ttl.tensor.DataType.BFLOAT16:
        assert (
            str(tt_tensor)
            == "Tensor([ [[[0.671875, 0.183594, 0.457031, 0.75],\n    [0.261719, 0.980469, 0.761719, 0.402344],\n    [0.0351562, 0.824219, 0.0820312, 0.945312]],\n\n   [[0.140625, 0.339844, 0.273438, 0.84375],\n    [0.34375, 0.546875, 0.226562, 0.753906],\n    [0.898438, 0.152344, 0.339844, 0.679688]]]], dtype=bfloat16 )\n"
        )
    else:
        raise ValueError(f"Unsupported dtype: {tt_dtype}")
