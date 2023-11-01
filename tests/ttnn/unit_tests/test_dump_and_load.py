# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import ttnn


@pytest.mark.parametrize("h", [1024])
@pytest.mark.parametrize("w", [1024])
def test_dump_and_load(tmp_path, h, w):
    file_name = tmp_path / pathlib.Path("tensor.bin")

    torch_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor)
    ttnn.dump_tensor(file_name, tt_tensor)

    loaded_tt_tensor = ttnn.load_tensor(file_name)
    loaded_torch_tensor = ttnn.to_torch(loaded_tt_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)


@pytest.mark.parametrize("h", [1024])
@pytest.mark.parametrize("w", [1024])
def test_dump_and_load_tilized(tmp_path, h, w):
    file_name = tmp_path / pathlib.Path("tensor.bin")

    torch_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor)
    tt_tensor = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
    ttnn.dump_tensor(file_name, tt_tensor)

    loaded_tt_tensor = ttnn.load_tensor(file_name)
    loaded_tt_tensor = ttnn.to_layout(loaded_tt_tensor, ttnn.ROW_MAJOR_LAYOUT)
    loaded_torch_tensor = ttnn.to_torch(loaded_tt_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)
