# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import ttnn


@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_dump_and_load(tmp_path, height, width, layout):
    file_name = tmp_path / pathlib.Path("tensor.bin")

    torch_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor, layout=layout)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name)
    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)


@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_dump_from_device_and_load_to_device(tmp_path, device, height, width, layout, memory_config):
    file_name = tmp_path / pathlib.Path("tensor.bin")

    torch_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, memory_config=memory_config)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name, device=device)
    if memory_config is not None:
        assert ttnn.get_memory_config(loaded_tensor) == memory_config
    else:
        assert ttnn.get_memory_config(loaded_tensor) == ttnn.DRAM_MEMORY_CONFIG

    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)
