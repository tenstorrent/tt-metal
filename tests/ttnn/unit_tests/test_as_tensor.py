# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_as_tensor(device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [2**15])
@pytest.mark.parametrize("width", [2**16])
def test_allocate_large_tensor(device, height, width):
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    reshard_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, height, width]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        memory_config,
    )


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
def test_as_tensor_with_device_tilizer(device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config, use_device_tilizer=True
    )

    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_as_tensor_with_cache(tmp_path, device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.float32)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.device() == device
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.device() == device
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)
