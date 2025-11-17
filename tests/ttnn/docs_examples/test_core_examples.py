# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_as_tensor(device):
    # Create a Torch tensor and convert it to a TT-NN tensor
    tensor = ttnn.as_tensor(torch.randn((2, 3)), dtype=ttnn.bfloat16)
    logger.info(tensor.shape)


def test_from_torch(device):
    # Create a Torch tensor and convert it to a TT-NN tensor
    torch_tensor = torch.randn((4, 5), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, device=device)

    logger.info("TT-NN tensor shape", ttnn_tensor.shape)


def test_to_torch(device):
    # Create a TT-NN tensor and convert it to a Torch tensor
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    torch_tensor = ttnn.to_torch(ttnn_tensor)

    logger.info("Torch tensor shape", torch_tensor.shape)


def test_to_device(device):
    # Open the device
    # device_id = 0
    # device = ttnn.open_device(device_id=device_id)

    # Create a TT-NN tensor and move it to the specified device
    tensor_on_host = ttnn.from_torch(torch.randn((10, 64, 32)), dtype=ttnn.bfloat16)
    ttnn_tensor = ttnn.to_device(tensor_on_host, device=device)

    logger.info("TT-NN tensor shape after moving to device", ttnn_tensor.shape)


def test_from_device(device):
    # Open the device
    # device_id = 0
    # device = ttnn.open_device(device_id=device_id)

    # Create a TT-NN tensor on the device and move it back to the host
    ttnn_tensor_on_device = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_on_host = ttnn.from_device(ttnn_tensor_on_device)

    logger.info("TT-NN tensor shape after moving from device to host", ttnn_tensor_on_host.shape)


def test_to_layout(device):
    # Create a TT-NN tensor and change its layout
    ttnn_tensor = ttnn.rand((2, 3, 4), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_layout_changed = ttnn.to_layout(ttnn_tensor, layout=ttnn.TILE_LAYOUT)

    logger.info("TT-NN tensor shape after changing layout", ttnn_tensor_layout_changed.shape)


def test_dump_tensor(device):
    # Create a TT-NN tensor and dump its contents
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn.dump_tensor(file_name="ttnn_tensor.tensorbin", tensor=ttnn_tensor)


def test_load_tensor(device):
    # Create a TT-NN tensor, dump its contents, and load it back
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn.dump_tensor(file_name="ttnn_tensor.tensorbin", tensor=ttnn_tensor)

    loaded_tensor = ttnn.load_tensor(file_name="ttnn_tensor.tensorbin", device=device)
    logger.info("Loaded TT-NN tensor shape", loaded_tensor.shape)


def test_deallocate(device):
    # Create a TT-NN tensor and deallocate its memory
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn.deallocate(ttnn_tensor)


def test_reallocate(device):
    # Create a TT-NN tensor, deallocate it, and then reallocate it
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_reallocated = ttnn.reallocate(ttnn_tensor)

    logger.info("Reallocated TT-NN tensor shape", ttnn_tensor_reallocated.shape)


def test_to_memory_config(device):
    # Create a TT-NN tensor and change its memory configuration
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_memory_config_changed = ttnn.to_memory_config(ttnn_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

    logger.info("TT-NN tensor shape after changing memory configuration", ttnn_tensor_memory_config_changed.shape)
