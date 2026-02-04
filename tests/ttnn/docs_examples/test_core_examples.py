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


def test_copy_device_to_host_tensor(device):
    # Create a TT-NN tensor and copy it to the host
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(ttnn_tensor.spec, device)
    ttnn.copy_device_to_host_tensor(ttnn_tensor, host_tensor)

    logger.info("Host tensor shape after copying from device", host_tensor.shape)


def test_copy_host_to_device_tensor(device):
    # Create a host tensor and copy it to a pre-allocated device tensor
    dtype = ttnn.bfloat16
    layout = ttnn.ROW_MAJOR_LAYOUT

    tensor = ttnn.rand((10, 64, 32), device=device, dtype=dtype, layout=layout)
    host_tensor = ttnn.from_device(tensor)
    device_tensor_copy = ttnn.allocate_tensor_on_device(host_tensor.spec, device)
    ttnn.copy_host_to_device_tensor(host_tensor, device_tensor_copy)

    logger.info("TT-NN tensor shape after copying to device", device_tensor_copy.shape)


def test_to_dtype(device):
    # Create a TT-NN tensor on the host and convert it to a different data type
    tensor = ttnn.rand((10, 64, 32), device=device, dtype=ttnn.bfloat16)
    tensor = ttnn.from_device(tensor)  # to_dtype requires a host tensor
    tensor = ttnn.to_dtype(tensor, dtype=ttnn.uint8)
    assert tensor.dtype == ttnn.uint8
    assert tensor.shape == (10, 64, 32)

    logger.info("TT-NN tensor shape after converting to uint8", tensor.shape)


def test_typecast(device):
    # Create a TT-NN tensor (on host or device) and typecast it to a different data type
    tensor = ttnn.typecast(ttnn.rand((10, 3, 32, 32), dtype=ttnn.bfloat16, device=device), dtype=ttnn.uint16)
    assert tensor.dtype == ttnn.uint16
    assert tensor.shape == (10, 3, 32, 32)

    logger.info("TT-NN tensor shape after typecasting to uint16", tensor.shape)
