import time
import pytest
import torch
import ttnn

from models.common.utility_functions import (
    divup,
)
from loguru import logger

from models.experimental.functional_unet.tests.common import is_n300_with_eth_dispatch_cores


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 512 + 64),
        (1024, 512 + 64),
        (8192, 512 + 64),
        (16384, 512 + 64),
    ],
    ids=["128x576", "1024x576", "8192x576", "16384x576"],
)
def test_tensor_read(device, H, W):
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    assert num_cores == 64

    torch_input_tensor = torch.randn(H, W)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

    tensor_on_device = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outputs = []

    num_iters = 100
    start = time.time()
    for i in range(num_iters):
        outputs.append(ttnn.from_device(tensor_on_device, blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()

    total_time = end - start
    total_size_in_bytes = 2 * H * W * num_iters  # 2 in for bfloat16
    total_size_in_GiB = total_size_in_bytes / (1024**3)
    time_per_tensor = total_time / num_iters
    num_devices = device.get_num_devices()
    logger.warning(
        f"[{H}x{W}] Devices={num_devices}; Payload={H*W*2/1024}KiB; Avg read= {time_per_tensor*1000:.1f}ms; bandwidth = {total_size_in_GiB / total_time:.3f} GiB/s;"
    )


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 512 + 64),
        (1024, 512 + 64),
        (8192, 512 + 64),
        (16384, 512 + 64),
    ],
    ids=["128x576", "1024x576", "8192x576", "16384x576"],
)
def test_tensor_write(device, H, W):
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    assert num_cores == 64

    torch_input_tensor = torch.randn(H, W)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

    tensor_on_device = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    num_iters = 1000
    start = time.time()
    for i in range(num_iters):
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor, tensor_on_device)
    ttnn.synchronize_device(device)
    end = time.time()

    total_time = end - start
    total_size_in_bytes = 2 * H * W * num_iters  # 2 in for bfloat16
    total_size_in_GiB = total_size_in_bytes / (1024**3)
    time_per_tensor = total_time / num_iters
    num_devices = device.get_num_devices()
    logger.warning(
        f"[{H}x{W}] Devices={num_devices}; Payload={H*W*2/1024}KiB; Avg write= {time_per_tensor*1000:.1f}ms; bandwidth = {total_size_in_GiB / total_time:.3f} GiB/s;"
    )


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 512 + 64),
        (1024, 512 + 64),
        (8192, 512 + 64),
        (16384, 512 + 64),
    ],
    ids=["128x576", "1024x576", "8192x576", "16384x576"],
)
def test_mesh_tensor_read(mesh_device, H, W):
    if not is_n300_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300")

    # Create mesh mapper for sharding input tensor across 2 chips
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input_tensor = torch.randn(H, W)

    # Create sharded tensor across 2 devices
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)

    tensor_on_device = ttnn.to_device(ttnn_input_tensor, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outputs = []

    num_iters = 100
    start = time.time()
    for i in range(num_iters):
        outputs.append(ttnn.from_device(tensor_on_device, blocking=False))
    ttnn.synchronize_device(mesh_device)
    end = time.time()

    total_time = end - start
    total_size_in_bytes = 2 * H * W * num_iters  # 2 bytes for bfloat16
    total_size_in_GiB = total_size_in_bytes / (1024**3)
    time_per_tensor = total_time / num_iters
    num_devices = mesh_device.get_num_devices()
    logger.warning(
        f"[{H}x{W}] Devices={num_devices}; Payload={H*W*2/1024}KiB; Avg read= {time_per_tensor*1000:.1f}ms; bandwidth = {total_size_in_GiB / total_time:.3f} GiB/s;"
    )


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 512 + 64),
        (1024, 512 + 64),
        (8192, 512 + 64),
        (16384, 512 + 64),
    ],
    ids=["128x576", "1024x576", "8192x576", "16384x576"],
)
def test_mesh_tensor_write(mesh_device, H, W):
    if not is_n300_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300")

    # Create mesh mapper for sharding input tensor across 2 chips
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    torch_input_tensor = torch.randn(H, W)

    # Create sharded tensor across 2 devices
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)

    tensor_on_device = ttnn.to_device(ttnn_input_tensor, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    num_iters = 1000
    start = time.time()
    for i in range(num_iters):
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor, tensor_on_device)
    ttnn.synchronize_device(mesh_device)
    end = time.time()

    total_time = end - start
    total_size_in_bytes = 2 * H * W * num_iters  # 2 bytes for bfloat16
    total_size_in_GiB = total_size_in_bytes / (1024**3)
    time_per_tensor = total_time / num_iters
    num_devices = mesh_device.get_num_devices()
    logger.warning(
        f"[{H}x{W}] Devices={num_devices}; Payload={H*W*2/1024}KiB; Avg write= {time_per_tensor*1000:.1f}ms; bandwidth = {total_size_in_GiB / total_time:.3f} GiB/s;"
    )
