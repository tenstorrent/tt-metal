import time
import pytest
import torch
import ttnn

from models.common.utility_functions import (
    divup,
)
from loguru import logger


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 512 + 64),
        (1024, 512 + 64),
        (2048, 512 + 64),
        (4096, 512 + 64),
        (8192, 512 + 64),
    ],
    ids=["128x576", "1024x576", "2048x576", "4096x576", "8192x576"],
)
def test_tensor_read(device, H, W):
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    assert num_cores == 64

    torch_input_tensor = torch.randn(1, 1, H, W)
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
    logger.warning(
        f"[{H}x{W}] Payload={H*W*2/1024}KiB; Avg read= {time_per_tensor*1000:.1f}ms; bandwidth = {total_size_in_GiB / total_time:.3f} GiB/s;"
    )


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 512 + 64),
        (1024, 512 + 64),
        (2048, 512 + 64),
        (4096, 512 + 64),
        (8192, 512 + 64),
    ],
    ids=["128x576", "1024x576", "2048x576", "4096x576", "8192x576"],
)
def test_tensor_write(device, H, W):
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    assert num_cores == 64

    torch_input_tensor = torch.randn(1, 1, H, W)
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
    logger.warning(
        f"[{H}x{W}] Payload={H*W*2/1024}KiB; Avg write= {time_per_tensor*1000:.1f}ms; bandwidth = {total_size_in_GiB / total_time:.3f} GiB/s;"
    )
