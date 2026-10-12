# Scratch: DRAM->DRAM copy floor for a Q2-sized tensor (~11 MB in, ~11 MB out per chip).
import pytest
import torch

import ttnn


@pytest.mark.parametrize("rows, width", [(1536, 3584), (640, 6144)], ids=["k3q2", "glmq1"])
def test_dram_copy_floor(device, rows, width):
    # K3 Q2 input per chip ~11 MB bf16 (1536 x 3584); GLM Q1 input per chip 640 x 6144 bf16 (7.9 MB)
    x = torch.randn(1, 1, rows, width, dtype=torch.bfloat16)
    for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for _ in range(5):
            y = ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
        ttnn.synchronize_device(device)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(5):
        y = ttnn.untilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True)
        ttnn.deallocate(y)
    for _ in range(5):
        y = ttnn.typecast(t, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
