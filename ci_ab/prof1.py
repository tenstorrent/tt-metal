# One case of ttnn.experimental.dropout under the device profiler: 2 warm-up calls, then 6 measured.
import sys

import torch
import ttnn

SHAPES = {"dropout_32x32": (1, 1, 32, 32), "dropout_1024x1024": (1, 1, 1024, 1024), "dropout_4096x2048": (1, 1, 4096, 2048)}
case = sys.argv[1]
dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(SHAPES[case], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(8):
        ttnn.experimental.dropout(x, probability=0.5, scale=2.0, seed=1234)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    print(f"CASE {case} ok", flush=True)
finally:
    ttnn.close_device(dev)
