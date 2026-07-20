import torch, ttnn
from ttnn.operations.tilize import tilize as gen_tilize

prod_tilize = ttnn.tilize

SHAPES = [[1, 1, 32, 32], [3, 1, 320, 384], [1, 1, 128, 7328], [1, 1, 2048, 2048]]
K = 16

dev = ttnn.open_device(device_id=0)
try:
    inputs = []
    for shape in SHAPES:
        t = torch.randn(shape, dtype=torch.float32)
        tin = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        inputs.append(tin)
    ttnn.synchronize_device(dev)

    # PROD first (all shapes), then GEN (all shapes) -> attribute by OP CODE + order
    for tin in inputs:
        for _ in range(K):
            o = prod_tilize(tin, use_multicore=True)
    ttnn.synchronize_device(dev)
    for tin in inputs:
        for _ in range(K):
            o = gen_tilize(tin, use_multicore=True)
    ttnn.synchronize_device(dev)
finally:
    ttnn.close_device(dev)
