import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

dev = ttnn.open_device(device_id=0)
try:
    T, C = 64, 128

    def d(t, mc=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    sh = ttnn.create_sharded_memory_config((T, C), ttnn.CoreGrid(x=1, y=1), ttnn.ShardStrategy.HEIGHT)
    try:
        mhc_post(d(torch.randn(T, C), sh), d(torch.randn(T, 4 * C)), d(torch.rand(T, 4)), d(torch.rand(T, 16)))
        print("PROBE: NOT refused")
    except ValueError as e:
        print("PROBE: refused:", e)
finally:
    ttnn.close_device(dev)
