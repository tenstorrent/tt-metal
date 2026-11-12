import torch, ttnn, gc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def l1():
    v = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    return v.largest_contiguous_bytes_free_per_bank


for shape, layout in [
    ((1, 1, 32, 64), ttnn.TILE_LAYOUT),
    ((1, 1, 2048, 256), ttnn.ROW_MAJOR_LAYOUT),
    ((128, 8192), ttnn.TILE_LAYOUT),
    ((1, 1, 32, 64), ttnn.TILE_LAYOUT),
]:
    mc = auto_shard_config(shape, HS, layout=layout, dtype=ttnn.bfloat16, device=device)
    print("L1 before", shape, l1())
    x = ttnn.from_torch(
        torch.randn(shape).bfloat16(), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc
    )
    print("L1 after input", l1())
    o = rms_norm(x)
    print("L1 after op", l1())
    del x, o
    gc.collect()
    print("L1 after del", l1())
ttnn.close_device(device)
