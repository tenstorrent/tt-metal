import torch, ttnn
from eval.memory import memory_axes
from ttnn.operations.reshape import reshape
device = ttnn.open_device(device_id=0)
try:
    t = torch.randn((1, 1, 256, 64), dtype=torch.bfloat16)
    mc = ttnn.create_sharded_memory_config((1,1,256,64), core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=False)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    print("AXES", memory_axes(x))
    try:
        reshape(x, (1, 4, 64, 64))
        print("NOT REFUSED")
    except NotImplementedError as e:
        print("REFUSED", type(e).__name__, e)
    out = reshape(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG), (1, 4, 64, 64))
    print("OUT MC", out.memory_config())
    out = reshape(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG), (1, 4, 64, 64))
    print("OUT MC", out.memory_config())
finally:
    ttnn.close_device(device)
