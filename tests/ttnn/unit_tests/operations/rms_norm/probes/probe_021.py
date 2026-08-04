import torch, ttnn, math
from eval.sharding import auto_shard_config, shard_config

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print("GRID", g.x, g.y, "=", g.x * g.y)
print("L1 unreserved", ttnn.get_max_worker_l1_unreserved_size())
print("l1 align", ttnn._ttnn.device.get_l1_alignment())

ML = ttnn.TensorMemoryLayout
cases = [
    ((1, 1, 256, 512), ML.HEIGHT_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 2048), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 256, 512), ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 40), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 40), ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 4064), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 7168), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 224, 72), ML.HEIGHT_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 224, 72), ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 3232, 96), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 256, 512), ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 256, 512), ML.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 17, 50), ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 4095), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
]
for shape, ml, lay in cases:
    try:
        mc = auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=dev)
    except Exception as e:
        print(shape, ml, lay, "ERR", e)
        continue
    ss = mc.shard_spec
    cores = list(ttnn.corerange_to_cores(ss.grid, None, True))
    print(
        f"{shape} {str(ml).split('.')[-1]:16s} {str(lay).split('.')[-1]:18s} shard={list(ss.shape)} nc={ss.grid.num_cores()} bbox={ss.grid.bounding_box()} orient={ss.orientation} first={[(c.x,c.y) for c in cores[:4]]} last={[(c.x,c.y) for c in cores[-2:]]}"
    )

# BLOCK orientation check: which axis maps to x?
t = torch.arange(256 * 512, dtype=torch.float32).reshape(1, 1, 256, 512).to(torch.bfloat16)
mc = auto_shard_config([1, 1, 256, 512], ML.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
print("block mc", mc)
ttnn.close_device(dev)
