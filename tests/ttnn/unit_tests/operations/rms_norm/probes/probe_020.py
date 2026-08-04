import torch, ttnn, math
from eval.sharding import auto_shard_config, shard_config

dev = device
g = dev.compute_with_storage_grid_size()
print("GRID", g.x, g.y, "=", g.x * g.y)
print("L1 unreserved", ttnn.get_max_worker_l1_unreserved_size())
print("l1 align", ttnn._ttnn.device.get_l1_alignment())

cases = [
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 2048), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 40), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 40), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 4064), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 224, 72), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 224, 72), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 3232, 96), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 17, 50), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
]
for shape, ml, lay in cases:
    mc = auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=dev)
    ss = mc.shard_spec
    print(
        f"{shape} {str(ml).split('.')[-1]:16s} {str(lay).split('.')[-1]:18s} shard={list(ss.shape)} grid={ss.grid} nc={ss.grid.num_cores()} orient={ss.orientation}"
    )
