import ttnn
from eval.sharding import auto_shard_config

WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED

device = ttnn.open_device(device_id=0)
try:
    shapes = [
        ((1, 1, 32, 64), WIDTH, ttnn.bfloat16),
        ((1, 1, 32, 64), BLOCK, ttnn.bfloat16),
        ((1, 1, 32, 50), WIDTH, ttnn.bfloat16),  # w_non
        ((1, 1, 32, 50), WIDTH, ttnn.float32),  # w_non fp32 (ragged)
        ((2, 4, 128, 512), WIDTH, ttnn.bfloat16),
        ((1, 1, 256, 512), BLOCK, ttnn.bfloat16),
        ((1, 1, 64, 17), BLOCK, ttnn.bfloat16),  # w_non block
        ((1, 1, 32, 4096), WIDTH, ttnn.bfloat16),  # tile-aligned Ws
    ]
    for shp, ml, dt in shapes:
        mc = auto_shard_config(list(shp), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device)
        ss = mc.shard_spec
        grid = ss.grid
        bb = grid.bounding_box()
        nx = int(bb.end.x) - int(bb.start.x) + 1
        ny = int(bb.end.y) - int(bb.start.y) + 1
        Hs, Ws = int(ss.shape[0]), int(ss.shape[1])
        print(
            f"{shp} {str(ml).split('.')[-1]:15s} {str(dt).split('.')[-1]:9s} -> Hs={Hs} Ws={Ws} ncores={grid.num_cores()} bbox={nx}x{ny} ragged={grid.num_cores()!=nx*ny} Ws%32={Ws%32}"
        )
finally:
    ttnn.close_device(device)
