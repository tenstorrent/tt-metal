import ttnn, torch
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
try:
    WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    for shape in [(1, 1, 32, 64)]:
        cfg = auto_shard_config(list(shape), WIDTH, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        print("shape", shape)
        print("  memory_config:", cfg)
        try:
            grid = cfg.shard_spec.grid
            print("  shard grid:", grid)
        except Exception as e:
            print("  grid err", e)
finally:
    ttnn.close_device(device)
