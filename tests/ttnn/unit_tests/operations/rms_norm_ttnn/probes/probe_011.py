import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as pd
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
shape = (1, 1, 32, 8192)
for dt, td in ((ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)):
    mc = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.BLOCK_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device
    )
    print("dt", dt, "shard", mc.shard_spec.shape, "grid", mc.shard_spec.grid)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=td), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
    )
    plan = pd._plan_placement(device, x, x, is_tile=False, Rt=1, Wt=256, W=8192, R_rm=32, partial_w=0)
    print("  band", plan.band, "wt_per_core", plan.wt_per_core, "group", plan.group_size, "l1res", plan.l1_reserved)
    print("  avail", ttnn.get_max_worker_l1_unreserved_size(), "tile", ttnn.tile_size(dt))
ttnn.close_device(device)
