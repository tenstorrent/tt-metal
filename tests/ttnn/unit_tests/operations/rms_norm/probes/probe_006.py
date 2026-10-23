import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
for shape, ml, (ss, grid) in [
    ((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10))),
    ((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10))),
    ((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    ((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),
    ((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
]:
    mc = shard_config(ss, grid, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(
        torch.zeros(shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    to = ttnn.allocate_tensor_on_device(tx.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    tg = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1]).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    geo = pd._geometry(tx, to, tg, False)
    sp = pd._shard_placement(tx, geo)
    budget = pd._l1_budget_bytes(device)
    g2, plan = pd._select_sharded_blocking(geo, sp, budget)
    print("geom", shape, ss, "budget", budget)
    print("plan", plan)
    print(
        "fp",
        pd._footprint(
            g2,
            plan.block_rows,
            plan.core_w_tiles,
            plan.chunk_w_tiles,
            plan.chunks_per_row,
            plan.xg_lookahead_chunks,
            plan.w_group_size,
        ),
        pd._footprint(g2, sp.rows_hi, plan.core_w_tiles, plan.core_w_tiles, 1, 1, plan.w_group_size),
        pd._footprint(g2, sp.rows_hi, plan.core_w_tiles, plan.core_w_tiles, 1, 1, plan.w_group_size)
        - pd._footprint(g2, 1, plan.core_w_tiles, plan.core_w_tiles, 1, 1, plan.w_group_size),
    )
    del tx, to, tg
ttnn.close_device(device)
