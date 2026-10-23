import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    C = grid.x * grid.y
    budget = pd._l1_budget_bytes(device)
    print("PLAN budget", budget, "cores", C)
    cases = [
        ((1, 1, 8192, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 8192, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 8192, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 16384), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 16300), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 64, 12288), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 4096, 8192), ttnn.float32, ttnn.TILE_LAYOUT),
        ((1, 1, 4096, 8192), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((1, 1, 4096, 8192), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ]
    for shape, dt, lay in cases:
        x = ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=lay, device=device)
        g = ttnn.from_torch(
            torch.zeros((1, 1, 1, shape[-1])),
            dtype=dt if dt != ttnn.bfloat8_b else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        for fp32 in (True, False):
            if dt == ttnn.float32 and not fp32:
                continue
            geo = pd._geometry(x, x, g, fp32)
            p = pd._select_blocking(geo, C, budget)
            print(
                f"PLAN {shape} {dt} {lay} fp32={fp32}: gw={p.w_group_size} n={p.core_w_tiles} gr={p.num_row_groups} b={p.block_rows} c={p.chunk_w_tiles} k={p.chunks_per_row} L={p.xg_lookahead_chunks}"
            )
        x.deallocate()
        g.deallocate()
finally:
    ttnn.close_device(device)
