import torch, ttnn
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd

d = ttnn.open_device(device_id=0)
for shp in [(1, 1, 1280, 16384), (1, 1, 640, 7168), (1, 1, 640, 28672), (1, 1, 4096, 7168)]:
    for xd in [ttnn.bfloat16, ttnn.float32]:
        for cap, dep in [(1, 2), (2, 2), (4, 2), (8, 2), (1, 3), (2, 3)]:
            pd.BLOCK_TOKEN_TILES_CAP = cap
            pd.X_BLOCK_DEPTH_DEFAULT = dep
            tx = ttnn.from_torch(torch.zeros(shp), dtype=xd, layout=ttnn.TILE_LAYOUT, device=d)
            tw = ttnn.from_torch(torch.zeros(shp[-1], 24), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
            p = pd.make_plan(d, tx, tw, 4)
            print(
                "PLAN",
                shp[-2],
                shp[-1] // 4,
                xd,
                cap,
                dep,
                "grid",
                p.grid_x,
                p.grid_y,
                "gw",
                p.group_w,
                "gh",
                p.group_h,
                "ngroups",
                p.num_groups,
                "ctt",
                max(p.core_token_tiles),
                "cc",
                p.core_c_tiles[:3],
                "bt",
                p.block_token_tiles,
                "depth",
                p.x_block_depth,
                "ych",
                p.y_chunk_tiles,
            )
ttnn.close_device(d)
