import ttnn, torch
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print("GRID", g.x, g.y)
for shp in [(1, 1, 640, 4 * 1792), (1, 1, 640, 4 * 7168), (1, 1, 1280, 4 * 4096), (1, 1, 4096, 4 * 1792)]:
    for xd in (ttnn.bfloat16, ttnn.float32):
        x = ttnn.from_torch(torch.zeros(shp), dtype=xd, layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch(torch.zeros(shp[-1], 24), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        p = pd.make_plan(dev, x, w, 4)
        print(
            "PLAN",
            shp,
            xd,
            "gw",
            p.group_w,
            "gh",
            p.group_h,
            "groups",
            p.groups_x,
            p.groups_y,
            "bt",
            p.block_token_tiles,
            "kmax",
            p.core_k_tiles_max,
            "ctt",
            max(p.core_token_tiles),
        )
ttnn.close_device(dev)
