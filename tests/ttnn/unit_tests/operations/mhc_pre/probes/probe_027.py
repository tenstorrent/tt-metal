import torch, ttnn
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
for shp in [
    (1, 1, 640, 4 * 1792),
    (1, 1, 640, 4 * 7168),
    (1, 1, 1280, 4 * 4096),
    (1, 1, 4096, 4 * 1792),
    (1, 1, 1, 4 * 7168),
    (1, 1, 1000, 4 * 7168),
    (1, 1, 256, 4 * 6144),
]:
    for xd in (ttnn.bfloat16, ttnn.float32):
        tx = ttnn.from_torch(torch.zeros(shp), dtype=xd, layout=ttnn.TILE_LAYOUT, device=dev)
        tw = ttnn.from_torch(torch.zeros(shp[-1], 24), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        for nar in (False, True):
            pd.NARROW_GROUPS = nar
            p = pd.make_plan(dev, tx, tw, 4)
            print(
                "PLAN",
                shp,
                xd,
                nar,
                "gw",
                p.group_w,
                "gh",
                p.group_h,
                "groups",
                p.num_groups,
                "kmax",
                p.core_k_tiles_max,
                "ctt",
                max(p.core_token_tiles),
                "bt",
                p.block_token_tiles,
                "depth",
                p.x_block_depth,
            )
ttnn.close_device(dev)
