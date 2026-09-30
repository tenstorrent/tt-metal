import ttnn, torch
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
print("L1 unreserved", ttnn.get_max_worker_l1_unreserved_size())
for shp in [(1, 1, 2048, 20480)]:
    x = ttnn.from_torch(torch.zeros(shp), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(torch.zeros(shp[-1], 24), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    p = pd.make_plan(dev, x, w, 4)
    print(
        "PLAN",
        shp,
        "gw",
        p.group_w,
        "gh",
        p.group_h,
        "groups",
        p.groups_x,
        p.groups_y,
        "bt",
        p.block_token_tiles,
        "depth",
        p.x_block_depth,
        "kmax",
        p.core_k_tiles_max,
        "ctt",
        max(p.core_token_tiles),
        "y_chunk",
        p.y_chunk_tiles,
    )
    kw = dict(
        bt=p.block_token_tiles,
        depth=p.x_block_depth,
        kmax=p.core_k_tiles_max,
        G=p.group_cores,
        y_chunk=p.y_chunk_tiles,
        n=4,
        x_tile=x.buffer_page_size(),
        w_tile=w.buffer_page_size(),
        y_tile=x.buffer_page_size(),
        x_dtype=x.dtype,
        w_dtype=w.dtype,
        y_dtype=x.dtype,
    )
    print("L1 footprint", pd._l1_bytes(**kw), "budget", ttnn.get_max_worker_l1_unreserved_size() - pd.L1_SAFETY_MARGIN)
    for e in pd._cb_table(**kw):
        print("TAB", e[0], e[1], e[2], e[1] * e[2])
ttnn.close_device(dev)
