import torch, ttnn
from ttnn.operations.mhc_pre import mhc_pre_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    for dt in (ttnn.float32, ttnn.bfloat16):
        for nc in (4 * 7168, 4 * 1792, 4 * 4096):
            x = ttnn.from_torch(torch.zeros(1, 1, 640, nc), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            w = ttnn.from_torch(torch.zeros(nc, 24), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            p = pd.make_plan(device, x, w, 4)
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
                x_dtype=dt,
                w_dtype=ttnn.float32,
                y_dtype=dt,
            )
            print(
                "PROBE",
                dt,
                nc,
                "group",
                p.group_w,
                p.group_h,
                "kmax",
                p.core_k_tiles_max,
                "bt",
                p.block_token_tiles,
                "depth",
                p.x_block_depth,
                "L1",
                pd._l1_bytes(**kw),
                "budget",
                ttnn.get_max_worker_l1_unreserved_size() - pd.L1_SAFETY_MARGIN,
            )
finally:
    ttnn.close_device(device)
