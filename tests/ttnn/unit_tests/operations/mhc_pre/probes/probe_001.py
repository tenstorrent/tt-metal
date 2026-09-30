import torch, ttnn
from ttnn.operations.mhc_pre.mhc_pre_program_descriptor import make_plan

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y, "budget", ttnn.get_max_worker_l1_unreserved_size())
for shape in [
    (32, 128),
    (64, 1024),
    (1, 128, 4096),
    (1, 2, 64, 2048),
    (17, 512),
    (1, 1, 640, 7168),
    (1, 1, 640, 28672),
    (1, 1, 4096, 28672),
    (1, 1, 1280, 16384),
    (1, 28672),
    (1, 1, 256, 24576),
    (1, 1, 2048, 28672),
]:
    x = ttnn.from_torch(torch.zeros(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(torch.zeros(shape[-1], 24), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    p = make_plan(device, x, w, 4)
    busy = sum(1 for c in p.core_token_tiles if c) * p.group_cores
    print(
        "PLAN",
        shape,
        f"Mt={p.Mt} Ct={p.Ct} grp={p.group_w}x{p.group_h} G={p.group_cores} Gt={p.num_groups} busy={busy} kmax={p.core_k_tiles_max} ctt={max(p.core_token_tiles)} bt={p.block_token_tiles} depth={p.x_block_depth} blocks={-(-max(p.core_token_tiles)//p.block_token_tiles)}",
    )
ttnn.close_device(device)
