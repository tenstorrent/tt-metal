import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    for shape in [
        (1, 1, 32, 1024),
        (1, 1, 32, 2304),
        (1, 1, 32, 5120),
        (1, 1, 32, 7168),
        (1, 1, 8192, 1024),
        (1, 1, 64, 12288),
        (1, 1, 32, 16384),
        (1, 1, 32, 32768),
    ]:
        tt_x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_g = ttnn.from_torch(
            torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ht, wt = pd._tile_geometry(tt_x)
        p = pd._select_placement(device, grid, tt_x, ht, wt, False)
        blk = pd._derive_blocking(tt_x, tt_g, grid.x * grid.y, p)
        nleaders = sum(1 for w in p.works if w.is_leader)
        print(
            f"{str(shape):18s} Wt={wt:5d} cw={p.cw:3d} cw1={p.cw1:3d} cw2={p.cw2:3d} cores={p.num_cores:4d} leaders={nleaders:3d} wt_core={blk.Wt:3d} nw={blk.nw} cbKB={blk.cb_total_bytes//1024}"
        )
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_g)
finally:
    ttnn.close_device(device)
