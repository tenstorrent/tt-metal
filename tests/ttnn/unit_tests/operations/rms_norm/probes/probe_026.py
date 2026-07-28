import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    print("AICLK MHz:", device.get_clock_rate_mhz() if hasattr(device, "get_clock_rate_mhz") else "N/A")
    grid = device.compute_with_storage_grid_size()
    print("grid:", grid.x, grid.y)
    print("virtual x runs:", pd._virtual_x_runs(device, grid))
    for shape in [(1, 1, 32, 1024), (1, 1, 32, 2304), (1, 1, 32, 5120), (1, 1, 32, 7168)]:
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
        print(
            f"{shape} Wt_global={wt} cw={p.cw} cores={p.num_cores} wt_core={blk.Wt} chunk={blk.wt_chunk} nw={blk.nw} ht_block={blk.ht_block} xres={blk.x_res_depth} gres={blk.gamma_resident} readbat={pd._x_read_chunks(blk)} cbKB={blk.cb_total_bytes//1024} groups={len(p.groups)}"
        )
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_g)
finally:
    ttnn.close_device(device)
