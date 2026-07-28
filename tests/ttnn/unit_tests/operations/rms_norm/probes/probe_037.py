import ttnn, torch
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    print(f"grid={grid.x}x{grid.y}  aiclk={device.get_clock_rate_mhz()} MHz")
    l1b = pd._l1_total_budget(device)
    print(f"l1_total_budget={l1b}")

    CASES = [
        # (shape, kind, shard_shape, core_grid)
        ((1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8)),
        ((1, 1, 32, 1024), "WIDTH", [32, 128], (8, 1)),
        ((1, 1, 32, 7168), "WIDTH", [32, 256], (7, 4)),
        ((1, 1, 32, 5120), "WIDTH", [32, 160], (8, 4)),
    ]
    for shape, kind, ss, cg in CASES:
        ml = getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED")
        mc = shard_config(list(ss), cg, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        g = ttnn.from_torch(
            torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ht_total, wt = pd._tile_geometry(x)
        runs = pd._virtual_x_runs(device, grid)
        pl = pd._placement_sharded(x, ht_total, wt, runs)
        blk = pd._derive_blocking(x, g, grid.x * grid.y, pl, sharded_in=True, sharded_out=True, l1_total_budget=l1b)
        prog, shard = blk._cb_bytes(blk.x_res_depth, blk.gamma_resident)
        print(f"\n{shape} {kind} shard={ss} grid={cg}")
        print(f"  ht_total={ht_total} Wt_global={wt} cores={len(pl.works)} cw={pl.cw} cw1={pl.cw1} cw2={pl.cw2}")
        print(
            f"  per-core: rows_core_max={pl.rows_core_max} Wt={blk.Wt} wt_chunk={blk.wt_chunk} nw={blk.nw} "
            f"ht_block={blk.ht_block} nh_core={blk.nh_core_max}"
        )
        print(
            f"  fuse_sq={blk.fuse_sq} x_resident={blk.x_resident} gamma_resident={blk.gamma_resident} grid_full={blk.grid_full}"
        )
        print(f"  L1: prog={prog} shard={shard} total={prog+shard} / {l1b}")
        ttnn.deallocate(x)
        ttnn.deallocate(g)
finally:
    ttnn.close_device(device)
