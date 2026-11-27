import torch, ttnn
from ttnn.operations.g6_untilize import describe_plan, g6_untilize
from ttnn.operations.g6_untilize.g6_untilize_program_descriptor import plan, create_program_descriptor

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    shape = (1, 1, 32, 128)
    x = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64).to(torch.int32).view(torch.uint32)
    xd = ttnn.from_torch(
        x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    d = describe_plan(xd)
    print(
        "PLAN",
        d["regime"],
        d["rectangle"],
        d["knobs"],
        d["staging_bytes_per_core"],
        len(d["cores"]),
        list(d["cores"].items())[:3],
    )
    grid = device.compute_with_storage_grid_size()
    print("GRID", grid.x, grid.y)
    cfgs = {
        "16c_2rows_1blk": dict(max_cores=16),
        "16c_1row_2blk_depth2": dict(max_cores=16, block_rows=1),
        "8c_1row_4blk_depth2": dict(max_cores=8, block_rows=1),
        "8c_1row_4blk_depth1": dict(max_cores=8, block_rows=1, staging_depth=1),
        "32c_halfrow": dict(block_tile_cols=2),
        "32c_tilecol1_depth2": dict(block_tile_cols=1),
        "32c_unrotated": dict(rotate_tile_order=False),
        "1core": dict(max_cores=1),
        "1core_3rows_blk": dict(max_cores=1, block_rows=3, block_tile_cols=3),
    }
    for name, kw in cfgs.items():
        p = plan(shape, 4, grid, **kw)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        desc, _ = create_program_descriptor(p, xd, out)
        r = ttnn.to_torch(ttnn.generic_op([xd, out], desc))
        ok = torch.equal(r.view(torch.int32), x.view(torch.int32))
        print(
            f"{name:24s} {p.cx}x{p.cy} rows={p.block_rows} cols={p.block_tile_cols} depth={p.staging_depth} -> {'OK' if ok else 'MISMATCH'}"
        )
finally:
    ttnn.close_device(device)
