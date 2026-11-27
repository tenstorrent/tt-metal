import ttnn
from ttnn.operations.g6_to_layout import select_plan

device = ttnn.open_device(device_id=0)
cases = [
    ("c301", (1, 128, 256, 256), ttnn.DRAM_MEMORY_CONFIG, None, None),
    ("c38", (1, 128, 256, 256), ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, None),
    ("c49", (1, 128, 256, 128), ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, None),
    ("c1989", (16, 1, 1536), ttnn.DRAM_MEMORY_CONFIG, None, None),
    ("c296", (1, 128, 256, 32), ttnn.DRAM_MEMORY_CONFIG, None, None),
    ("c16", (1, 1, 1, 256), ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat8_b),
    ("Q3", (1, 1, 32, 256), ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat8_b),
    ("c45", (1, 32, 128, 256), ttnn.L1_MEMORY_CONFIG, None, None),
]
for name, shape, mc, out_mc, dt in cases:
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, mc)
    kw = {}
    if out_mc is not None:
        kw["memory_config"] = out_mc
    if dt is not None:
        kw["dtype"] = dt
    p = select_plan(t, ttnn.TILE_LAYOUT, **kw)
    print(
        name,
        p["regime"],
        "cores",
        p["num_cores"],
        "block",
        p["block_rows_tiles"],
        "x",
        p["block_cols_tiles"],
        "stage_rows",
        p["tile_row_stage_rows"],
        "cb",
        p["cb_bytes"],
        "out_l1",
        p["out_l1_share"],
        "total",
        p["l1_bytes_per_core"],
    )
ttnn.close_device(device)
