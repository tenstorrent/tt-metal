import ttnn
from ttnn.operations.g6_to_layout import select_plan

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    print(
        "BANKS dram",
        ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks,
        "l1",
        ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks,
    )
    for shape, mc in [
        ((1, 2048, 2048, 3), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1024, 2048, 128), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1024, 1024, 3), ttnn.DRAM_MEMORY_CONFIG),
        ((16, 4096, 2560), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 4096, 2560), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 131072), ttnn.DRAM_MEMORY_CONFIG),
        ((16, 1, 2560), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 32, 2560), ttnn.L1_MEMORY_CONFIG),
        ((8, 1024, 2560), ttnn.DRAM_MEMORY_CONFIG),
        ((1, 185), ttnn.DRAM_MEMORY_CONFIG),
    ]:
        t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        p = select_plan(t, ttnn.ROW_MAJOR_LAYOUT)
        cores = p["cores"]
        print(
            "PLAN",
            shape,
            p["regime"],
            "cores",
            p["num_cores"],
            "tiles/core",
            cores[0]["num_tiles"],
            cores[-1]["num_tiles"],
            "rows",
            p["block_rows_tiles"],
            "cols",
            p["block_cols_tiles"],
            "stage_rows",
            p["tile_row_stage_rows"],
            "cb",
            p["cb_bytes"],
            "l1",
            p["l1_bytes_per_core"],
        )
finally:
    ttnn.close_device(device)
