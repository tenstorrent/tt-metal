import torch, ttnn
from ttnn.operations.g6_to_layout import plan_for

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("PLAN grid", g.x, g.y)
for shape, mc in [
    ((1, 2048, 2048, 3), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 1024, 2048, 128), ttnn.DRAM_MEMORY_CONFIG),
    ((16, 4096, 2560), ttnn.DRAM_MEMORY_CONFIG),
    ((32, 5120), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 131072), ttnn.DRAM_MEMORY_CONFIG),
    ((16, 1, 2560), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 185), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 32, 2560), ttnn.L1_MEMORY_CONFIG),
]:
    x = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    p = plan_for(x, ttnn.ROW_MAJOR_LAYOUT)
    print(
        "PLAN",
        shape,
        {
            k: p[k]
            for k in (
                "tensor_tile_rows",
                "tensor_tile_cols",
                "block_tile_cols",
                "num_col_chunks",
                "block_tile_rows",
                "num_units",
                "num_cores_used",
            )
        },
    )
ttnn.close_device(device)
