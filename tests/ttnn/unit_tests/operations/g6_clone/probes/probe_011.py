import torch, ttnn
from ttnn.operations.g6_clone.g6_clone import g6_clone_plan

device = ttnn.open_device(device_id=0)
for shape, dt in [
    ((16, 12, 128, 128), ttnn.float32),
    ((1, 512, 5120), ttnn.bfloat16),
    ((8, 12, 128, 128), ttnn.float32),
    ((1, 1, 32, 62080), ttnn.bfloat16),
]:
    x = ttnn.from_torch(
        torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    p = g6_clone_plan(x)
    print(
        "PLAN",
        shape,
        p["regime"],
        p["num_cores"],
        p["core_pages_max"],
        p["block_pages"],
        p["cb_depth_blocks"],
        p["num_blocks_max"],
        p["cb_depth_blocks"] * p["block_pages"] * p["aligned_page_bytes"],
        len(p["core_ranges"].ranges()),
    )
ttnn.close_device(device)
