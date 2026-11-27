import torch, ttnn
from ttnn.operations.g6_clone import g6_clone_plan

device = ttnn.open_device(device_id=0)
try:
    for shape in [(16, 12, 128, 128), (8, 12, 128, 128), (1, 12, 128, 128), (1, 1)]:
        x = ttnn.from_torch(
            torch.zeros(shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        p = g6_clone_plan(x)
        pages = [r[3] for r in p["roles"]]
        print(
            "PLAN",
            shape,
            "pages",
            p["tensor_pages"],
            "pgB",
            p["aligned_page_bytes"],
            "grid",
            p["grid"],
            "cores",
            p["num_cores"],
            "g1/g2",
            p["core_pages_g1"],
            p["core_pages_g2"],
            "block",
            p["block_pages"],
            "depth",
            p["cb_depth_blocks"],
            "blocks",
            p["num_blocks_max"],
            "ranges",
            len(p["core_ranges"].ranges()),
            "min/max",
            min(pages),
            max(pages),
        )
finally:
    ttnn.close_device(device)
