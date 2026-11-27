import torch, ttnn
from ttnn.operations.g6_to_layout import g6_to_layout, select_plan

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    print(
        "banks dram",
        ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks,
        "l1",
        ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks,
    )
    for shape in [
        (1, 1024, 1024, 3),
        (1, 2048, 2048, 3),
        (1, 1024, 2048, 128),
        (16, 4096, 2560),
        (1, 4096, 2560),
        (1, 131072),
        (16, 1, 2560),
        (32, 5120),
        (1, 512),
    ]:
        x = torch.randn(shape, dtype=torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        p = select_plan(t, ttnn.ROW_MAJOR_LAYOUT)
        print(
            "plan",
            shape,
            "cores",
            p.cores_used,
            "chunks",
            p.num_col_chunks,
            "segw",
            p.segment_width,
            "nseg",
            p.num_segments,
            "blk",
            p.block_segments,
            "g1",
            len(p.group_1),
            p.group_1_segments,
            "g2",
            len(p.group_2),
            p.group_2_segments,
            "cb_tiles",
            p.cb_tiles_pages,
        )
    out = g6_to_layout(
        ttnn.from_torch(
            torch.randn((1, 64, 3), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        ttnn.ROW_MAJOR_LAYOUT,
    )
    print("page", out.buffer_page_size())
finally:
    ttnn.close_device(device)
