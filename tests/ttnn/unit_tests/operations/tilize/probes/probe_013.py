import torch, ttnn
from ttnn.operations.tilize.tilize import tilize

dev = ttnn.open_device(device_id=0)
print("RES align dram", ttnn.get_dram_alignment(), "l1", ttnn.get_l1_alignment())
for mem in [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]:
    for shape in [
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 1, 32, 96),
        (1, 1, 32, 64),
        (1, 1, 32, 1056),
        (1, 1, 3200, 32),
        (2, 3, 96, 1056),
    ]:
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        t = ttnn.from_torch(x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mem)
        y = ttnn.to_torch(tilize(t))
        bad = y != x
        rows = bad.any(-1).nonzero()[:, -2].unique().tolist()
        print(
            "RES",
            mem.buffer_type,
            shape,
            t.buffer_aligned_page_size() if hasattr(t, "buffer_aligned_page_size") else "",
            "nbad",
            bad.sum().item(),
            "badrows",
            rows[:10],
        )
ttnn.close_device(dev)
