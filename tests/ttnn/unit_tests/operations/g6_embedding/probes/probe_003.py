import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    allbits = torch.arange(65536, dtype=torch.int32)
    for shape in ((32, 2048), (1024, 64)):
        g = torch.Generator().manual_seed(1)
        u = allbits[torch.randperm(65536, generator=g)].reshape(shape)
        rm = ttnn.from_torch(
            u, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tl = ttnn.tilize(rm)
        back = ttnn.to_torch(tl).to(torch.int32) & 0xFFFF
        print("all-65536 uint16 tilize", shape, "mismatches", int((back != u).sum()))
finally:
    ttnn.close_device(device)
