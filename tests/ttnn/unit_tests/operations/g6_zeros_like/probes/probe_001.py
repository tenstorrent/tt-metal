import torch, ttnn
from ttnn.operations.g6_zeros_like import describe_plan
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    for shape, dt, tdt in [
        ((1, 12, 128, 128), ttnn.float32, torch.float32),
        ((5, 1, 32, 256), ttnn.bfloat8_b, torch.bfloat16),
    ]:
        x = ttnn.from_torch(
            torch.randn(shape, dtype=tdt),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        p = pd.plan(x)
        print("PROBE", dt, "P", p.num_pages, "page", p.page_size, "cores", len(p.cores), "all_cores", p.all_cores)
        counts = {}
        for c in p.cores:
            counts[c.num_pages] = counts.get(c.num_pages, 0) + 1
        print("PROBE pages/core histogram", counts)
        # internal dispatch bypassing SUPPORTED (bf8 kernel path check only)
        ttnn.deallocate(
            ttnn.from_torch(
                torch.ones(shape, dtype=tdt),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        out = ttnn.allocate_tensor_on_device(x.spec, device)
        ttnn.generic_op([out], pd.get_program_descriptor(out))
        r = ttnn.to_torch(out).float()
        print("PROBE nonzero elements:", int((r != 0).sum()), "of", r.numel(), "neg-zero:", int(torch.signbit(r).sum()))
finally:
    ttnn.close_device(device)
