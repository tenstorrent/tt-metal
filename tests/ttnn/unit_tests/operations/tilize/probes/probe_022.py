import torch, ttnn
from ttnn.operations.tilize import tilize as gen_tilize

dev = ttnn.open_device(device_id=0)
grid = dev.compute_with_storage_grid_size()
print("GRID:", grid.x, grid.y, "=", grid.x * grid.y)

# Shapes that ENGAGE width-split (nt_h*4 < grid_cores, C>=2) and a couple that don't.
cases = [
    ([1, 1, 32, 16384], ttnn.bfloat16),  # nt_h=1, Wt=512 -> engage
    ([8, 1, 32, 7168], ttnn.bfloat16),  # nt_h=8, Wt=224 -> engage
    ([1, 1, 32, 512], ttnn.bfloat16),  # nt_h=1, Wt=16 -> engage (C=2)
    ([1, 1, 96, 2048], ttnn.bfloat16),  # nt_h=3, Wt=64 -> engage
    ([1, 1, 32, 16384], ttnn.float32),  # fp32 engage
    ([2, 1, 32, 4096], ttnn.uint32),  # uint32 nt_h=2 engage
    ([1, 1, 2048, 2048], ttnn.bfloat16),  # nt_h=64 -> height path (no engage)
    ([512, 512], ttnn.bfloat16),  # nt_h=16 -> height path
]
tdtype = {ttnn.bfloat16: torch.float32, ttnn.float32: torch.float32, ttnn.uint32: torch.int32}

for shape, dt in cases:
    if dt == ttnn.uint32:
        t = torch.randint(0, 100000, shape, dtype=torch.int32)
    else:
        t = torch.randn(shape, dtype=torch.float32)
    tin = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = gen_tilize(tin)
    r = ttnn.to_torch(out)
    ref = t.to(r.dtype) if dt != ttnn.uint32 else t
    if dt == ttnn.bfloat16:
        ok = torch.equal(r.float(), t.to(torch.bfloat16).float())
    elif dt == ttnn.float32:
        ok = torch.equal(r.float(), t)
    else:
        ok = torch.equal(r.to(torch.int64), t.to(torch.int64))
    md = (
        (r.float() - t.float()).abs().max().item()
        if dt != ttnn.uint32
        else (r.to(torch.int64) - t.to(torch.int64)).abs().max().item()
    )
    print(f"{str(shape):22} {str(dt):16} identity={ok} max_diff={md}")

ttnn.close_device(dev)
