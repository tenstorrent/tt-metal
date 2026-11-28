import sys, importlib, torch, ttnn
from ttnn.operations import g6_zeros as G

M = importlib.import_module("ttnn.operations.g6_zeros.g6_zeros")

device = ttnn.open_device(device_id=0)
try:
    M.SUPPORTED["dtype"][:] = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]
    for shape, dt in (((1, 4, 256, 128), ttnn.bfloat16), ((32831, 1, 32, 256), ttnn.bfloat8_b)):
        warm = G.g6_zeros(
            list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(warm)
        dirty = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.full(shape, 3.0, dtype=torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT), dirty
        )
        chk = ttnn.to_torch(dirty)
        landed = bool((chk == 3.0).all())
        del chk
        a = dirty.buffer_address()
        ttnn.deallocate(dirty)
        out = G.g6_zeros(
            list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        r = ttnn.to_torch(out).to(torch.float32)
        print(
            f"PROBE {dt} {shape}: poison_landed={landed} same_addr={out.buffer_address()==a} nonzero_bits={int((r.view(torch.int32)!=0).sum())} of {r.numel()}"
        )
        del r
        ttnn.deallocate(out)
    plan = G.describe_plan((32831, 1, 32, 256), dtype=ttnn.bfloat8_b, device=device)
    print("PROBE plan p1:", {k: v for k, v in plan.items() if k not in ("roles",)})
finally:
    ttnn.close_device(device)
