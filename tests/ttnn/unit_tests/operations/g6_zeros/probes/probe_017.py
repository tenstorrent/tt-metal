import torch, ttnn
from ttnn.operations import g6_zeros as G
import ttnn.operations.g6_zeros.g6_zeros as M

device = ttnn.open_device(device_id=0)
try:
    # 1. Cache hit at a new address: keep the first output alive so the poisoned range differs.
    shape, dt = (8, 12, 128, 128), ttnn.float32
    out1 = G.g6_zeros(
        list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    a1 = out1.buffer_address()
    pat = torch.full(shape, 7.0)
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(pat, dtype=dt, layout=ttnn.TILE_LAYOUT), dirty)
    a2 = dirty.buffer_address()
    ttnn.deallocate(dirty)
    out2 = G.g6_zeros(
        list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    r2 = ttnn.to_torch(out2)
    print(
        f"PROBE cache-hit-new-address: a1={a1:#x} poisoned={a2:#x} out2={out2.buffer_address():#x} "
        f"nonzero={int((r2.view(torch.int32)!=0).sum())} out1_nonzero={int((ttnn.to_torch(out1).view(torch.int32)!=0).sum())}"
    )

    # 2. bf16 / bf8b past validate() over a poisoned range.
    saved = list(M.SUPPORTED["dtype"])
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
        a = dirty.buffer_address()
        ttnn.deallocate(dirty)
        out = G.g6_zeros(
            list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        r = ttnn.to_torch(out).to(torch.float32)
        print(
            f"PROBE {dt} {shape}: same_addr={out.buffer_address()==a} nonzero_bits={int((r.view(torch.int32)!=0).sum())} of {r.numel()}"
        )
        ttnn.deallocate(out)
    M.SUPPORTED["dtype"][:] = saved
    plan = G.describe_plan((32831, 1, 32, 256), dtype=ttnn.bfloat8_b, device=device)
    print("PROBE plan p1:", {k: v for k, v in plan.items() if k not in ("roles",)})
finally:
    ttnn.close_device(device)
