import torch, ttnn
from ttnn.operations import g6_zeros as G

SUP = G.validate.__globals__["SUPPORTED"]
TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def call(shape, dt):
    return G.g6_zeros(
        list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def poison(shape, dt):
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.full(shape, 3.0, dtype=TORCH[dt]), dtype=dt, layout=ttnn.TILE_LAYOUT), t
    )
    chk = ttnn.to_torch(t)
    landed = bool((chk == 3.0).all())
    del chk
    a = t.buffer_address()
    ttnn.deallocate(t)
    return a, landed


def nonzero_bits(out):
    r = ttnn.to_torch(out).to(torch.float32).contiguous().view(torch.int32)
    return int((r != 0).sum())


device = ttnn.open_device(device_id=0)
try:
    p3 = (8, 12, 128, 128)
    out1 = call(p3, ttnn.float32)
    a1 = out1.buffer_address()
    b, landed = poison(p3, ttnn.float32)
    out2 = call(p3, ttnn.float32)
    print(
        f"PROBE cache-hit-new-address: a1={a1:#x} poisoned={b:#x} landed={landed} out2={out2.buffer_address():#x} "
        f"same={out2.buffer_address()==b} out2_nonzero={nonzero_bits(out2)} out1_nonzero={nonzero_bits(out1)}"
    )
    ttnn.deallocate(out1)
    ttnn.deallocate(out2)

    SUP["dtype"][:] = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]
    for shape, dt in (((1, 4, 256, 128), ttnn.bfloat16), ((32831, 1, 32, 256), ttnn.bfloat8_b)):
        warm = call(shape, dt)
        ttnn.deallocate(warm)
        for rnd in (1, 2):
            a, landed = poison(shape, dt)
            out = call(shape, dt)
            print(
                f"PROBE {dt} {shape} round{rnd}: poison_landed={landed} same_addr={out.buffer_address()==a} nonzero_bits={nonzero_bits(out)}"
            )
            ttnn.deallocate(out)
    plan = G.describe_plan((32831, 1, 32, 256), dtype=ttnn.bfloat8_b, device=device)
    print("PROBE plan p1:", {k: v for k, v in plan.items() if k != "roles"})
    plan = G.describe_plan((1, 4, 256, 128), dtype=ttnn.bfloat16, device=device)
    print("PROBE plan p4:", {k: v for k, v in plan.items() if k != "roles"})
finally:
    SUP["dtype"][:] = [ttnn.float32]
    ttnn.close_device(device)
