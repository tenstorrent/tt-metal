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


def stats(out, dt):
    r = ttnn.to_torch(out)
    if dt == ttnn.bfloat8_b:
        r = r.to(torch.float32)
    view = torch.int32 if r.dtype == torch.float32 else torch.int16
    bits = r.contiguous().view(view)
    return int((bits != 0).sum()), int((bits == torch.iinfo(view).min).sum()), float(r.abs().max()), r.numel()


device = ttnn.open_device(device_id=0)
try:
    SUP["dtype"][:] = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]
    for shape, dt in (((1, 4, 256, 128), ttnn.bfloat16), ((32831, 1, 32, 256), ttnn.bfloat8_b)):
        warm = call(shape, dt)
        ttnn.deallocate(warm)
        for rnd in (1, 2):
            a, landed = poison(shape, dt)
            out = call(shape, dt)
            nz, negz, mx, n = stats(out, dt)
            print(
                f"PROBE {dt} {shape} round{rnd}: poison_landed={landed} same_addr={out.buffer_address()==a} nonzero_bits={nz} neg_zero={negz} max_abs={mx} numel={n}"
            )
            ttnn.deallocate(out)
    for shape, dt in (
        ((32831, 1, 32, 256), ttnn.bfloat8_b),
        ((1, 4, 256, 128), ttnn.bfloat16),
        ((16, 12, 128, 128), ttnn.float32),
    ):
        p = G.describe_plan(shape, dtype=dt, device=device)
        print("PROBE plan", shape, {k: v for k, v in p.items() if k not in ("roles", "core_ranges")})
finally:
    SUP["dtype"][:] = [ttnn.float32]
    ttnn.close_device(device)
