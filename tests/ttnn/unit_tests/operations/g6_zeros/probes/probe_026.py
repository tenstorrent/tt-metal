# Verifier probe: R1 readiness. Runs p4 (bf16) and p1 (bf8b) through the public entry point with
# SUPPORTED widened in this process only, over a poisoned freed range, under --dev (watcher, asserts).
import torch, ttnn
import sys
import ttnn.operations.g6_zeros  # noqa: F401

op = sys.modules["ttnn.operations.g6_zeros.g6_zeros"]

TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def call(shape, dt):
    return op.g6_zeros(
        list(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def poison(shape, dt):
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.full(shape, 3.0, dtype=TORCH[dt]), dtype=dt, layout=ttnn.TILE_LAYOUT), t
    )
    landed = bool((ttnn.to_torch(t) == 3.0).all())
    a = t.buffer_address()
    ttnn.deallocate(t)
    return a, landed


device = ttnn.open_device(device_id=0)
saved = list(op.SUPPORTED["dtype"])
try:
    op.SUPPORTED["dtype"][:] = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]
    for shape, dt in (((1, 4, 256, 128), ttnn.bfloat16), ((32831, 1, 32, 256), ttnn.bfloat8_b)):
        ttnn.deallocate(call(shape, dt))
        for rnd in (1, 2):
            a, landed = poison(shape, dt)
            out = call(shape, dt)
            r = ttnn.to_torch(out).to(torch.float32)
            bits = r.contiguous().view(torch.int32)
            print(
                f"PROBE {dt} {shape} round{rnd}: poison_landed={landed} same_addr={out.buffer_address()==a} "
                f"nonzero_bits={int((bits != 0).sum())} neg_zero={int((bits == -2**31).sum())} numel={r.numel()}"
            )
            del r, bits
            ttnn.deallocate(out)
finally:
    op.SUPPORTED["dtype"][:] = saved
    ttnn.close_device(device)
