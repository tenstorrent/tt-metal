import importlib
import torch
import ttnn

op_mod = importlib.import_module("ttnn.operations.g6_zeros.g6_zeros")
from ttnn.operations.g6_zeros import g6_zeros

TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def call(shape, dtype):
    return g6_zeros(
        list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def poison(shape, dtype):
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.full(tuple(shape), 3.0, dtype=TORCH[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT), t
    )
    rb = ttnn.to_torch(t)
    assert bool((rb != 0).all()), "poison did not land"
    del rb
    addr = t.buffer_address()
    ttnn.deallocate(t)
    return addr


def all_zero_bits(out):
    rb = ttnn.to_torch(out).to(torch.float32).contiguous().view(torch.int32)
    return int((rb != 0).sum())


device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    # 1. cache hit at a new address: out1 stays alive, out2 must land on a freshly poisoned range B != A
    p3 = (8, 12, 128, 128)
    out1 = call(p3, ttnn.float32)
    a = out1.buffer_address()
    b = poison(p3, ttnn.float32)
    out2 = call(p3, ttnn.float32)
    print(
        f"CHECK patch: A={a:#x} B={b:#x} out2={out2.buffer_address():#x} landed={out2.buffer_address()==b} wrong={all_zero_bits(out2)} out1_wrong={all_zero_bits(out1)}"
    )
    ttnn.deallocate(out1)
    ttnn.deallocate(out2)

    # 2. dtypes outside SUPPORTED, through the public entry point with SUPPORTED widened in-process
    op_mod.SUPPORTED["dtype"] = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]
    for shape, dtype in (((1, 4, 256, 128), ttnn.bfloat16), ((32831, 1, 32, 256), ttnn.bfloat8_b)):
        warm = call(shape, dtype)
        ttnn.deallocate(warm)
        for rnd in (1, 2):
            addr = poison(shape, dtype)
            out = call(shape, dtype)
            print(f"CHECK {dtype} {shape} round {rnd}: landed={out.buffer_address()==addr} wrong={all_zero_bits(out)}")
            ttnn.deallocate(out)
finally:
    ttnn.close_mesh_device(device)
