import torch, ttnn
import importlib

gz = importlib.import_module("ttnn.operations.g6_zeros.g6_zeros")
from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


def dispatch_past_validate(shape, dtype, device):
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = gz._plan_for(shape, dtype, device)
    assert plan.num_pages == out.buffer_num_pages() and plan.aligned_page_size == out.buffer_aligned_page_size()
    ttnn.generic_op([out], pdm.create_program_descriptor(out, plan))
    return out


def poison(shape, dtype, device):
    pat = torch.randn(tuple(shape), dtype=TORCH[dtype]).abs_().add_(1.0)
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(pat, dtype=dtype, layout=ttnn.TILE_LAYOUT), t)
    del pat
    assert not bool((ttnn.to_torch(t) == 0).any())
    a = t.buffer_address()
    ttnn.deallocate(t)
    return a


device = ttnn.open_device(device_id=0)
try:
    for shape, dtype in [
        ((1, 4, 256, 128), ttnn.bfloat16),
        ((32831, 1, 32, 256), ttnn.bfloat8_b),
        ((8, 12, 128, 128), ttnn.float32),
    ]:
        warm = dispatch_past_validate(shape, dtype, device)
        ttnn.deallocate(warm)
        addr = poison(shape, dtype, device)
        out = dispatch_past_validate(shape, dtype, device)
        assert out.buffer_address() == addr, "did not land on poisoned range"
        rb = ttnn.to_torch(out)
        cmp = torch.float32 if dtype == ttnn.bfloat8_b else TORCH[dtype]
        wrong = 0
        for s in range(0, shape[0], 4096):
            sl = rb[s : s + 4096].to(cmp).contiguous().view(BITS[cmp])
            wrong += int((sl != 0).sum())
        print(f"PROBE {shape} {dtype}: wrong_bits={wrong} of {rb.numel()} on poisoned range {addr:#x}")
        del rb
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
