import sys, time, torch, ttnn
import ttnn.operations.linear_gelu

mod = sys.modules["ttnn.operations.linear_gelu.linear_gelu"]
desc = sys.modules["ttnn.operations.linear_gelu.linear_gelu_program_descriptor"]
from eval.golden_tests.linear_gelu.helpers import make_inputs, pytorch_linear_gelu

dev = ttnn.open_device(device_id=0, l1_small_size=0)


def t(x, mc):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


D, L = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
for b, s in [(8, 512), (2, 512)]:
    x, w, bias = make_inputs(b, s)
    ref = pytorch_linear_gelu(x, w, bias)
    xd, wd, bd = t(x, L), t(w, D), t(bias, D)
    for fast in (0, 1):
        desc.GELU_FAST = fast
        out = mod.linear_gelu(xd, wd, bd)
        a = ttnn.to_torch(out).double()
        out.deallocate()
        pcc = torch.corrcoef(torch.stack([a.flatten(), ref.flatten()]))[0, 1].item()
        rr = ((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        n = 50
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            o = mod.linear_gelu(xd, wd, bd)
            o.deallocate()
        ttnn.synchronize_device(dev)
        print(
            f"RES {b}x{s} fast={fast} wall_per_call={(time.perf_counter()-t0)/n*1e6:.1f}us pcc={pcc:.6f} rel_rms={rr:.4e}"
        )
ttnn.close_device(dev)
