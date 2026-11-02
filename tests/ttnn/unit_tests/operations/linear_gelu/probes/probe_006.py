import sys, torch, ttnn, math
import ttnn.operations.linear_gelu

mod = sys.modules["ttnn.operations.linear_gelu.linear_gelu"]
from eval.golden_tests.linear_gelu.helpers import make_inputs, pytorch_linear_gelu, read_padded

mod.SUPPORTED["buffer_type"].append(ttnn.BufferType.DRAM)
mod.SUPPORTED["out_buffer_type"].append(ttnn.BufferType.DRAM)
mod.SUPPORTED["alignment"].append("h_non_aligned")
dev = ttnn.open_device(device_id=0)


def t(x, mc):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


D, L = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
for b, s in [(8, 512), (2, 37), (8, 500)]:
    x, w, bias = make_inputs(b, s)
    ref = pytorch_linear_gelu(x, w, bias)
    for xm, om in [(D, L), (L, D), (D, D), (L, L)]:
        out = mod.linear_gelu(t(x, xm), t(w, D), t(bias, D), memory_config=om)
        a = ttnn.to_torch(out).double()
        pcc = torch.corrcoef(torch.stack([a.flatten(), ref.flatten()]))[0, 1].item()
        rr = ((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        pad = read_padded(out)[..., s:, :]
        print(
            f"RES {b}x{s} x={xm.buffer_type} out={om.buffer_type} pcc={pcc:.6f} rel_rms={rr:.4e} pad_nonzero={(pad != 0).sum().item()}"
        )
ttnn.close_device(dev)
