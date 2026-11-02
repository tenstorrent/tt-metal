import math, torch, ttnn
import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

device = ttnn.open_device(device_id=0)
HIDDEN, FFN = 768, 3072


def run(batch, seq):
    torch.manual_seed(42)
    x = torch.randn((batch, 1, seq, HIDDEN)).to(torch.bfloat16)
    w = (torch.randn((HIDDEN, FFN)) * (1.7 / math.sqrt(HIDDEN))).to(torch.bfloat16)
    b = (torch.randn((FFN,)) * 0.1 - 2.8).to(torch.bfloat16)
    t = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **t)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **t)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **t)
    ref = torch.nn.functional.gelu(
        x.double().reshape(batch, seq, HIDDEN) @ w.double() + b.double(), approximate="tanh"
    ).flatten()
    res = {}
    for fast in (1, 0):
        device.disable_and_clear_program_cache()
        pd.GELU_FAST = fast
        out = ttnn.to_torch(linear_gelu(xd, wd, bd)).double().flatten()
        pcc = torch.corrcoef(torch.stack([out, ref]))[0, 1].item()
        rms = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        res[fast] = (pcc, rms, (out - ref).abs().max().item())
    print(
        f"shape {batch}x{seq} acc pcc={res[0][0]:.6f} rms={res[0][1]:.4e} max={res[0][2]:.3e} | fast pcc={res[1][0]:.6f} rms={res[1][1]:.4e} max={res[1][2]:.3e}"
    )


for s in [(8, 512), (2, 512), (1, 32)]:
    run(*s)
ttnn.close_device(device)
