import math, torch, ttnn
import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

device = ttnn.open_device(device_id=0)
HIDDEN, FFN = 768, 3072
for batch, seq in [(8, 512), (2, 512), (1, 32), (4, 256)]:
    torch.manual_seed(42)
    x = torch.randn((batch, 1, seq, HIDDEN)).to(torch.bfloat16)
    w = (torch.randn((HIDDEN, FFN)) * (1.7 / math.sqrt(HIDDEN))).to(torch.bfloat16)
    b = (torch.randn((FFN,)) * 0.1 - 2.8).to(torch.bfloat16)
    t = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **t)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **t)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **t)
    z = x.double().reshape(batch, seq, HIDDEN) @ w.double() + b.double()
    ref = torch.nn.functional.gelu(z, approximate="tanh").flatten()
    out = ttnn.to_torch(linear_gelu(xd, wd, bd)).double().flatten()
    torch.save(out.float(), f"/tmp/out_fast{pd.GELU_FAST}_{batch}x{seq}.pt")
    pcc = torch.corrcoef(torch.stack([out, ref]))[0, 1].item()
    rms = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print(f"shape {batch}x{seq} fast={pd.GELU_FAST} pcc={pcc:.7f} rms={rms:.5e} max={(out-ref).abs().max().item():.4e}")
ttnn.close_device(device)
