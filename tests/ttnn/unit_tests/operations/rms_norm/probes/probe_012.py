import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:
    # all ones -> output should be ~1 everywhere
    x = torch.ones((32, 64), dtype=torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(rms_norm(tx)).to(torch.float32)
    print("ONES row0:", out[0, :8].tolist())
    print("ONES row1:", out[1, :8].tolist())
    print("ONES min/max:", out.min().item(), out.max().item())

    # monotonic: value = r*100 + c  (bf16 exact up to 256)
    y = torch.zeros((32, 64), dtype=torch.float32)
    for r in range(32):
        for c in range(64):
            y[r, c] = r + c / 100.0
    y = y.to(torch.bfloat16)
    ty = ttnn.from_torch(
        y, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    o2 = ttnn.to_torch(rms_norm(ty)).to(torch.float32)
    yf = y.to(torch.float32)
    e2 = yf * torch.rsqrt(yf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    print("MONO actual row3:", o2[3, :6].tolist())
    print("MONO expect row3:", e2[3, :6].tolist())
    print("MONO actual row3 tail:", o2[3, 60:64].tolist())
    print("MONO expect row3 tail:", e2[3, 60:64].tolist())
finally:
    ttnn.close_device(device)
