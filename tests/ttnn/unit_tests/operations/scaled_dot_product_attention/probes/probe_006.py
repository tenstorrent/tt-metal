import math, torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def run(dtype, shape, ckc=None):
    torch.manual_seed(0)
    B, H, S, D = shape
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    scale = 1.0 / math.sqrt(D)
    gold = torch.softmax(q @ k.transpose(-2, -1) * scale, -1) @ v
    tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kw = {"compute_kernel_config": ckc} if ckc else {}
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, **kw)).float()
    g = gold.flatten()
    a = out.flatten()
    pcc = torch.corrcoef(torch.stack([g, a]))[0, 1].item()
    rms = ((a - g).pow(2).mean().sqrt() / g.std()).item()
    print(f"{str(dtype):20s} {shape} pcc={pcc:.6f} rms={rms:.5f}")


for dt in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
    run(dt, (1, 1, 128, 64))
    run(dt, (1, 4, 256, 64))
run(ttnn.float32, (1, 1, 128, 1024))  # L1-pressure corner
run(
    ttnn.bfloat16,
    (1, 1, 128, 64),
    ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False),
)
try:
    run(
        ttnn.bfloat16,
        (1, 1, 128, 64),
        ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    print("known-bad guard MISSING")
except ValueError as e:
    print("known-bad guard OK:", str(e)[:60])
ttnn.close_device(device)
