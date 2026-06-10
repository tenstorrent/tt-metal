import math, torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def run(label, ckc):
    torch.manual_seed(0)
    q = torch.randn(1, 1, 128, 64)
    k = torch.randn(1, 1, 128, 64)
    v = torch.randn(1, 1, 128, 64)
    gold = torch.softmax(q @ k.transpose(-2, -1) / 8.0, -1) @ v
    t = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(t(q), t(k), t(v), compute_kernel_config=ckc)).float()
    g = gold.flatten()
    a = out.flatten()
    pcc = torch.corrcoef(torch.stack([g, a]))[0, 1].item()
    print(f"{label:30s} pcc={pcc:.6f}")


F = ttnn.MathFidelity
for fid in (F.HiFi4, F.HiFi3, F.HiFi2, F.LoFi):
    for acc in (True, False):
        if fid == F.HiFi4 and acc:
            continue
        run(
            f"{fid} fp32_acc={acc}",
            ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=acc, math_approx_mode=False),
        )
ttnn.close_device(device)
