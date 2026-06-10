import math, torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def run(label, ckc, dtype=ttnn.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(1, 1, 128, 64)
    k = torch.randn(1, 1, 128, 64)
    v = torch.randn(1, 1, 128, 64)
    gold = torch.softmax(q @ k.transpose(-2, -1) / 8.0, -1) @ v
    t = lambda x: ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(t(q), t(k), t(v), compute_kernel_config=ckc)).float()
    g = gold.flatten()
    a = out.flatten()
    print(f"{label:35s} pcc={torch.corrcoef(torch.stack([g,a]))[0,1].item():.6f}")


F = ttnn.MathFidelity
for fid in (F.HiFi4, F.HiFi3, F.HiFi2, F.LoFi):
    run(
        f"bf16 {fid} acc=False",
        ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=False, math_approx_mode=False),
    )
run(
    "fp32 HiFi4 acc=False",
    ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False),
    ttnn.float32,
)
run(
    "bf8b HiFi2 acc=False",
    ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False),
    ttnn.bfloat8_b,
)
run(
    "bf16 HiFi2 acc=True approx=True",
    ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi2, fp32_dest_acc_en=True, math_approx_mode=True),
)
run(
    "bf16 HiFi2 acc=True fullsync",
    ttnn.WormholeComputeKernelConfig(
        math_fidelity=F.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True
    ),
)
ttnn.close_device(device)
