import torch, math, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
shape = (1, 1, 4096, 64)
B, H, S, D = shape
torch.manual_seed(0)


def run(Q, K, V, label, ck=None):
    ref = torch.softmax((Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(D), -1) @ V.float()
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=ck)).float()
    err = out - ref
    print(
        f"{label}: rms/std={err.pow(2).mean().sqrt()/ref.std():.5f} abs_rms={err.pow(2).mean().sqrt():.6f} max={err.abs().max():.5f} ref_std={ref.std():.6f}"
    )


Qr = torch.randn(shape, dtype=torch.bfloat16)
Kr = torch.randn(shape, dtype=torch.bfloat16)
Vr = torch.randn(shape, dtype=torch.bfloat16)
run(Qr, Kr, Vr, "randn HiFi2")
ck3 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
)
run(Qr, Kr, Vr, "randn HiFi3", ck3)
run(torch.zeros(shape, dtype=torch.bfloat16), Kr, Vr, "Q=0 (uniform P)")
run(Qr, Kr, torch.ones(shape, dtype=torch.bfloat16), "V=1 (O==1)")
ttnn.close_device(device)
