import torch, math, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
ck3 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
)


def vones(S, ck, label):
    sh = (1, 1, S, 64)
    torch.manual_seed(0)
    Q = torch.randn(sh, dtype=torch.bfloat16)
    K = torch.randn(sh, dtype=torch.bfloat16)
    V = torch.ones(sh, dtype=torch.bfloat16)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=ck)).float()[0, 0]
    err = out - 1.0
    print(f"PROBE V=1 S={S} {label}: max={err.abs().max():.5f} mean={err.mean():.6f}")


def randn(S, ck, label):
    sh = (1, 1, S, 64)
    torch.manual_seed(42)
    Q, K, V = (torch.randn(sh, dtype=torch.bfloat16) for _ in range(3))
    ref = torch.softmax((Q.float() @ K.float().transpose(-2, -1)) / 8.0, -1) @ V.float()
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=ck)).float()
    err = out - ref
    print(f"PROBE randn S={S} {label}: rms/std={err.pow(2).mean().sqrt()/ref.std():.5f}")


for S in (128, 4096):
    vones(S, None, "fp32probs HiFi2")
    vones(S, ck3, "fp32probs HiFi3")
for S in (4096, 8192):
    randn(S, None, "fp32probs HiFi2")
    randn(S, ck3, "fp32probs HiFi3")
ttnn.close_device(device)
