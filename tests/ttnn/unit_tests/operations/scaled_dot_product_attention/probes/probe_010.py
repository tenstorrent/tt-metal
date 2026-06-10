import torch, math, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
ck3 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
)


def run(Q, K, V, label, dtype=ttnn.bfloat16, ck=ck3):
    D = Q.shape[-1]
    ref = torch.softmax((Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(D), -1) @ V.float()
    tq = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=ck)).float()
    err = out - ref
    print(f"{label}: rms/std={err.pow(2).mean().sqrt()/ref.std():.5f} max={err.abs().max():.5f}")


torch.manual_seed(42)
for S in (4096, 8192):
    sh = (1, 1, S, 64)
    Q, K, V = (torch.randn(sh, dtype=torch.bfloat16) for _ in range(3))
    run(Q, K, V, f"randn S={S} bf16 HiFi3")
    run(Q, K, V, f"randn S={S} bf8b HiFi3", dtype=ttnn.bfloat8_b)
sh = (1, 4, 4096, 64)
Q, K, V = (torch.randn(sh, dtype=torch.bfloat16) for _ in range(3))
run(Q, K, V, "randn 1x4x4096x64 bf16 HiFi3")
# uniform / negative S=512 regressions
sh = (1, 12, 512, 64)
torch.manual_seed(42)
Q, K, V = (torch.rand(sh, dtype=torch.bfloat16) for _ in range(3))
run(Q, K, V, "uniform S=512 HiFi3")
torch.manual_seed(42)
Q, K, V = (-(torch.rand(sh, dtype=torch.bfloat16) + 0.5) for _ in range(3))
run(Q, K, V, "negative S=512 HiFi3")
# V=ones consistency at HiFi3
sh = (1, 1, 4096, 64)
torch.manual_seed(0)
run(
    torch.randn(sh, dtype=torch.bfloat16),
    torch.randn(sh, dtype=torch.bfloat16),
    torch.ones(sh, dtype=torch.bfloat16),
    "V=1 HiFi3 (abs err vs exact 1)",
)
ttnn.close_device(device)
