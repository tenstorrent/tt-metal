import torch, math, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)


def run(Q, K, V, label, dtype=ttnn.bfloat16):
    D = Q.shape[-1]
    Kf, Vf = K.float(), V.float()
    if Q.shape[1] != K.shape[1]:
        r = Q.shape[1] // K.shape[1]
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    ref = torch.softmax((Q.float() @ Kf.transpose(-2, -1)) / math.sqrt(D), -1) @ Vf
    tq = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()
    err = out - ref
    print(f"PROBE {label}: rms/std={err.pow(2).mean().sqrt()/ref.std():.5f} max={err.abs().max():.5f}")


torch.manual_seed(42)
for sh in [(1, 1, 32, 32), (1, 4, 128, 64), (1, 8, 256, 64), (2, 4, 128, 64), (1, 12, 512, 64)]:
    Q, K, V = (torch.rand(sh, dtype=torch.bfloat16) for _ in range(3))
    run(Q, K, V, f"uniform {sh}")
torch.manual_seed(42)
for sh in [(1, 12, 512, 64), (1, 8, 256, 64)]:
    Q, K, V = (-(torch.rand(sh, dtype=torch.bfloat16) + 0.5) for _ in range(3))
    run(Q, K, V, f"negative {sh}")
torch.manual_seed(42)
Q = torch.randn(1, 4, 4096, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16)
run(Q, K, V, "mqa Q1x4x4096x64")
Q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16)
K = torch.randn(1, 2, 4096, 128, dtype=torch.bfloat16)
V = torch.randn(1, 2, 4096, 128, dtype=torch.bfloat16)
run(Q, K, V, "gqa Q1x8x4096x128")
Q, K, V = (torch.randn(1, 1, 8192, 64, dtype=torch.bfloat16) for _ in range(3))
run(Q, K, V, "bf8b 8192", dtype=ttnn.bfloat8_b)
run(Q, K, V, "bf16 8192")
ttnn.close_device(device)
