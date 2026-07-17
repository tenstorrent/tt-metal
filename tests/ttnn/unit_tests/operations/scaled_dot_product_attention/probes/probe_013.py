import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = device
# All-ones -> output should be exactly 1.0 everywhere (uniform softmax, V=ones).
B, H, S, D = 1, 1, 64, 64
Q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
K = torch.ones(B, H, S, D, dtype=torch.bfloat16)
V = torch.ones(B, H, S, D, dtype=torch.bfloat16)
scale = 1.0 / math.sqrt(D)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
r = ttnn.to_torch(out).float()
print("shape", r.shape, "min", r.min().item(), "max", r.max().item(), "mean", r.mean().item())
print("all close to 1.0:", torch.allclose(r, torch.ones_like(r), atol=0.02))
