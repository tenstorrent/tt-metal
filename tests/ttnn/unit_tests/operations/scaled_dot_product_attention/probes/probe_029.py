import math, torch, ttnn, numpy as np
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(0)
B, H, S, D = 1, 2, 3072, 64
Q = torch.randn(B, H, S, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)
scale = 1.0 / math.sqrt(D)
expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=False, scale=scale)
device = ttnn.open_device(device_id=0)
tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
res = ttnn.to_torch(out).to(torch.float32)
a = res.flatten().numpy()
e = expected.flatten().numpy()
pcc = np.corrcoef(a, e)[0, 1]
print(f"MCAST smoke: max_abs_diff={np.abs(a-e).max():.4f} PCC={pcc:.6f}")
ttnn.close_device(device)
