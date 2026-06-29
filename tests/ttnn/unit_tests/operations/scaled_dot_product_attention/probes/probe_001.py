import torch, ttnn, math

torch.manual_seed(42)
Q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
K = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
V = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
scale = 1.0 / math.sqrt(32)

expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), scale=scale)
print(f"Expected (first 8): {expected.flatten()[:8]}")
print(f"Expected stats: min={expected.min():.4f} max={expected.max():.4f} mean={expected.mean():.4f}")

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=ttnn.devices[0])
tK = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=ttnn.devices[0])
tV = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=ttnn.devices[0])

# bf8b + fp32_dest_acc_en=True (this works)
config_fp32 = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
)
out_fp32 = ttnn.to_torch(scaled_dot_product_attention(tQ, tK, tV, scale=scale, compute_kernel_config=config_fp32))
print(f"\nbf8b + fp32_acc: first 8: {out_fp32.flatten()[:8]}")
print(f"bf8b + fp32_acc stats: min={out_fp32.min():.4f} max={out_fp32.max():.4f} mean={out_fp32.mean():.4f}")

# bf8b + fp32_dest_acc_en=False (this fails with PCC=0.0)
config_bf16 = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
)
out_bf16 = ttnn.to_torch(scaled_dot_product_attention(tQ, tK, tV, scale=scale, compute_kernel_config=config_bf16))
print(f"\nbf8b + bf16_acc: first 8: {out_bf16.flatten()[:8]}")
print(f"bf8b + bf16_acc stats: min={out_bf16.min():.4f} max={out_bf16.max():.4f} mean={out_bf16.mean():.4f}")

# Check for NaN/Inf
has_nan = torch.isnan(out_bf16).any().item()
has_inf = torch.isinf(out_bf16).any().item()
print(f"\nbf8b + bf16_acc has NaN: {has_nan}, has Inf: {has_inf}")
if has_nan or has_inf:
    print(f"  NaN count: {torch.isnan(out_bf16).sum().item()}")
    print(f"  Inf count: {torch.isinf(out_bf16).sum().item()}")

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
