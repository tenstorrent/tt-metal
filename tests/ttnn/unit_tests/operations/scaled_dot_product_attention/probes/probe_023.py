import torch, ttnn

device = ttnn.open_device(device_id=0)

# Test bf8b D=256 fp32_dest_acc_en=False (reported as regression)
q_shape = (1, 1, 128, 256)
torch.manual_seed(42)
Q = torch.randn(q_shape, dtype=torch.bfloat16)
K = torch.randn(q_shape, dtype=torch.bfloat16)
V = torch.randn(q_shape, dtype=torch.bfloat16)

expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).to(torch.bfloat16)

ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

compute_kernel_config = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False,
    math_approx_mode=False,
)

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, compute_kernel_config=compute_kernel_config)
out = ttnn.to_torch(result)

has_nan = torch.isnan(out).any().item()
print(f"bf8b D=256 False: NaN={has_nan}")
if not has_nan:
    pcc = torch.corrcoef(torch.stack([out.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"PCC: {pcc}")

ttnn.close_device(device)
