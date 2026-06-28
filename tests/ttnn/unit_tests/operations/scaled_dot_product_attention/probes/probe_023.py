import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
# Shape (1,1,256,64): num_q_blocks=2, num_kv_blocks=2
# Use random input — the actual test uses seed 42
torch.manual_seed(42)
Q = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)

scale = 1.0 / math.sqrt(64)
Qf, Kf, Vf = Q.float(), K.float(), V.float()
scores = (Qf @ Kf.transpose(-1, -2)) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

ttnn_Q = ttnn.from_torch(
    Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_K = ttnn.from_torch(
    K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_V = ttnn.from_torch(
    V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
result = ttnn.to_torch(output)

of = result.float().flatten()
ef = expected.float().flatten()
oc = of - of.mean()
ec = ef - ef.mean()
num = (oc * ec).sum()
den = torch.sqrt((oc**2).sum()) * torch.sqrt((ec**2).sum())
pcc = (num / den).item() if den > 0 else 1.0
print(f"PCC: {pcc:.6f}")
print(f"Max diff: {(of - ef).abs().max():.6f}")
# Check per Q-block
for qb in range(2):
    r1 = result[0, 0, qb * 128 : (qb + 1) * 128, :].float().flatten()
    e1 = expected[0, 0, qb * 128 : (qb + 1) * 128, :].float().flatten()
    r1c = r1 - r1.mean()
    e1c = e1 - e1.mean()
    n1 = (r1c * e1c).sum()
    d1 = torch.sqrt((r1c**2).sum()) * torch.sqrt((e1c**2).sum())
    p1 = (n1 / d1).item() if d1 > 0 else 1.0
    print(f"Q-block {qb} PCC: {p1:.6f}, max diff: {(r1-e1).abs().max():.6f}")

ttnn.close_device(device)
