import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
# Shape (1,1,128,256): S_q=128 (1 Q-block), S_kv=256 (2 KV-blocks), D=64
torch.manual_seed(42)
Q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
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
print(f"PCC for (1,1,128,64)x(1,1,256,64) q=1,kv=2: {pcc:.6f}")
print(f"Max diff: {(of - ef).abs().max():.6f}")

ttnn.close_device(device)
