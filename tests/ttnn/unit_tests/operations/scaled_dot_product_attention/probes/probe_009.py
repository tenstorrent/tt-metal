import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(0)
B, H, S_q, D = 1, 1, 128, 128
S_kv = S_q

device = ttnn.open_device(device_id=0)

Q = torch.randn(B, H, S_q, D, dtype=torch.float32).to(torch.bfloat16)
K = torch.randn(B, H, S_kv, D, dtype=torch.float32).to(torch.bfloat16)
V = torch.randn(B, H, S_kv, D, dtype=torch.float32).to(torch.bfloat16)

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
print("Probe complete")
ttnn.close_device(device)
