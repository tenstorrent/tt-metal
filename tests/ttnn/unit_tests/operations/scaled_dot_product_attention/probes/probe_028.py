import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
# Shape (1,1,128,256): S_q=128 (1 Q-block), S_kv=256 (2 KV-blocks), D=64
torch.manual_seed(42)
Q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)

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
print("Done")
ttnn.close_device(device)
