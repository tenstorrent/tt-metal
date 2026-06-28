import torch
import ttnn
import math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Use reference.py's seed and shape
torch.manual_seed(0)
B, H, S_q, D = 1, 1, 128, 128
S_kv = S_q

Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

device = ttnn.open_device(device_id=0)
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
torch_output = ttnn.to_torch(output)
print("=== TTNN Output (first 4x4) ===")
print(torch_output[0, 0, :4, :4])

# Compare with reference
scale = 1.0 / math.sqrt(D)
expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), scale=scale)
print("\n=== Reference Output (first 4x4) ===")
print(expected[0, 0, :4, :4])

max_diff = (torch_output.float() - expected).abs().max().item()
print(f"\nMax diff: {max_diff:.2e}")
print(f"Pass: {max_diff < 1e-3}")

ttnn.close_device(device)
