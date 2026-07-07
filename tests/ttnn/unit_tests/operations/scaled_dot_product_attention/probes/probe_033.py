import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(42)
B, H, S, D = 1, 1, 128, 64
q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

# PyTorch reference with is_causal=True
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

# TTNN with is_causal=True
q_t = ttnn.from_torch(
    q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
k_t = ttnn.from_torch(
    k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
v_t = ttnn.from_torch(
    v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
output_torch = ttnn.to_torch(output)

# Check PCC
from tests.ttnn.utils_for_testing import assert_with_pcc

assert_with_pcc(ref, output_torch, pcc=0.995)
print("PASS: causal mask 128x64")
