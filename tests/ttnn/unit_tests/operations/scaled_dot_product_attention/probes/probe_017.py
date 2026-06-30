import torch
import ttnn
import math

device = ttnn.open_device(device_id=0)

# Falcon-7B shape: B=1, H_q=71, H_kv=1 (MQA), S=2048, D=64
B, H_q, H_kv, S, D = 1, 71, 1, 2048, 64
torch.manual_seed(1234)
Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16) * 0.1
K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1
V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16) * 0.1

tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
)

print("Running SDPA with Falcon-7B shape...")
tt_out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
    tt_Q, tt_K, tt_V, is_causal=False, attn_mask=None, compute_kernel_config=cfg
)
tt_back = ttnn.to_torch(tt_out)
print(f"Output shape: {tt_back.shape}")
print("SUCCESS - no hang")

# Compute reference
import torch.nn.functional as F

gt = F.scaled_dot_product_attention(Q, K.expand(-1, H_q, -1, -1), V.expand(-1, H_q, -1, -1))
diff = (gt - tt_back).abs()
print(f"Max diff: {diff.max().item()}")
print(f"Mean diff: {diff.mean().item()}")

ttnn.close_device(device)
