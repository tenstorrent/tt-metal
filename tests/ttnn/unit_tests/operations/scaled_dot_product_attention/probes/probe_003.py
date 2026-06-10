import torch
import ttnn

torch.manual_seed(0)

# 1) D non-aligned: shape (1,1,32,50)
t_torch = torch.arange(32 * 50, dtype=torch.float32).reshape(1, 1, 32, 50).to(torch.bfloat16)
t_ttnn = ttnn.from_torch(t_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print("D non-aligned (1,1,32,50):")
print("  shape:", t_ttnn.shape)
print("  padded_shape:", t_ttnn.padded_shape if hasattr(t_ttnn, "padded_shape") else "n/a")
print("  buffer_aligned_page_size():", t_ttnn.buffer_aligned_page_size())
print("  tile_size(bf16):", ttnn.tile_size(ttnn.bfloat16))

# Round-trip
t_back = ttnn.to_torch(t_ttnn)
print("  round-trip shape:", t_back.shape)
print("  max diff:", (t_back.float() - t_torch.float()).abs().max().item())

# Convert to RM and see padded shape
t_rm = ttnn.to_layout(t_ttnn, ttnn.ROW_MAJOR_LAYOUT)
print("  after to_layout(RM), shape:", t_rm.shape)
t_back_rm = ttnn.to_torch(t_rm)
print("  RM round-trip shape:", t_back_rm.shape)

# 2) S_q non-aligned (1,1,47,64)
t2 = torch.arange(47 * 64, dtype=torch.float32).reshape(1, 1, 47, 64).to(torch.bfloat16)
t2_ttnn = ttnn.from_torch(t2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print("\nS non-aligned (1,1,47,64):")
print("  shape:", t2_ttnn.shape)
print("  padded_shape:", t2_ttnn.padded_shape if hasattr(t2_ttnn, "padded_shape") else "n/a")
print("  buffer_aligned_page_size():", t2_ttnn.buffer_aligned_page_size())
t2_back = ttnn.to_torch(t2_ttnn)
print("  round-trip shape:", t2_back.shape)
print("  max diff:", (t2_back.float() - t2.float()).abs().max().item())
