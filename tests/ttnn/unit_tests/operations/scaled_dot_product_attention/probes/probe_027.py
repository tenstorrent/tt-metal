import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Q=K=0 => P=1.0, l=S=512. recip(512) should be exactly 2^-9 = 0x3b000000.
# DPRINT will show INV bits. If != 0x3b000000, SFPU recip is approximate (H2).
dev = ttnn.open_device(device_id=0)
S, D = 512, 64
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
# constant V => mean == V exactly; any flip is purely recip/normalize error
V = torch.full(shape, 0.75, dtype=torch.bfloat16)
out = ttnn.to_torch(
    scaled_dot_product_attention(
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    )
).float()
print("const V=0.75 out unique:", out.unique().tolist())
print("flips vs 0.75:", (out != 0.75).float().mean().item() * 100, "%")
ttnn.close_device(dev)
