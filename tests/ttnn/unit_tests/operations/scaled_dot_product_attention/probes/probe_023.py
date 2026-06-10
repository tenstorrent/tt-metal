import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 64, 32
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
ulp = 0.00390625
V = torch.full(shape, 0.5, dtype=torch.bfloat16)
V[..., 0::4, :] = 0.5 + 3 * ulp  # 16 of 64 rows -> mean = 0.5 + 0.75*ulp = 0.50292969
out = ttnn.to_torch(
    scaled_dot_product_attention(
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    )
).float()
print("Q75 mean=0.50292969 RNE->0.50390625 TRUNC->0.5 device:", out.unique().tolist())
ttnn.close_device(dev)
