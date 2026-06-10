import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 64, 32
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
ulp = 0.00390625


def run(V):
    return (
        ttnn.to_torch(
            scaled_dot_product_attention(
                ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            )
        )
        .float()
        .unique()
        .tolist()
    )


V = torch.full(shape, 0.5, dtype=torch.bfloat16)
V[..., 0::4, :] = 0.5 + 3 * ulp
print("Q75 frac=.75 RNE->0.50390625 TRUNC->0.5 :", run(V))
V = torch.full(shape, 0.5, dtype=torch.bfloat16)
V[..., 0::4, :] = 0.5 + ulp
print("Q25 frac=.25 RNE->0.5 TRUNC->0.5        :", run(V))
V = torch.full(shape, 0.5, dtype=torch.bfloat16)
V[..., 0::2, :] = 0.5 + ulp
print("Q50 frac=.50 RNE->0.5(even) TRUNC->0.5  :", run(V))
ttnn.close_device(dev)
