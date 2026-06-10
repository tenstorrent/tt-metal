import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 64, 32
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)


def mk(a, b):
    V = torch.empty(shape, dtype=torch.bfloat16)
    V[..., 0::2, :] = a
    V[..., 1::2, :] = b
    return V


def run(V):
    return ttnn.to_torch(
        scaled_dot_product_attention(
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        )
    ).float()


print("OUT +1ulp  (expect 0.501953125):", run(mk(0.5, 0.50390625)).unique().tolist())
print("MID tie    (expect 0.5, RNE-even):", run(mk(0.5, 0.501953125)).unique().tolist())
print("OUT neg    (expect -0.501953125):", run(mk(-0.5, -0.50390625)).unique().tolist())
ttnn.close_device(dev)
