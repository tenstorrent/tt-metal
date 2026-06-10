import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# S=128 single block. Print O0/L0/INV via DPRINT; compare out vs models.
dev = ttnn.open_device(device_id=0)
S, D = 128, 64
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
torch.manual_seed(7)
V = (0.5 + torch.randint(0, 256, shape) * (2**-9)).to(torch.bfloat16)
out = ttnn.to_torch(
    scaled_dot_product_attention(
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    )
).float()
O = V.double().sum(-2)  # exact O, (1,1,D)
print("exact O[:4]   =", [round(x, 6) for x in O.flatten()[:4].tolist()])
print("exact mean[:4]=", [round(x, 9) for x in (O / S).flatten()[:4].tolist()])
print("device out[:4]=", out[0, 0, 0, :4].tolist())
rne = (O / S).float().to(torch.bfloat16).float()
print("rne mean[:4]  =", rne.flatten()[:4].tolist())
print("flips=%.3f%%" % ((out[0, 0, 0, :] != rne.flatten()).float().mean() * 100))
ttnn.close_device(dev)
