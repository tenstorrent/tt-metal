import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 64, 32
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
torch.manual_seed(0)
V = torch.rand(shape, dtype=torch.bfloat16)
out = ttnn.to_torch(
    scaled_dot_product_attention(
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    )
).float()
exact = V.float().mean(dim=-2, keepdim=True).expand(shape)
rne = exact.to(torch.bfloat16).float()
flips = (out != rne).float().mean().item()
diff = out - rne
print(
    f"RND flips={flips*100:.2f}% up={(diff>0).float().mean()*100:.2f}% down={(diff<0).float().mean()*100:.2f}% maxulp={(diff.abs()/0.00390625).max():.1f}"
)
ttnn.close_device(dev)
