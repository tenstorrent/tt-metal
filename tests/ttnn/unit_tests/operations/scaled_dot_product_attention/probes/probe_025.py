import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 512, 64
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
torch.manual_seed(42)
V = torch.rand(shape, dtype=torch.bfloat16)
out = ttnn.to_torch(
    scaled_dot_product_attention(
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    )
).float()
exact = V.double().mean(dim=-2, keepdim=True).expand(shape)  # exact mean
rne = exact.float().to(torch.bfloat16).float()
flips = (out != rne).float().mean().item()
prepack_err = None
print(
    f"B512 flips={flips*100:.2f}% maxdiff={(out-rne).abs().max():.6f} exact_frac_std={(exact - exact.float().to(torch.bfloat16).double()).abs().div(0.00390625).mean():.3f}"
)
ttnn.close_device(dev)
