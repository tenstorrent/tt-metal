import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Vary S to isolate multi-block (Phase 10 alpha*O rescale) vs single-block.
# D=64 -> c_kv=4 -> Nkv = ceil(S/128). S=128 => Nkv=1 (no rescale, O=PV exact).
dev = ttnn.open_device(device_id=0)
D = 64
torch.manual_seed(7)
for S in (128, 256, 512):
    shape = (1, 1, S, D)
    Z = torch.zeros(shape, dtype=torch.bfloat16)
    V = (0.5 + torch.randint(0, 256, shape) * (2**-9)).to(torch.bfloat16)
    out = ttnn.to_torch(
        scaled_dot_product_attention(
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        )
    ).float()
    exact = V.double().mean(-2, keepdim=True).expand(shape)
    rne = exact.float().to(torch.bfloat16).float()
    nkv = -(-(S // 32) // 4)
    print(f"S={S} Nkv={nkv} flips={(out!=rne).float().mean()*100:.3f}% maxdiff={(out-rne).abs().max():.6f}")
ttnn.close_device(dev)
