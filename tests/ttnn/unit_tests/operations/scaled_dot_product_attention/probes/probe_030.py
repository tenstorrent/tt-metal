import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Fine-grid: DPRINT PV per block to check matmul exactness vs hand sum.
dev = ttnn.open_device(device_id=0)
S, D = 512, 64
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
torch.manual_seed(1)
V = (0.5 + torch.randint(0, 256, shape) * (2**-9)).to(torch.bfloat16)
out = ttnn.to_torch(
    scaled_dot_product_attention(
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
    )
).float()
# per-block exact PV col0/col1 (128 keys each block). DPRINT shows PV[0,0] PV[0,1]
for kb in range(4):
    blk = V[..., kb * 128 : (kb + 1) * 128, :].double()
    s = blk.sum(-2)  # (1,1,D)
    print(f"exact PV kb{kb} col0={s.flatten()[0].item():.6f} col1={s.flatten()[1].item():.6f}")
exact = V.double().mean(-2, keepdim=True).expand(shape)
rne = exact.float().to(torch.bfloat16).float()
print("FINE flips=%.3f%% maxdiff=%.6f" % ((out != rne).float().mean() * 100, (out - rne).abs().max()))
print("out[0,0,0,:4]=", out[0, 0, 0, :4].tolist())
print("rne[0,0,0,:4]=", rne[0, 0, 0, :4].tolist())
ttnn.close_device(dev)
