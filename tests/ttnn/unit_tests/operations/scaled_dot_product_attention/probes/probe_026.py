import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 512, 64
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)
torch.manual_seed(42)


def run(V):
    return ttnn.to_torch(
        scaled_dot_product_attention(
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        )
    ).float()


# coarse: V in {0, .25, .5, .75} -> all sums exact in fp32, mean exact in bf16
Vc = (torch.randint(0, 4, shape) * 0.25).to(torch.bfloat16)
out = run(Vc)
exact = Vc.double().mean(-2, keepdim=True).expand(shape).float()
print(f"COARSE flips={(out != exact.to(torch.bfloat16).float()).float().mean()*100:.3f}%")
# fine: V on 2^-9 grid in [0.5, 1): sums exact in fp32, mean has frac at 2^-9 grid
Vf = (0.5 + torch.randint(0, 256, shape) * (2**-9)).to(torch.bfloat16)
out = run(Vf)
exact = Vf.double().mean(-2, keepdim=True).expand(shape)
rne = exact.float().to(torch.bfloat16).float()
print(f"FINE flips={(out != rne).float().mean()*100:.3f}% maxdiff={(out-rne).abs().max():.6f}")
ttnn.close_device(dev)
