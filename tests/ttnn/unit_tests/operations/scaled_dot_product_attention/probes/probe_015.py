import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 64, 32
shape = (1, 1, S, D)
Z = torch.zeros(shape, dtype=torch.bfloat16)

# V rows alternate a,b -> O = (a+b)/2 exactly per element.
ulp = 2.0**-9  # bf16 ulp at 0.5
for frac in (0.25, 0.5, 0.75):
    a = 0.5
    b = 0.5 + 2 * frac * ulp  # mean = 0.5 + frac*ulp
    V = torch.empty(shape, dtype=torch.bfloat16)
    V[..., 0::2, :] = a
    V[..., 1::2, :] = b
    mean = (a + b) / 2.0
    out = ttnn.to_torch(
        scaled_dot_product_attention(
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        )
    ).float()
    rne = torch.tensor(mean).to(torch.bfloat16).float().item()
    vals = out.unique()
    print(f"ROUND frac={frac}: mean_fp32={mean:.6f} RNE={rne:.6f} device_unique={vals.tolist()}")
ttnn.close_device(dev)
