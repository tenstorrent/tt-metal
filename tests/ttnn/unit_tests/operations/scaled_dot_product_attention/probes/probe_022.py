import torch, ttnn
import numpy as np
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
exact = V.float().mean(dim=-2, keepdim=True).expand(shape).contiguous()
bits = exact.view(torch.int32)
frac = (bits & 0xFFFF).float() / 65536.0
up = out > exact.to(torch.bfloat16).float()
# round-up threshold: min frac that flipped up, max frac that didn't
print("FRAC flipped-up fracs: min={:.4f}".format(frac[up].min().item() if up.any() else -1))
notup = ~up
print("FRAC not-flipped fracs: max={:.4f}".format(frac[notup & (frac < 0.5)].max().item()))
ttnn.close_device(dev)
