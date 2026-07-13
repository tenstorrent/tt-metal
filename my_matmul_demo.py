"""Tiny demo: call the custom ttnn.my_matmul op on-device."""
import torch
import ttnn

# Two tile-aligned matrices: A [64, 96] @ B [96, 128] -> C [64, 128].
a = torch.randn(64, 96, dtype=torch.bfloat16)
b = torch.randn(96, 128, dtype=torch.bfloat16)
# a = torch.ones(1, 96, dtype=torch.bfloat16)
# b = torch.ones(96, 1, dtype=torch.bfloat16)

device = ttnn.open_device(device_id=0)
try:
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    tc = ttnn.my_matmul(ta, tb)

    print("A:", tuple(a.shape), " B:", tuple(b.shape), " -> C:", tuple(ttnn.to_torch(tc).shape))
    print(ttnn.to_torch(tc))
finally:
    ttnn.close_device(device)
