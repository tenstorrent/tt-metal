import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Shape [32]: tile-aligned 1D
fake = torch.zeros([32], dtype=torch.float32)
fake[0:3] = torch.tensor([0.2056, 0.7745, 0.1535])
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"shape={tt_in.shape}, padded={tt_in.padded_shape}")
out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
print(f"sum [32]={ttnn.to_torch(r).reshape([-1])[:3]}")

# Shape [16]: half-tile
fake = torch.zeros([16], dtype=torch.float32)
fake[0:3] = torch.tensor([0.2056, 0.7745, 0.1535])
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"shape={tt_in.shape}, padded={tt_in.padded_shape}")
out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
print(f"sum [16]={ttnn.to_torch(r).reshape([-1])[:3]}")

# Shape [5, 32]: 2D, full row
fake = torch.zeros([5, 32], dtype=torch.float32)
fake[0, 0:3] = torch.tensor([0.2056, 0.7745, 0.1535])
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"shape={tt_in.shape}, padded={tt_in.padded_shape}")
out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=out)
print(f"sum [5,32]={ttnn.to_torch(r).reshape([-1])[:3]}")

ttnn.close_device(device)
