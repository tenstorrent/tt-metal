import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Shape [64, 32] — 2 tile rows. Total = 2048 elements.
fake = torch.ones([64, 32], dtype=torch.float32)
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"shape={tt_in.shape}, padded={tt_in.padded_shape}")
out = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r = ttnn.operations.moreh.sum(tt_in, output=out)
print(f"2-tile sum = {ttnn.to_torch(r).reshape([-1])[:3]} (expected 2048)")

# Shape [32, 32] — 1 tile, all ones
fake1 = torch.ones([32, 32], dtype=torch.float32)
tt_in1 = ttnn.from_torch(fake1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out1 = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r1 = ttnn.operations.moreh.sum(tt_in1, output=out1)
print(f"1-tile sum = {ttnn.to_torch(r1).reshape([-1])[:3]} (expected 1024)")

# Shape [16, 16] — fractional tile
fake2 = torch.ones([16, 16], dtype=torch.float32)
tt_in2 = ttnn.from_torch(fake2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out2 = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r2 = ttnn.operations.moreh.sum(tt_in2, output=out2)
print(f"frac-tile sum = {ttnn.to_torch(r2).reshape([-1])[:3]} (expected 256)")

ttnn.close_device(device)
