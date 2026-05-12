import torch
import ttnn

device = ttnn.open_device(device_id=0)

# All zeros — should sum to 0
fake_zeros = torch.zeros([5], dtype=torch.float32)
tt_zero = ttnn.from_torch(fake_zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out_zero = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r_zero = ttnn.operations.moreh.sum(tt_zero, output=out_zero)
print(f"zero sum={ttnn.to_torch(r_zero).reshape([-1])[:3]}")

# Single value 1.0
fake_one = torch.zeros([5], dtype=torch.float32)
fake_one[0] = 1.0
tt_one = ttnn.from_torch(fake_one, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out_one = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r_one = ttnn.operations.moreh.sum(tt_one, output=out_one)
print(f"one-element sum={ttnn.to_torch(r_one).reshape([-1])[:3]}")

# All 1.0 in valid positions; padded should be 0
fake_full = torch.ones([5], dtype=torch.float32)
tt_full = ttnn.from_torch(fake_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out_full = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r_full = ttnn.operations.moreh.sum(tt_full, output=out_full)
print(f"all-ones sum={ttnn.to_torch(r_full).reshape([-1])[:3]}")

# Larger shape
fake_big = torch.ones([100], dtype=torch.float32)
tt_big = ttnn.from_torch(fake_big, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out_big = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r_big = ttnn.operations.moreh.sum(tt_big, output=out_big)
print(f"100-ones sum={ttnn.to_torch(r_big).reshape([-1])[:3]}")
ttnn.close_device(device)
