import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Shape [64, 32] — 2 tile rows. Total = 2048 elements.
fake = torch.ones([64, 32], dtype=torch.float32)
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"shape={tt_in.shape}, padded={tt_in.padded_shape}")
r = ttnn.operations.moreh.sum(tt_in)
print(f"no-output sum = {ttnn.to_torch(r).reshape([-1])[:3]} (expected 2048)")
ttnn.close_device(device)
