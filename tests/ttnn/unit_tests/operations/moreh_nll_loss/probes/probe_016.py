import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Small shape [5] with output_tensor provided — like nll_loss does
fake = torch.zeros([5], dtype=torch.float32)
fake[0] = 0.2
fake[1] = 0.7
fake[2] = 0.1
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"in shape={tt_in.shape}, padded={tt_in.padded_shape}")

# WITHOUT output_tensor
r1 = ttnn.operations.moreh.sum(tt_in)
print(f"no-output sum = {ttnn.to_torch(r1).reshape([-1])[:3]} (expected 1.0)")

# WITH output_tensor
out = ttnn.from_torch(
    torch.zeros([1], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
r2 = ttnn.operations.moreh.sum(tt_in, output=out)
print(f"with-output sum = {ttnn.to_torch(r2).reshape([-1])[:3]} (expected 1.0)")

ttnn.close_device(device)
