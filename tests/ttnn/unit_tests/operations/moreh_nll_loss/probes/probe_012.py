import torch
import ttnn

torch.manual_seed(0)
# step1 output is shape [5] (padded to 32x32 tile)
fake = torch.zeros([5], dtype=torch.float32)
fake[0] = 0.2056
fake[1] = 0.7745
fake[2] = 0.1535
print(f"fake_input={fake}, fake.sum()={fake.sum().item():.4f}")

device = ttnn.open_device(device_id=0)
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"tt_in shape={tt_in.shape}, padded={tt_in.padded_shape}")
in_back = ttnn.to_torch(tt_in)
print(f"tt_in back={in_back}")

# Apply moreh_sum with no dim (full reduction)
out_t = torch.tensor([0.0], dtype=torch.float32)
tt_out = ttnn.from_torch(out_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
result = ttnn.operations.moreh.sum(tt_in, output=tt_out)
print(f"moreh.sum result shape={result.shape}, padded={result.padded_shape}")
print(f"moreh.sum vals = {ttnn.to_torch(result).reshape([-1])[:5]}")
ttnn.close_device(device)
