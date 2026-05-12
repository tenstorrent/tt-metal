import torch
import ttnn

torch.manual_seed(0)
# step1 output is shape [5] (target_shape with C dropped), padded to (32, 32) tile internally
# Create a fake step1 output: shape [5] with sparse weight values
fake = torch.zeros([5], dtype=torch.float32)
fake[0] = 0.2056  # weight[target[0]=0]
fake[1] = 0.7745
fake[2] = 0.1535
# fake[3]=0 (ignore), fake[4]=0 (ignore)
print(f"fake_input={fake}, fake.sum()={fake.sum().item():.4f}")

device = ttnn.open_device(device_id=0)
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"tt_in shape={tt_in.shape}, padded={tt_in.padded_shape}")
print(f"tt_in vals = {ttnn.to_torch(tt_in)}")

# Now apply moreh_sum
out_t = torch.tensor([0.0], dtype=torch.float32)
tt_out = ttnn.from_torch(out_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
result = ttnn.operations.moreh.moreh_sum(tt_in, output_tensor=tt_out)
print(f"moreh_sum result shape={result.shape}, padded={result.padded_shape}")
print(f"moreh_sum vals = {ttnn.to_torch(result).reshape([-1])[:5]}")
ttnn.close_device(device)
