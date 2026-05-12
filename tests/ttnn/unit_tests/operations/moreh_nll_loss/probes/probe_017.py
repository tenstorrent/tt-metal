import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Reproduce exact moreh_nll_loss invocation:
# moreh_sum(step1_result, std::nullopt, false, divisor_tensor, mem_cfg, ckc_val)
# In Python: ttnn.operations.moreh.sum(input, dim=None, keepdim=False, output=divisor)

fake = torch.zeros([5], dtype=torch.float32)
fake[0] = 0.2056
fake[1] = 0.7745
fake[2] = 0.1535
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"in shape={tt_in.shape}, padded={tt_in.padded_shape}")

# Mimic the divisor tensor — initially [0] tensor
divisor = ttnn.from_torch(
    torch.tensor([0.0], dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
print(f"divisor.shape={divisor.shape}, padded={divisor.padded_shape}")

r = ttnn.operations.moreh.sum(tt_in, dim=None, keepdim=False, output=divisor)
print(f"sum result = {ttnn.to_torch(r).reshape([-1])[:3]} (expected 1.1336)")
print(f"divisor after = {ttnn.to_torch(divisor).reshape([-1])[:3]}")

ttnn.close_device(device)
