import torch
import ttnn

torch.manual_seed(0)
shape = [5, 10]
C = shape[1]
target_shape = shape[:1] + shape[2:]

torch_target = torch.randint(0, C, target_shape, dtype=torch.long)
print(f"torch_target = {torch_target}")

device = ttnn.open_device(device_id=0)
tt_target = ttnn.from_torch(torch_target, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
print(f"tt_target shape={tt_target.shape}, padded_shape={tt_target.padded_shape}")
tt_target_back = ttnn.to_torch(tt_target, dtype=torch.int32)
print(f"tt_target_back shape={tt_target_back.shape}")
print(f"tt_target_back values (first 10):")
print(tt_target_back[:10] if tt_target_back.dim() > 0 else tt_target_back)

# Check if padded shape is actually 32x32 = 1024 elements
import numpy as np
import math

print(f"physical_volume = {tt_target.physical_volume()}")
ttnn.close_device(device)
