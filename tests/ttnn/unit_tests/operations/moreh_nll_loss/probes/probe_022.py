import torch
import ttnn

device = ttnn.open_device(device_id=0)
fake = torch.ones([16], dtype=torch.float32)
tt_in = ttnn.from_torch(fake, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print(f"logical_shape={tt_in.shape}, padded_shape={tt_in.padded_shape}")
print(f"rank logical={len(tt_in.shape)}, rank padded={len(tt_in.padded_shape)}")
ttnn.close_device(device)
