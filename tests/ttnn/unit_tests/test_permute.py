import ttnn
import torch

print("Creating tensor and device")
device = ttnn.CreateDevice(0, l1_small_size=8192)
act = torch.randn((1024, 3, 32, 32), dtype=torch.bfloat16)

print("Moving tensor to device")
tt_act = ttnn.from_torch(act, ttnn.bfloat16)
tt_act_device = ttnn.to_device(tt_act, device)

print("permuting tensor")
tt_perm_device = ttnn.permute(tt_act_device, (2, 3, 1, 0))

print("Moving permuted tensor to host")
tt_perm = ttnn.from_device(tt_perm_device)
