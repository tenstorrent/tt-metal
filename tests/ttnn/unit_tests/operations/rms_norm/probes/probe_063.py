import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc
from ttnn.operations.rms_norm import rms_norm

desc._FORCE_REGIME = "B"
desc._FORCE_TRANSPORT = 2
desc._FORCE_K = 8
device = ttnn.open_device(device_id=0)
try:
    to = ttnn.from_torch(
        torch.ones(1, 1, 32, 2048).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn.to_torch(rms_norm(to))
finally:
    ttnn.close_device(device)
