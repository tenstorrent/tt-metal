import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

print("split", pd.pick_w_split(256, 1, 130, 4096), pd.pick_w_split(160, 1, 130, 2048))
dev = ttnn.open_device(device_id=0) if False else None
