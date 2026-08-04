import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0) if False else None
