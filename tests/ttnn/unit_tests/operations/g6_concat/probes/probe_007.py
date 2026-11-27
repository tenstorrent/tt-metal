import torch, ttnn
from ttnn.operations.g6_concat import g6_concat, g6_concat_plan

device = ttnn.open_device(device_id=0) if False else None
