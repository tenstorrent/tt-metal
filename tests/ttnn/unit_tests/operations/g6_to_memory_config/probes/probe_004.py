import torch, ttnn

device = ttnn.open_device(device_id=0) if False else None
