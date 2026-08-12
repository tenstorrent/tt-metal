import torch, ttnn
from ttnn.operations.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
