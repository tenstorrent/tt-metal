import torch, ttnn
from ttnn.operations.add import add
from eval.memory import height_shards
device = ttnn.open_device(device_id=0) if False else None
