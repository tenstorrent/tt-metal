import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
device = ttnn.open_device(device_id=0) if False else None
