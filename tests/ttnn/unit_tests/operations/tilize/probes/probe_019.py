import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    t = torch.arange(32 * 64).reshape(1, 1, 32, 64).to(torch.bfloat16)
    tt = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = _dispatch(tt, ttnn.DRAM_MEMORY_CONFIG, use_multicore=True, levers=dict(stateful_reads=1))
    ttnn.synchronize_device(dev)
finally:
    ttnn.close_device(dev)
