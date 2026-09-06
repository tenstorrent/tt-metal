import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


shape = (1, 1, 8192, 1024)
W = shape[-1]
device = ttnn.open_device(device_id=0)
try:
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = ttnn.from_torch(
        torch.randn(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for _ in range(2):
        o = rms_norm_ttnn(x, epsilon=1e-12, weight=g, compute_kernel_config=cfg(), memory_config=x.memory_config())
        ttnn.deallocate(o)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
finally:
    ttnn.close_device(device)
print("DONE")
