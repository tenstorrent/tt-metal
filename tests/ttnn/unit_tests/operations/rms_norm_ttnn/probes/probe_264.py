"""Perf 3, Step 1: dump the solved plan for the focus shape (1,1,8192,1024) INT
and its two nearest perf neighbours."""
import os
os.environ["RMS_TRACE_BLOCKING"] = "1"
os.environ["RMS_PC_TRACE"] = "1"
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

dev = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 2304)]:
        W = shape[-1]
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = False
        cfg.math_approx_mode = False
        tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        gm = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        print(f"##### PLAN for {shape} #####", flush=True)
        out = rms_norm_ttnn(x, epsilon=1e-12, compute_kernel_config=cfg, weight=gm,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out); ttnn.deallocate(x); ttnn.deallocate(gm)
finally:
    ttnn.close_device(dev)
