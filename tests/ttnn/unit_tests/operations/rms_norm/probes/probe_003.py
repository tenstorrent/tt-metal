import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:
    x = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16)
    ti = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    c = ttnn.ComputeConfigDescriptor()
    c.fp32_dest_acc_en = True
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.math_approx_mode = False
    out = rms_norm(ti, epsilon=1e-6, compute_kernel_config=c)
    print("RAN OK", tuple(out.shape))
finally:
    ttnn.close_device(device)
