import torch, ttnn, traceback
import ttnn.operations.rms_norm as M

M.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    # bf8b input, non-aligned W, no gamma, TILE
    shape = (128, 100)
    x = torch.randn(shape)
    ti = ttnn.from_torch(
        x.bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    try:
        o = rms_norm(ti, compute_kernel_config=cfg)
        out = ttnn.to_torch(o)
        print(f"bf8b nonalign no-gamma OK shape={out.shape}")
    except Exception as e:
        traceback.print_exc()
finally:
    ttnn.close_device(dev)
