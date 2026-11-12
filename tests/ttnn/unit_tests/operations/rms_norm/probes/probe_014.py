import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
for dt in (ttnn.float32, ttnn.bfloat16):
    for shape in [(1, 1, 64, 512), (1, 1, 32 * 240, 512), (1, 1, 32 * 240, 4096)]:
        torch.manual_seed(0)
        x = torch.randn(shape)
        tx = ttnn.from_torch(
            x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
        r = x / torch.sqrt(torch.mean(x * x, -1, keepdim=True) + 1e-6)
        print("R", dt, shape, "maxerr", (o - r).abs().max().item())
ttnn.close_device(device)
