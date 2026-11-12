import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
    for dt in (ttnn.float32,):
        shape = (1, 1, 32 * 240, 512)
        torch.manual_seed(0)
        x = torch.randn(shape)
        tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
        r = x / torch.sqrt(torch.mean(x * x, -1, keepdim=True) + 1e-6)
        e = (o - r).abs()[0, 0]
        te = e.reshape(240, 32, 512).amax((1, 2))
        print("R", layout, "maxerr", e.max().item(), "tile-row err (first 12)", [round(v, 3) for v in te[:12].tolist()])
        # within tile-row 1, per-row
        print("R rows in tr1", [round(v, 3) for v in e[32:64].amax(-1).tolist()])
        print("R x/o", x[0, 0, 33, :4].tolist(), o[0, 0, 33, :4].tolist(), r[0, 0, 33, :4].tolist())
ttnn.close_device(device)
