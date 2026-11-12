import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
shape = (1, 1, 32 * 240, 512)
torch.manual_seed(0)
x = torch.randn(shape)
tx = ttnn.from_torch(
    x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()[0, 0]
x = x[0, 0]
rs = 1 / torch.sqrt(torch.mean(x * x, -1) + 1e-6)
implied = (o / x).median(-1).values
print("R implied tr1", [round(v, 4) for v in implied[32:40].tolist()])
print("R true tr1", [round(v, 4) for v in rs[32:40].tolist()])
print("R true tr0", [round(v, 4) for v in rs[0:8].tolist()])
ss = (x * x).sum(-1)
print("R implied sumsq tr1", [round(v, 2) for v in (512 / implied[32:40] ** 2).tolist()])
print("R true sumsq tr1", [round(v, 2) for v in ss[32:40].tolist()], "tr0", [round(v, 2) for v in ss[0:8].tolist()])
ttnn.close_device(device)
