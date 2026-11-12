import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
x = torch.arange(64 * 512).reshape(1, 1, 64, 512).float() / 1000.0 + 1.0
tx = ttnn.from_torch(
    x,
    dtype=ttnn.float32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=shard_config([64, 512], (1, 1), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32),
)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
r = x / torch.sqrt(torch.mean(x * x, -1, keepdim=True) + 1e-6)
print("R out row0", o[0, 0, 0, :6].tolist(), o[0, 0, 0, 30:36].tolist())
print("R ref row0", r[0, 0, 0, :6].tolist(), r[0, 0, 0, 30:36].tolist())
print("R out row1", o[0, 0, 1, :4].tolist(), "ref", r[0, 0, 1, :4].tolist())
print("R ratio rows", (o / r)[0, 0, :, 0][:40:4].tolist())
ttnn.close_device(device)
