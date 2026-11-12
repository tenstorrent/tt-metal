import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
torch.manual_seed(0)
x = torch.randn(1, 1, 64, 512)
mc = shard_config(
    [64, 512], (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32
)
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()[0, 0]
x = x[0, 0]
r = x / torch.sqrt(torch.mean(x * x, -1, keepdim=True) + 1e-6)
e = (o - r).abs()
print("R per tile err", [[round(v, 3) for v in row] for row in e.reshape(2, 32, 16, 32).amax((1, 3)).tolist()])
implied = o / x
print(
    "R row33 implied rstd per tile",
    [round(v, 4) for v in implied[33].reshape(16, 32).median(-1).values.tolist()],
    "true",
    (1 / torch.sqrt(torch.mean(x[33] ** 2) + 1e-6)).item(),
)
print(
    "R row1 implied rstd per tile",
    [round(v, 4) for v in implied[1].reshape(16, 32).median(-1).values.tolist()],
    "true",
    (1 / torch.sqrt(torch.mean(x[1] ** 2) + 1e-6)).item(),
)
ttnn.close_device(device)
