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
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg))
ttnn.synchronize_device(device)
x = x[0, 0]
st = (x * x).reshape(2, 32, 16, 32).sum(2)  # tr,row,col
print("H expected sumsq tile0 rows0-7 col0-1", st[0, :8, :2].tolist())
print("H expected sumsq tile1 rows0-7 col0-1", st[1, :8, :2].tolist())
print("H expected rstd tr1 rows0-7", (1 / torch.sqrt((x * x).mean(-1) + 1e-6))[32:40].tolist())
ttnn.close_device(device)
