import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi4
c.fp32_dest_acc_en = True
c.math_approx_mode = False
shape = (1, 1, 224, 3072)
x = torch.randn(*shape, dtype=torch.bfloat16)
g = torch.randn(shape[-1], dtype=torch.bfloat16)
mc = auto_shard_config(list(shape), ML.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
print("shard", list(mc.shard_spec.shape), mc.shard_spec.grid.num_cores())
xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
gd = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
try:
    out = rms_norm(xd, gamma=gd, compute_kernel_config=c, memory_config=xd.memory_config())
    print("OK")
except Exception as ex:
    print("ERR", str(ex)[:2000])
ttnn.close_device(dev)
