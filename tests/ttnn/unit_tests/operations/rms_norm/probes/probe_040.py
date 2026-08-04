import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout


def cfg(acc=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


shape = (1, 1, 64, 512)
torch.manual_seed(0)
x = torch.ones(*shape, dtype=torch.bfloat16)
g = (torch.arange(512, dtype=torch.float32) + 1).to(torch.bfloat16)  # gamma = w+1, so we can read positions
mc = auto_shard_config(list(shape), _ML.WIDTH_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev)
xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
gd = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
out = ttnn.to_torch(rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=xd.memory_config())).float()
print("RES got row0[0:40] =", out[0, 0, 0, :40].tolist())
print("RES exp row0[0:40] =", g.float()[:40].tolist())
print("RES got row0[248:280] =", out[0, 0, 0, 248:280].tolist())
ttnn.close_device(dev)
