import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

torch.manual_seed(0)
# K28_HT16_512x224: rows=512, W=224, sh=512, sw=32, gx=7, gy=4
rows, W, sh, sw, gx, gy = 512, 224, 512, 32, 7, 4
shape = (1, 1, rows, W)
ti = torch.randn(shape, dtype=torch.bfloat16)
tg = torch.randn(W, dtype=torch.bfloat16)
exp = ti.float() * torch.rsqrt(ti.float().pow(2).mean(-1, keepdim=True) + 1e-6) * tg.float().reshape(-1)
_ML = ttnn.TensorMemoryLayout
ml = _ML.BLOCK_SHARDED if gy > 1 and sh > 32 else _ML.WIDTH_SHARDED
print("memory_layout:", ml)
cfg = shard_config([sh, sw], (gx, gy), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
print("cfg:", cfg)
xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
gt = ttnn.from_torch(
    tg.reshape(1, 1, 1, W),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
cc = ttnn.ComputeConfigDescriptor()
cc.math_fidelity = ttnn.MathFidelity.HiFi2
cc.fp32_dest_acc_en = False
cc.math_approx_mode = False
out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=cfg)
res = ttnn.to_torch(out)
print(
    "out shape:",
    res.shape,
    "has_nan:",
    torch.isnan(res.float()).any().item(),
    "n_nan:",
    torch.isnan(res.float()).sum().item(),
)
a = res.flatten().float()
b = exp.flatten().float()
a = a - a.mean()
b = b - b.mean()
d = (a.norm() * b.norm()).item()
pcc = 1.0 if d == 0 else torch.dot(a, b).item() / d
print("PCC:", pcc)
