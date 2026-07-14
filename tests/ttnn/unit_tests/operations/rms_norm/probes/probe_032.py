import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

torch.manual_seed(0)
shape = (1, 1, 32, 128)  # 4 W-tiles -> 4 cores, root=(3,0)
x = torch.randn(shape, dtype=torch.bfloat16)
g = torch.ones(shape[-1], dtype=torch.bfloat16)
dev = ttnn.open_device(device_id=0)
try:
    mc = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev
    )
    print("shard grid:", mc.shard_spec.grid, "shape:", mc.shard_spec.shape)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gt = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    res = ttnn.to_torch(out).float()
    exp = (x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)) * g.float().reshape(-1)
    m = torch.mean(x.float() ** 2, dim=-1)
    print("torch mean(x^2) row0:", m[0, 0, 0].item(), "1/rms row0:", (1 / torch.sqrt(m + 1e-6))[0, 0, 0].item())
    diff = (res - exp).abs()
    print(
        f"PCC/relRMS: {torch.corrcoef(torch.stack([res.flatten(),exp.flatten()]))[0,1].item():.4f} {(diff.pow(2).mean().sqrt()/exp.std()).item():.4f}"
    )
finally:
    ttnn.close_device(dev)
