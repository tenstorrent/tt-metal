import torch, ttnn, sys
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
cfgf = lambda f: ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=f, math_approx_mode=False
)
for shape in [(1504, 544), (7168, 1024)]:
    for gam in [False, True]:
        for f in [True, False]:
            x = torch.randn(shape, dtype=torch.bfloat16)
            g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
            mc = auto_shard_config(
                list(shape),
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=dev,
            )
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) if gam else None
            out = ttnn.to_torch(rms_norm(tx, gamma=tg, memory_config=mc, compute_kernel_config=cfgf(f))).float()
            ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            if gam:
                ref = ref * g.float()
            err = (out - ref).abs().amax(-1)
            bad = (err > 0.1).nonzero().flatten()
            print("RES", shape, "gamma", gam, "fp32dest", f, "bad tile rows", sorted(set((bad // 32).tolist())))
ttnn.close_device(dev)
