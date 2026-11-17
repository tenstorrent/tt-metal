import torch, ttnn, sys
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
for ml in [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED]:
    for shape in [(32, 1000), (64, 1000), (96, 1000), (64, 1024), (160, 544)]:
        x = torch.randn(shape, dtype=torch.bfloat16)
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        out = ttnn.to_torch(rms_norm(tx, memory_config=mc, compute_kernel_config=cfg)).float()
        ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        err = (out - ref).abs()
        bad = (err.amax(-1) > 0.1).nonzero().flatten()
        print("RES", ml, shape, mc.shard_spec.shape, "bad tile rows", sorted(set((bad // 32).tolist())))
ttnn.close_device(dev)
