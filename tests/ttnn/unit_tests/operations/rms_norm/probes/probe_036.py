import torch, ttnn, sys
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
for shape in [(1504, 544), (7168, 1024)]:
    x = torch.randn(shape, dtype=torch.bfloat16)
    mc = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev
    )
    print("RES", shape, mc.shard_spec.shape, mc.shard_spec.grid)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    out = ttnn.to_torch(rms_norm(tx, memory_config=mc, compute_kernel_config=cfg)).float()
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    err = (out - ref).abs()
    bad = (err.amax(-1) > 0.1).nonzero().flatten()
    print(
        "RES bad rows",
        bad.tolist(),
        "ratio",
        (out[bad] / ref[bad]).median(-1).values.tolist(),
        "badcols",
        (err[bad] > 0.1).sum(-1).tolist(),
    )
ttnn.close_device(dev)
