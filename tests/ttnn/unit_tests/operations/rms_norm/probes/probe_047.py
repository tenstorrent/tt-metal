import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
dev = ttnn.open_device(device_id=0)
try:
    for shape, tag in [
        ((1, 1, 32, 8192), "FAIL 32x8192"),
        ((1, 1, 128, 8192), "PASS 128x8192"),
        ((1, 1, 32, 4096), "PASS 32x4096"),
    ]:
        print("\n##### CASE", tag, "#####")
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn(shape[-1], dtype=torch.bfloat16)
        mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        gt = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = False
        cfg.math_approx_mode = False
        out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
        ttnn.to_torch(out)
finally:
    ttnn.close_device(dev)
