import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config
shape = (1, 1, 32, 8192)
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
x = torch.ones(shape, dtype=torch.bfloat16)          # each row constant -> 1/rms known ~1.0
g = torch.full((shape[-1],), 3.0, dtype=torch.bfloat16)  # const gamma 3.0 -> output should be 3.0
dev = ttnn.open_device(device_id=0)
try:
    mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gt = ttnn.from_torch(g.reshape(1,1,1,shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.ComputeConfigDescriptor(); cfg.math_fidelity = ttnn.MathFidelity.HiFi4; cfg.fp32_dest_acc_en = False; cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    res = ttnn.to_torch(out).float().reshape(32,8192)
    print("CASE all-ones x, const gamma 3.0: out[0,0:4]=", [round(v,4) for v in res[0,:4].tolist()], "(expect 3.0, bug 6.0)")
    print("global mean out (expect 3.0):", round(res.mean().item(),4))
finally:
    ttnn.close_device(dev)
