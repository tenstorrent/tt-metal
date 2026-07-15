import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config
shape = (1, 1, 32, 8192)
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
dev = ttnn.open_device(device_id=0)
try:
    for rep in range(5):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16); g = torch.randn(shape[-1], dtype=torch.bfloat16)
        mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        gt = ttnn.from_torch(g.reshape(1,1,1,shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = ttnn.ComputeConfigDescriptor(); cfg.math_fidelity = ttnn.MathFidelity.HiFi4; cfg.fp32_dest_acc_en = False; cfg.math_approx_mode = False
        out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
        res = ttnn.to_torch(out).float().reshape(32,8192)
        inv = 1.0/torch.sqrt(torch.mean(x.float()**2,dim=-1,keepdim=True)+1e-6)
        exp = (x.float()*inv*g.float().reshape(-1)).reshape(32,8192)
        m=exp.abs()>1e-2
        print(f"REP{rep} relRMS={((res-exp).pow(2).mean().sqrt()/exp.std()).item():.4f} ratio={(res[m]/exp[m]).mean():.4f}")
finally:
    ttnn.close_device(dev)
