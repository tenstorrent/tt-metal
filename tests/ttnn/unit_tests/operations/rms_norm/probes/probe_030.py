import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

_ML = ttnn.TensorMemoryLayout
device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 3232, 96)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    xf = x.float()
    true_scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)  # (1,1,3232,1)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    for name, ml in [("INTERLEAVED", None), ("WIDTH", _ML.WIDTH_SHARDED)]:
        mc = (
            ttnn.DRAM_MEMORY_CONFIG
            if ml is None
            else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        )
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        kw = {} if ml is None else {"memory_config": tx.memory_config()}
        got = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg, **kw)).float()
        # per-row scale estimate from the largest-|x| element of each row (least quantization noise)
        idx = xf.abs().argmax(dim=-1, keepdim=True)
        dev_scale = got.gather(-1, idx) / xf.gather(-1, idx)
        r = (dev_scale / true_scale).flatten()
        print(
            f"{name}: scale ratio mean={r.mean():.6f} std={r.std():.6f} min={r.min():.6f} max={r.max():.6f}", flush=True
        )
        # per-element residual after removing the per-row scale error
        corrected = got / (dev_scale / true_scale)
        expct = xf * true_scale
        print(
            f"   relRMS total={( (got-expct).pow(2).mean().sqrt()/expct.std()).item():.4f}  after-scale-correction={((corrected-expct).pow(2).mean().sqrt()/expct.std()).item():.4f}",
            flush=True,
        )
        print("   first 8 ratios:", [round(v, 5) for v in r[:8].tolist()], flush=True)
finally:
    ttnn.close_device(device)
