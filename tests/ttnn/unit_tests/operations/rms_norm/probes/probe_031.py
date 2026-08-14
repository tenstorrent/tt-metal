import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

_ML = ttnn.TensorMemoryLayout
device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 3232, 96)
    W = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    for gname, gt in [
        ("ones", torch.ones(W, dtype=torch.bfloat16)),
        ("arange", (torch.arange(W).float() + 1).to(torch.bfloat16)),
    ]:
        for name, ml in [("INTERLEAVED", None), ("WIDTH", _ML.WIDTH_SHARDED)]:
            mc = (
                ttnn.DRAM_MEMORY_CONFIG
                if ml is None
                else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
            )
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tg = ttnn.from_torch(gt.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            kw = {} if ml is None else {"memory_config": tx.memory_config()}
            got = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kw)).float()
            exp = xf * scale * gt.float()
            # implied per-(row,col) gamma
            impl = got / (xf * scale)
            err_per_col = (
                (impl - gt.float()).abs().mean(dim=(0, 1, 2))
                if False
                else (impl - gt.float()).abs().mean(dim=2).flatten()
            )
            err_per_row = (impl - gt.float()).abs().mean(dim=-1).flatten()
            bad_rows = (err_per_row > 0.05 * gt.float().mean()).nonzero().flatten()
            print(
                f"{gname} {name}: relRMS={((got-exp).pow(2).mean().sqrt()/exp.std()).item():.4f} "
                f"worst_col_err={err_per_col.max():.4f} at col {err_per_col.argmax().item()} "
                f"n_bad_rows={len(bad_rows)}/{impl.shape[2]} first_bad={bad_rows[:6].tolist()}",
                flush=True,
            )
            print(f"   col-err[0:12]={[round(v,3) for v in err_per_col[:12].tolist()]}", flush=True)
finally:
    ttnn.close_device(device)
