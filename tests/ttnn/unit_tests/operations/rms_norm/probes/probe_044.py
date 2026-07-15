import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

shape = (1, 1, 32, 8192)
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
torch.manual_seed(0)
x = torch.randn(shape, dtype=torch.bfloat16)
g = torch.randn(shape[-1], dtype=torch.bfloat16)
dev = ttnn.open_device(device_id=0)
try:
    mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gt = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    res = ttnn.to_torch(out).float().reshape(32, 8192)
    exp = (
        (x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)) * g.float().reshape(-1)
    ).reshape(32, 8192)
    diff = res - exp
    relrms = (diff.pow(2).mean().sqrt() / exp.std()).item()
    pcc = torch.corrcoef(torch.stack([res.flatten(), exp.flatten()]))[0, 1].item()
    print("relRMS", round(relrms, 4), "PCC", round(pcc, 6))
    # ratio per column, then aggregate per W-tile (32 cols) and per core (1024 cols = per_w=32 tiles)
    ratio = res / exp
    ratio = torch.where(exp.abs() > 1e-3, ratio, torch.full_like(ratio, float("nan")))
    r = ratio.nanmean(dim=0)  # per-column mean ratio, over 32 rows
    print(
        "global mean ratio",
        round(torch.nanmean(ratio).item(), 4),
        "median",
        round(ratio[~ratio.isnan()].median().item(), 4),
    )
    # per W-tile (256 tiles of 32 cols)
    rt = r.reshape(256, 32).nanmean(dim=1)
    print("per-W-tile ratio (256 tiles), first 40:")
    print([round(v, 3) for v in rt[:40].tolist()])
    print("per-core mean ratio (8 cores, 32 tiles each):")
    print([round(rt[i * 32 : (i + 1) * 32].nanmean().item(), 4) for i in range(8)])
finally:
    ttnn.close_device(dev)
