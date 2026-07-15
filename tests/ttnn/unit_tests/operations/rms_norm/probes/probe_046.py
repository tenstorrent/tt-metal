import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

B = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def run(shape, has_gamma, acc, dev):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16) if has_gamma else None
    mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gt = (
        ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        if has_gamma
        else None
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = acc
    cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    res = ttnn.to_torch(out).float()
    inv = 1.0 / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
    exp = x.float() * inv
    if has_gamma:
        exp = exp * g.float().reshape(-1)
    exp = exp.reshape(res.shape)
    relrms = ((res - exp).pow(2).mean().sqrt() / exp.std()).item()
    m = exp.abs() > 1e-2
    ratio = (res[m] / exp[m]).mean().item()
    return relrms, ratio


dev = ttnn.open_device(device_id=0)
try:
    cases = [
        ((1, 1, 32, 8192), False, False, "32x8192 no-gamma acc=F"),
        ((1, 1, 32, 8192), True, False, "32x8192 gamma    acc=F  <-FAIL"),
        ((1, 1, 32, 8192), True, True, "32x8192 gamma    acc=T"),
        ((1, 1, 128, 8192), True, False, "128x8192 gamma   acc=F (ny=4)"),
        ((1, 1, 32, 4096), True, False, "32x4096 gamma    acc=F (nwb=2)"),
    ]
    for shape, hg, acc, name in cases:
        rr, ra = run(shape, hg, acc, dev)
        print(f"{name:34s} relRMS={rr:.4f} ratio={ra:.4f}")
finally:
    ttnn.close_device(dev)
