import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm
from eval.sharding import shard_config

_ML = ttnn.TensorMemoryLayout
CASES = [
    ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "bshard_64c"),
    ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "wshard_w1024_8c"),
    ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "wshard_w2304_9c"),
    ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "wshard_w5120_32c"),
    ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "wshard_w7168_28c"),
    ((1, 1, 3232, 96), None, _ML.WIDTH_SHARDED, "tall_3232x96_wshard"),
]


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


device = ttnn.open_device(device_id=0)
try:
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    for shape, shard, ml, name in CASES:
        W = shape[-1]
        torch.manual_seed(42)
        tx = torch.randn(shape, dtype=torch.bfloat16)
        tg = torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W)
        if shard is not None:
            mc = shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        else:
            Rt = shape[-2] // 32
            Wt = W // 32
            mc = shard_config(
                [shape[-2], W // Wt], (Wt, 1), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
            )
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = rms_norm(x, gamma=g, epsilon=1e-6, compute_kernel_config=cfg, memory_config=mc)
        got = ttnn.to_torch(out).float()
        xf = tx.float()
        exp = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * tg.float()
        rel = float((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt())
        print(f"PREC {name:22s} pcc={pcc(got, exp):.7f} rel_rms={rel:.6f}")
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
