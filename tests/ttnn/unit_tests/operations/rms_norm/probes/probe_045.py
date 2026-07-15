import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

shape = (1, 1, 32, 8192)
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
torch.manual_seed(0)
x = torch.randn(shape, dtype=torch.bfloat16)
inv = 1.0 / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)  # 1/rms per row
norm = (x.float() * inv).reshape(32, 8192)  # x/rms


def run(g, dev):
    mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gt = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    return ttnn.to_torch(out).float().reshape(32, 8192)


dev = ttnn.open_device(device_id=0)
try:
    gammas = {
        "ones(1.0)": torch.ones(8192),
        "const2.0": torch.full((8192,), 2.0),
        "const0.5": torch.full((8192,), 0.5),
        "const3.0": torch.full((8192,), 3.0),
        "randn": torch.randn(8192),
        "arange_frac": (torch.arange(8192).float() % 7 - 3) * 0.3,  # varied but bounded
    }
    for name, g in gammas.items():
        g = g.to(torch.bfloat16)
        res = run(g, dev)
        gv = g.float().reshape(-1)
        exp = norm * gv  # x/rms * gamma
        # effective gamma applied = res / norm  (where norm != 0)
        mask = norm.abs() > 1e-2
        geff = res[mask] / norm[mask]
        ratio = res[mask] / exp[mask]
        relrms = ((res - exp).pow(2).mean().sqrt() / exp.std()).item()
        print(
            f"{name:14s} relRMS={relrms:.4f}  out/exp mean={ratio.mean():.4f}  geff/gamma mean={ (geff/gv[torch.where(mask)[1]]).mean():.4f}"
        )
finally:
    ttnn.close_device(dev)
