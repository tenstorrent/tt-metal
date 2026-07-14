import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

torch.manual_seed(0)
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
cases = [
    ((4, 8, 32, 256), W, ttnn.bfloat16, True, "gamma", "WIDTH multirow bf16"),
    ((2, 4, 128, 512), B, ttnn.bfloat16, True, "gamma", "BLOCK multirow bf16"),
    ((1, 1, 32, 2048), W, ttnn.float32, True, "gamma", "WIDTH fp32"),
    ((1, 1, 256, 512), B, ttnn.float32, True, "gamma", "BLOCK fp32"),
    ((1, 1, 32, 2048), W, ttnn.bfloat16, True, "no_gamma", "WIDTH bf16 no_gamma"),
    ((1, 1, 256, 512), B, ttnn.bfloat16, True, "no_gamma", "BLOCK bf16 no_gamma"),
    ((1, 1, 32, 2048), W, ttnn.bfloat8_b, True, "gamma", "WIDTH bf8b"),
]
for shape, ml, dt, acc, gm, name in cases:
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    x = torch.randn(shape, dtype=tdt)
    g = torch.randn(shape[-1], dtype=tdt)
    dev = ttnn.open_device(device_id=0)
    try:
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev)
        xt = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        gt = None
        if gm == "gamma":
            gt = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = acc
        cfg.math_approx_mode = False
        out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
        res = ttnn.to_torch(out).float()
        gg = g.float().reshape(-1) if gm == "gamma" else 1.0
        exp = (x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)) * gg
        diff = (res - exp).abs()
        pcc = torch.corrcoef(torch.stack([res.flatten(), exp.flatten()]))[0, 1].item()
        rms = (diff.pow(2).mean().sqrt() / exp.std()).item()
        print(f"CASE {name} {shape}: PCC={pcc:.6f} relRMS={rms:.4f} max={diff.max().item():.4f}")
    except Exception as e:
        print(f"CASE {name}: EXCEPTION {type(e).__name__}: {e}")
    finally:
        ttnn.close_device(dev)
