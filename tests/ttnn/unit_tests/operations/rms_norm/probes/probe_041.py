import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

torch.manual_seed(0)
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
cases = [
    ((1, 1, 32, 8192), B, True, "BLOCK 32x8192 acc=True"),
    ((1, 1, 32, 8192), B, False, "BLOCK 32x8192 acc=False"),
    ((1, 1, 128, 8192), B, False, "BLOCK 128x8192 acc=False"),
    ((1, 1, 32, 4096), B, False, "BLOCK 32x4096 acc=False"),
]
for shape, ml, acc, name in cases:
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    dev = ttnn.open_device(device_id=0)
    try:
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        gt = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = acc
        cfg.math_approx_mode = False
        out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
        res = ttnn.to_torch(out).float()
        exp = (x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)) * g.float().reshape(-1)
        diff = (res - exp).abs()
        pcc = torch.corrcoef(torch.stack([res.flatten(), exp.flatten()]))[0, 1].item()
        rms = (diff.pow(2).mean().sqrt() / exp.std()).item()
        print(f"CASE {name}: PCC={pcc:.6f} relRMS={rms:.4f}")
    except Exception as e:
        print(f"CASE {name}: EXCEPTION {type(e).__name__}: {str(e)[:100]}")
    finally:
        ttnn.close_device(dev)
