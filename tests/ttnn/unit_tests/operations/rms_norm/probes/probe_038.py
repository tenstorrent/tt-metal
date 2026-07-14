import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

torch.manual_seed(0)
I = ttnn.TensorMemoryLayout.INTERLEAVED
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
cases = [
    # non-regression (shared compute/writer changed)
    ((1, 1, 64, 8192), I, ttnn.TILE_LAYOUT, ttnn.bfloat16, "interleaved TILE bf16"),
    ((1, 1, 64, 512), I, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, "interleaved RM bf16"),
    ((1, 1, 256, 512), H, ttnn.TILE_LAYOUT, ttnn.bfloat16, "HEIGHT TILE bf16"),
    # fallback (WIDTH/BLOCK non-cross-core): RM input, non-aligned W
    ((1, 1, 64, 512), W, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, "WIDTH RM (fallback)"),
    ((1, 1, 32, 2000), W, ttnn.TILE_LAYOUT, ttnn.bfloat16, "WIDTH w_non_aligned (fallback)"),
    ((1, 1, 256, 500), B, ttnn.TILE_LAYOUT, ttnn.bfloat16, "BLOCK w_non_aligned (fallback)"),
]
for shape, ml, lay, dt, name in cases:
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    dev = ttnn.open_device(device_id=0)
    try:
        if ml == I:
            mc = None
        else:
            mc = auto_shard_config(list(shape), ml, layout=lay, dtype=dt, device=dev)
        xt = (
            ttnn.from_torch(x, dtype=dt, layout=lay, device=dev, memory_config=mc)
            if mc
            else ttnn.from_torch(x, dtype=dt, layout=lay, device=dev)
        )
        gt = ttnn.from_torch(
            g.reshape(1, 1, 1, shape[-1]),
            dtype=dt,
            layout=(ttnn.TILE_LAYOUT if lay == ttnn.TILE_LAYOUT else ttnn.ROW_MAJOR_LAYOUT),
            device=dev,
        )
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        kw = {"compute_kernel_config": cfg}
        if mc:
            kw["memory_config"] = xt.memory_config()
        out = rms_norm(xt, gamma=gt, epsilon=1e-6, **kw)
        res = ttnn.to_torch(out).float()
        exp = (x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)) * g.float().reshape(-1)
        diff = (res - exp).abs()
        pcc = torch.corrcoef(torch.stack([res.flatten(), exp.flatten()]))[0, 1].item()
        rms = (diff.pow(2).mean().sqrt() / exp.std()).item()
        print(f"CASE {name} {shape}: PCC={pcc:.6f} relRMS={rms:.4f}")
    except Exception as e:
        print(f"CASE {name}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
    finally:
        ttnn.close_device(dev)
