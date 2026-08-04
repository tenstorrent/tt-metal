import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi4
c.fp32_dest_acc_en = True
c.math_approx_mode = False
W, eps = 200, 1e-5
torch.manual_seed(0)
x = torch.randn(1, 1, 32, W)
for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
    for ml in ("INTERLEAVED", "WIDTH_SHARDED"):
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        if ml == "WIDTH_SHARDED":
            cfg = ttnn.create_sharded_memory_config(
                shape=(32, 96),
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            tx = ttnn.to_memory_config(tx, memory_config=cfg)
        tx = ttnn.fill_implicit_tile_padding(tx, 1000.0)
        xf = ttnn.to_torch(ttnn.to_memory_config(tx, ttnn.L1_MEMORY_CONFIG)).float()[..., :W]
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        g = ttnn.from_torch(
            torch.ones(1, 1, 1, W, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        out = rms_norm(
            tx,
            gamma=g,
            epsilon=eps,
            compute_kernel_config=c,
            memory_config=(tx.memory_config() if ml != "INTERLEAVED" else None),
        )
        a = ttnn.to_torch(out).float()[..., :W]
        fro = ((a - ref).norm() / ref.norm()).item()
        print(
            f"RES {str(dt).split('.')[-1]:12s} {ml:15s} frobenius={fro:.5f} (budget bf8b 0.10 / bf16 0.06)", flush=True
        )
ttnn.close_device(dev)
