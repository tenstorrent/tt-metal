import torch, ttnn
from eval.sharding import shard_config, auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
cases = [
    ((1, 1, 32, 1024), ML.WIDTH_SHARDED, ([32, 128], (8, 1)), ttnn.TILE_LAYOUT),
    ((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4)), ttnn.TILE_LAYOUT),
    ((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10)), ttnn.TILE_LAYOUT),
    ((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10)), ttnn.TILE_LAYOUT),
    ((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), ttnn.TILE_LAYOUT),
    ((1, 1, 256, 512), ML.HEIGHT_SHARDED, None, ttnn.TILE_LAYOUT),
    ((1, 1, 416, 1184), ML.WIDTH_SHARDED, None, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 333, 1000), ML.BLOCK_SHARDED, None, ttnn.ROW_MAJOR_LAYOUT),
]
for shape, ml, shard, layout in cases:
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    mc = (
        shard_config(shard[0], shard[1], ml, layout=layout, dtype=ttnn.bfloat16, device=device)
        if shard
        else auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=layout, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg)).float()
    xf = x.float()
    ref = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()
    a, b = ref.flatten() - ref.mean(), out.flatten() - out.mean()
    pcc = float((a * b).sum() / (a.norm() * b.norm()))
    rel = float((out - ref).pow(2).mean().sqrt() / ref.std())
    print("RES", shape, ml, layout, f"pcc={pcc:.6f} rel_rms={rel:.4f} max_abs={float((out-ref).abs().max()):.3f}")
ttnn.close_device(device)
