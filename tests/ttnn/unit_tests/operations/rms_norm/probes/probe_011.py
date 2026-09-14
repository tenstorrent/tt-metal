import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    print("grid", grid.x, grid.y)
    for shape in [(1, 1, 160, 11008), (1, 224, 11008), (1, 1, 32, 2848), (7, 224, 3072)]:
        mc = auto_shard_config(
            list(shape),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
        )
        ss = mc.shard_spec
        ncores = len(ttnn.corerange_to_cores(ss.grid, None, True))
        print(shape, "shard", ss.shape, "ncores", ncores, "cover", ncores * ss.shape[1], "W", shape[-1])
        x = torch.randn(shape).bfloat16()
        g = torch.randn(shape[-1]).bfloat16().reshape(1, 1, 1, -1)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        try:
            out = rms_norm(tx, gamma=tg, memory_config=mc)
            xf, gf = x.float(), g.float().reshape(-1)
            ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * gf
            got = ttnn.to_torch(out).float()
            err = (got - ref).abs().max().item()
            print("  OK max_abs_err", err)
        except Exception as e:
            print("  ERR", type(e).__name__, str(e)[:200])
        ttnn.deallocate(tx)
finally:
    ttnn.close_device(device)
