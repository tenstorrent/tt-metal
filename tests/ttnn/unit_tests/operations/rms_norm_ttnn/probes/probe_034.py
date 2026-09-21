# The model's norm input has std~58, absmax~3700. Sweep input scale; compare both
# ops against a float64 reference at the exact Gemma 60-core sharded config.
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    DIM, eps = 3840, 1e-6
    g = dev.compute_with_storage_grid_size()
    tiles = DIM // 32; best = None
    for gy in range(1, g.y+1):
        for gx in range(1, g.x+1):
            n = gx*gy
            if tiles % n == 0 and (best is None or n > best[0]): best = (n, gx, gy)
    n, gx, gy = best; bw = tiles//n; s = 4
    while s > 1 and bw % s: s -= 1
    mc = ttnn.create_sharded_memory_config(shape=(32, DIM//n), core_grid=ttnn.CoreGrid(x=gx, y=gy),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[gx,gy],
        subblock_w=s, block_h=1, block_w=bw, inplace=False)
    print(f"config: {n} cores {gx}x{gy} block_w={bw} subblock_w={s}", flush=True)
    torch.manual_seed(0)
    base = torch.randn(1,1,32,DIM)
    w_t = 1 + torch.randn(1,1,DIM//32,32)*0.05
    print(f"{'scale':>7} {'std':>8} {'absmax':>9} {'gen relerr':>12} {'nat relerr':>12} {'gen/nat':>8}")
    for scale in (1, 10, 30, 58, 120, 300):
        x_t = base*scale
        xd = x_t.double()
        ref = (xd/torch.sqrt(xd.pow(2).mean(-1,keepdim=True)+eps))*w_t.double().reshape(1,1,1,DIM)
        x = ttnn.from_torch(x_t.bfloat16(), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(w_t.bfloat16(), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r = {}
        for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
            try:
                xs = ttnn.to_memory_config(x, mc)
                y = fn(xs, weight=w, epsilon=eps, program_config=pc)
                yi = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
                d = ttnn.to_torch(yi).double()
                r[side] = ((d-ref).norm()/ref.norm()).item()
                xs.deallocate(True)
            except Exception as e:
                r[side] = str(e)[:40]
        rr = f"{r['gen']/r['nat']:.2f}x" if all(isinstance(v,float) for v in r.values()) else "-"
        print(f"{scale:7d} {x_t.std().item():8.2f} {x_t.abs().max().item():9.1f} "
              f"{r['gen']:12.4e} {r['nat']:12.4e} {rr:>8}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
