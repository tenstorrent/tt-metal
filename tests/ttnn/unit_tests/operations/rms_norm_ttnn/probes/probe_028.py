# Gemma RMSNorm decode fast path, generated vs native, at both models' hidden sizes.
# Replicates models/demos/gemma4/tt/rms_norm.py::_build_sharded_cfg exactly.
import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    grid = dev.compute_with_storage_grid_size()
    print("compute grid:", grid.x, "x", grid.y, flush=True)

    def build_cfg(dim):
        tiles = dim // ttnn.TILE_SIZE
        best = None
        for gy in range(1, grid.y + 1):
            for gx in range(1, grid.x + 1):
                n = gx * gy
                if tiles % n == 0 and (best is None or n > best[0]):
                    best = (n, gx, gy)
        if best is None or best[0] == 1:
            return None
        num_cores, gx, gy = best
        block_w = tiles // num_cores
        sub = 4
        while sub > 1 and block_w % sub != 0:
            sub -= 1
        mc = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, dim // num_cores),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy], subblock_w=sub,
            block_h=1, block_w=block_w, inplace=False)
        return mc, pc, (num_cores, gx, gy, block_w, sub)

    def pcc(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    eps = 1e-6
    for name, dim in (("E2B", 1536), ("12B", 3840)):
        cfg = build_cfg(dim)
        if cfg is None:
            print(f"{name} dim={dim}: no sharded cfg"); continue
        mc, pc, info = cfg
        n, gx, gy, bw, sub = info
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
        w_t = torch.randn(1, 1, dim // 32, 32, dtype=torch.bfloat16)
        ref = (x_t.float() / torch.sqrt(x_t.float().pow(2).mean(-1, keepdim=True) + eps)) * w_t.float().reshape(1, 1, 1, dim)

        x = ttnn.from_torch(x_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = {}
        for side, fn in (("generated", ttnn.rms_norm), ("native", ttnn._native_rms_norm)):
            xs = ttnn.to_memory_config(x, mc)
            try:
                y = fn(xs, weight=w, epsilon=eps, program_config=pc)
                yi = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
                out[side] = ttnn.to_torch(yi)
            except Exception as e:
                out[side] = e
            finally:
                xs.deallocate(True)
        line = f"{name} dim={dim} cores={n} grid={gx}x{gy} block_w={bw} subblock_w={sub}"
        for side in ("generated", "native"):
            r = out[side]
            line += f" | {side}: " + (f"ERR {type(r).__name__}: {str(r)[:70]}" if isinstance(r, Exception)
                                      else f"PCC {pcc(r, ref):.6f}")
        print(line, flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
