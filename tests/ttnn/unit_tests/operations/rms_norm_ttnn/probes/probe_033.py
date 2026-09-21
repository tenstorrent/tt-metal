# Gemma shapes on a 1x4 MESH (no fabric), generated vs native, per device.
import torch, ttnn

md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=8192)
try:
    print("mesh devices:", md.get_num_devices(), flush=True)
    g = md.compute_with_storage_grid_size(); print("grid:", g.x, g.y, flush=True)
    eps = 1e-6
    def build(dim):
        tiles = dim // 32; best = None
        for gy in range(1, g.y + 1):
            for gx in range(1, g.x + 1):
                n = gx * gy
                if tiles % n == 0 and (best is None or n > best[0]): best = (n, gx, gy)
        n, gx, gy = best; bw = tiles // n; s = 4
        while s > 1 and bw % s: s -= 1
        return (ttnn.create_sharded_memory_config(
                    shape=(32, dim // n), core_grid=ttnn.CoreGrid(x=gx, y=gy),
                    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True),
                ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[gx, gy], subblock_w=s,
                    block_h=1, block_w=bw, inplace=False), (n, gx, gy, bw, s))

    for dim in (1536, 3840):
        mc, pc, info = build(dim)
        torch.manual_seed(0)
        x_t = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
        w_t = torch.randn(1, 1, dim // 32, 32, dtype=torch.bfloat16)
        ref = (x_t.float() / torch.sqrt(x_t.float().pow(2).mean(-1, keepdim=True) + eps)) * w_t.float().reshape(1, 1, 1, dim)
        x = ttnn.from_torch(x_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(md))
        w = ttnn.from_torch(w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(md))
        for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
            try:
                xs = ttnn.to_memory_config(x, mc)
                y = fn(xs, weight=w, epsilon=eps, program_config=pc)
                yi = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
                per = []
                for i, t in enumerate(ttnn.get_device_tensors(yi)):
                    d = ttnn.to_torch(t).float()
                    per.append(f"d{i}:{torch.corrcoef(torch.stack([d.flatten(), ref.flatten()]))[0,1].item():.6f}")
                print(f"dim={dim} {info} {side}: " + " ".join(per), flush=True)
                xs.deallocate(True)
            except Exception as e:
                print(f"dim={dim} {side}: ERR {type(e).__name__}: {str(e)[:90]}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_mesh_device(md)
