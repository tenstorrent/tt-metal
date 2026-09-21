# Sweep every sharded config a Gemma-4-12B decode norm could build on ANY grid,
# plus the prefill and no-gamma paths. generated vs native, same input.
import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    g = dev.compute_with_storage_grid_size()
    DIM, eps = 3840, 1e-6
    def pcc(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    torch.manual_seed(0)
    def run(tag, x_t, w_t, mc, pc):
        ref = x_t.float() / torch.sqrt(x_t.float().pow(2).mean(-1, keepdim=True) + eps)
        if w_t is not None:
            ref = ref * w_t.float().reshape(1, 1, 1, -1)
        x = ttnn.from_torch(x_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = None if w_t is None else ttnn.from_torch(
            w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        res = {}
        for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
            try:
                xs = ttnn.to_memory_config(x, mc) if mc else x
                kw = {"epsilon": eps}
                if w is not None: kw["weight"] = w
                if pc is not None: kw["program_config"] = pc
                y = fn(xs, **kw)
                yi = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG) if mc else y
                res[side] = f"{pcc(ttnn.to_torch(yi), ref):.6f}"
                if mc: xs.deallocate(True)
            except Exception as e:
                res[side] = f"ERR {str(e)[:48]}"
        flag = "" if res["gen"] == res["nat"] else "   <<<< DIFFERS"
        print(f"{tag:46s} gen={res['gen']:>12s} nat={res['nat']:>12s}{flag}", flush=True)

    x_t = torch.randn(1, 1, 32, DIM, dtype=torch.bfloat16)
    w_t = torch.randn(1, 1, DIM // 32, 32, dtype=torch.bfloat16)
    tiles = DIM // 32
    seen = set()
    for gy in range(1, g.y + 1):
        for gx in range(1, g.x + 1):
            n = gx * gy
            if n == 1 or tiles % n or n in seen: continue
            seen.add(n)
            bw = tiles // n
            sub = 4
            while sub > 1 and bw % sub != 0: sub -= 1
            mc = ttnn.create_sharded_memory_config(
                shape=(32, DIM // n), core_grid=ttnn.CoreGrid(x=gx, y=gy),
                strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[gx, gy], subblock_w=sub,
                block_h=1, block_w=bw, inplace=False)
            run(f"decode sharded {n:3d}c {gx}x{gy} bw={bw} sub={sub}", x_t, w_t, mc, pc)

    run("prefill interleaved h=128 (plain path)",
        torch.randn(1, 1, 128, DIM, dtype=torch.bfloat16), w_t, None, None)
    run("per-head no-gamma [1,16,32,256]",
        torch.randn(1, 16, 32, 256, dtype=torch.bfloat16), None, None, None)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
