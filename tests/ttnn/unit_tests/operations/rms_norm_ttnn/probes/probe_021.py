# Back-to-back dispatch probe: rms_norm (L1 width-sharded in AND out -> the zero-copy
# plan whose write_block is a no-op) immediately followed by another op's kernel on the
# same cores, with NO readback in between.  That is the regime the model has and the
# whole golden suite does not.
import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    N, W = 32, 4096
    grid = ttnn.CoreGrid(y=4, x=8)          # 32 cores -> shard width 128
    shard = ttnn.create_sharded_memory_config(
        shape=(N, W), core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    x_t = torch.randn(1, 1, N, W, dtype=torch.bfloat16)
    r_t = torch.randn(1, 1, N, W, dtype=torch.bfloat16)
    g_t = torch.randn(1, 1, W // 32, 32, dtype=torch.bfloat16)   # the flat-2D gamma quirk
    b_t = torch.randn(1, 1, W, W // 8, dtype=torch.bfloat16)

    x = ttnn.from_torch(x_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=shard)
    r = ttnn.from_torch(r_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=shard)
    g = ttnn.from_torch(g_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(b_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    print("switch is_generated:", getattr(ttnn.rms_norm, "_is_generated_rms_norm", False), flush=True)
    out = None
    for i in range(8):
        y = ttnn.rms_norm(x, weight=g, residual_input_tensor=r, memory_config=shard)
        # next op's kernel on the same cores, no readback between
        out = ttnn.matmul(y, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print("iter", i, "ok", flush=True)
    print("readback", ttnn.to_torch(out).shape, flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
