# Same sequence as probe_b2b, but CAPTURED AS A TRACE and replayed -- the regime the
# model has (tt-triage named "MatmulDeviceOperation (trace id: 2)") and no suite has.
import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=8192, trace_region_size=64 << 20)
try:
    N, W = 32, 4096
    grid = ttnn.CoreGrid(y=4, x=8)
    shard = ttnn.create_sharded_memory_config(
        shape=(N, W), core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    x = ttnn.from_torch(torch.randn(1, 1, N, W, dtype=torch.bfloat16), ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=shard)
    r = ttnn.from_torch(torch.randn(1, 1, N, W, dtype=torch.bfloat16), ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=shard)
    g = ttnn.from_torch(torch.randn(1, 1, W // 32, 32, dtype=torch.bfloat16), ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(1, 1, W, W // 8, dtype=torch.bfloat16), ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    print("switch is_generated:", getattr(ttnn.rms_norm, "_is_generated_rms_norm", False), flush=True)

    def body():
        y = ttnn.rms_norm(x, weight=g, residual_input_tensor=r, memory_config=shard)
        return ttnn.matmul(y, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    body(); ttnn.synchronize_device(dev)          # compile + populate program cache
    print("warm ok", flush=True)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(4):
        out = body()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    print("captured", flush=True)

    for i in range(6):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        print("replay", i, "issued", flush=True)
    ttnn.synchronize_device(dev)
    print("replayed, readback", ttnn.to_torch(out).shape, flush=True)
    ttnn.release_trace(dev, tid)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
