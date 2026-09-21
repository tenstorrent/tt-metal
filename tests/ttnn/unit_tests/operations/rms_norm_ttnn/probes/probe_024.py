# Tighter: small follow-on ops on the SAME sharded cores, so the next kernel's
# noc_local_state_init snapshot happens while our ACKs are still in flight.
import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=8192, trace_region_size=64 << 20)
try:
    N, W = 32, 4096
    grid = ttnn.CoreGrid(y=4, x=8)
    shard = ttnn.create_sharded_memory_config(
        shape=(N, W), core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    mk = lambda: ttnn.from_torch(torch.randn(1, 1, N, W, dtype=torch.bfloat16), ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=shard)
    x, r, a = mk(), mk(), mk()
    g = ttnn.from_torch(torch.randn(1, 1, W // 32, 32, dtype=torch.bfloat16), ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("switch is_generated:", getattr(ttnn.rms_norm, "_is_generated_rms_norm", False), flush=True)

    def body():
        y = ttnn.rms_norm(x, weight=g, residual_input_tensor=r, memory_config=shard)
        y = ttnn.add(y, a, memory_config=shard)          # same cores, tiny, writer barriers
        y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        return ttnn.interleaved_to_sharded(y, shard)

    out = body(); ttnn.synchronize_device(dev)
    print("warm ok", flush=True)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(16):
        out = body()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    print("captured", flush=True)
    for i in range(20):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    print("replayed x20, readback", ttnn.to_torch(out).shape, flush=True)
    ttnn.release_trace(dev, tid)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
