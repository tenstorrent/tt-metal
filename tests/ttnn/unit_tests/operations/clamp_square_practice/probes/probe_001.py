import torch, ttnn, time
from ttnn.operations.clamp_square_practice import clamp_square_practice

dev = ttnn.open_device(device_id=0)
print("RESULT arch", dev.arch(), "grid", dev.compute_with_storage_grid_size())
recs = []


def cb(batch):
    for r in batch.records:
        recs.append((int(r.runtime_id), (r.end_timestamp - r.start_timestamp) / r.frequency, int(r.core_count)))


print("RESULT active", ttnn.device.IsProgramRealtimeProfilerActive())
h = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
for shape in [
    (1, 1, 32, 32),
    (1, 1, 256, 256),
    (1, 1, 32, 8192),
    (2, 4, 256, 512),
    (8, 1, 1024, 1024),
    (1, 1, 4096, 4096),
]:
    x = ttnn.from_torch(
        torch.randn(shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    times = []
    for i in range(6):
        recs.clear()
        y = clamp_square_practice(x, -1.0, 1.0)
        ttnn.synchronize_device(dev)
        time.sleep(0.05)
        if recs:
            times.append((recs[-1][1], recs[-1][2]))
    tiles = x.buffer_num_pages()
    best = min(t[0] for t in times[1:]) if len(times) > 1 else None
    gbs = 2 * tiles * 2048 / best if best else None
    print("RESULT", shape, "tiles", tiles, "best_ns", best, "cores", times[-1][1] if times else None, "GB/s", gbs)
ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)
ttnn.close_device(dev)
