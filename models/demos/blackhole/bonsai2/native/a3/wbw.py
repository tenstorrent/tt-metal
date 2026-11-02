"""Stock-op DRAM write reference: typecast/copy of a [5120, 34816] interleaved DRAM tensor (wall ms, median of 5)."""
import time
import numpy as np, torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
D = ttnn.DRAM_MEMORY_CONFIG
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
x = ttnn.from_torch(torch.randn(1, 1, 5120, 34816), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
x2 = ttnn.typecast(x, ttnn.bfloat4_b)
mb = 5120 * 34816 / 1e6
for name, f, wbytes in (("bf16->bfp8", lambda: ttnn.typecast(x, ttnn.bfloat8_b, memory_config=D), 1.0625),
                        ("bf16->bfp4", lambda: ttnn.typecast(x, ttnn.bfloat4_b, memory_config=D), 0.5625),
                        ("bfp4->bfp8", lambda: ttnn.typecast(x2, ttnn.bfloat8_b, memory_config=D), 1.0625),
                        ("bf16 copy", lambda: ttnn.clone(x, memory_config=D), 2.0)):
    t = timed(f); print(f"{name}: {t:.3f} ms  write {mb * wbytes / t:.0f} GB/s", flush=True)
ttnn.close_device(dev)
