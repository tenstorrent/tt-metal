import torch, ttnn, time
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 2048, 2048)
    ti = ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def bench(mc, n=20):
        # warm (program cache)
        o = tilize(ti, use_multicore=mc)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            o = tilize(ti, use_multicore=mc)
        ttnn.synchronize_device(dev)
        t1 = time.perf_counter()
        return (t1 - t0) / n * 1e6  # us wall (host+device)

    us_sc = bench(False)
    us_mc = bench(True)
    read_bytes = 2048 * 2048 * 2
    write_bytes = 2048 * 2048 * 2
    floor_us = (read_bytes + write_bytes) / 288e9 * 1e6
    print(f"BENCH [1,1,2048,2048] bf16 RM->TILE interleaved DRAM")
    print(f"  wall/iter single-core: {us_sc:.1f} us")
    print(f"  wall/iter multi-core : {us_mc:.1f} us   (speedup {us_sc/us_mc:.2f}x)")
    print(
        f"  DRAM roofline floor  : {floor_us:.1f} us  (read {read_bytes/1e6:.2f}MB + write {write_bytes/1e6:.2f}MB / 288 GB/s)"
    )
finally:
    ttnn.close_device(dev)
