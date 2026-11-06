import time, torch, ttnn
from eval.golden_tests.nomic_sdpa.helpers import make_qkv
import ttnn.operations.nomic_sdpa.nomic_sdpa as m
from ttnn.operations.nomic_sdpa import nomic_sdpa

device = ttnn.open_device(device_id=0)
try:
    q, k, v = make_qkv((8, 12, 512, 64), "real", 0)
    tq, tk, tv = (
        ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        for t in (q, k, v)
    )
    outs = []
    o = nomic_sdpa(tq, tk, tv)
    ttnn.synchronize_device(device)
    ttnn.deallocate(o)
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        o = nomic_sdpa(tq, tk, tv)
        ttnn.deallocate(o)
    t1 = time.perf_counter()
    ttnn.synchronize_device(device)
    print(f"H per-call host (incl. queueing) {(t1-t0)/n*1e6:.1f} us")
    # component timing
    t0 = time.perf_counter()
    for _ in range(n):
        m.validate(tq, tk, tv)
    print(f"H validate {(time.perf_counter()-t0)/n*1e6:.1f} us")
    t0 = time.perf_counter()
    for _ in range(n):
        m._l1_fits(device, 1)
    print(f"H _l1_fits {(time.perf_counter()-t0)/n*1e6:.1f} us")
    t0 = time.perf_counter()
    for _ in range(n):
        m.circular_buffer_bytes(8, 512)
    print(f"H circular_buffer_bytes {(time.perf_counter()-t0)/n*1e6:.1f} us")
finally:
    ttnn.close_device(device)
