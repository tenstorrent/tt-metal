import time, torch, ttnn
from ttnn.operations.onorm import onorm, default_compute_kernel_config

HV, V = 32, 128
FLAT = HV * V
device = ttnn.open_device(device_id=0)
try:
    for b, t in [(1, 32), (1, 64), (1, 128), (1, 640), (8, 640)]:
        o = ttnn.from_torch(
            torch.randn(b, t, HV, V, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        g = ttnn.from_torch(
            torch.randn(b, t, FLAT, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        w = ttnn.from_torch(
            torch.randn(1, 1, 1, V, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cfg = default_compute_kernel_config()
        for _ in range(3):
            out = onorm(o, g, w, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            out = onorm(o, g, w, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        dt = (time.perf_counter() - t0) / N
        bytes_moved = (b * t * FLAT * 2) * 3
        print(f"B={b} T={t:>4} blocks={b*((t+31)//32):>3}  wall={dt*1e6:9.1f} us   {bytes_moved/dt/1e9:7.1f} GB/s")
finally:
    ttnn.close_device(device)
