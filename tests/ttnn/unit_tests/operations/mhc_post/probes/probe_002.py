import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

dev = ttnn.open_device(device_id=0)
n, T, C = 2, 640, 1792
d = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
try:
    out = mhc_post(
        d(torch.randn(1, 1, T, C), ttnn.bfloat16),
        d(torch.randn(1, 1, T, n * C), ttnn.bfloat16),
        d(torch.rand(1, 1, T, n), ttnn.float32),
        d(torch.rand(1, 1, T, n * n), ttnn.float32),
    )
    print("OK_N2")
except Exception as e:
    print("PROBE_RESULT fail", str(e)[:200])
ttnn.close_device(dev)
