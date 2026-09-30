import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

device = ttnn.open_device(device_id=0)
for n, T, C in [(2, 640, 1792), (1, 640, 1792)]:
    f = torch.randn(1, 1, T, C)
    x = torch.randn(1, 1, T, n * C)
    post = torch.rand(1, 1, T, n)
    comb = torch.rand(1, 1, T, n * n)
    d = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out = mhc_post(d(f, ttnn.bfloat16), d(x, ttnn.bfloat16), d(post, ttnn.float32), d(comb, ttnn.float32))
        print("PROBE_RESULT n", n, "OK", ttnn.to_torch(out).shape)
    except Exception as e:
        print("PROBE_RESULT n", n, "FAIL", str(e)[:200])
ttnn.close_device(device)
