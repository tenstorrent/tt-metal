"""Host-side (enqueue) cost of dequant() and Rotator calls: ms per call over 50 calls without sync."""
import os, time
import numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
try:
    sh = Shared(dev); mlp = TernaryMLP(sh, 0, os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp"))
    x = ttnn.from_torch(torch.randn(1, 128, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rot = pf.rotator(sh)
    for name, f in (("dequant_gu", lambda: pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True)),
                    ("dequant_down", lambda: pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120)),
                    ("rot5120", lambda: rot(x)),
                    ("prefill_mlp", lambda: pf.prefill_mlp(mlp, x, ttnn.L1_MEMORY_CONFIG))):
        r = f(); ttnn.synchronize_device(dev)
        t0 = time.perf_counter(); ts = []
        for _ in range(20):
            a = time.perf_counter(); r = f(); ts.append(time.perf_counter() - a)
            for t in (r if isinstance(r, tuple) else (r,)): ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        print(f"{name}: host enqueue {np.median(ts) * 1e3:.3f} ms/call; wall incl. device {(time.perf_counter() - t0) / 20 * 1e3:.3f} ms/call", flush=True)
finally:
    ttnn.close_device(dev)
