"""Dequant gate|up wall ms (median of 5), layer 0: as configured by env (BONSAI_DQ_SKIPW, BONSAI_DQ_FP32)."""
import os, time
import numpy as np, ttnn
from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
try:
    sh = Shared(dev); mlp = TernaryMLP(sh, 0, os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp"))
    for dt in (ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat16):
        f = lambda: pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True, dtype=dt)
        [ttnn.deallocate(t) for t in f()]; ttnn.synchronize_device(dev); ts = []
        for _ in range(5):
            t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); [ttnn.deallocate(t) for t in r]
        print(f"skipw={os.environ.get('BONSAI_DQ_SKIPW','0')} fp32={os.environ.get('BONSAI_DQ_FP32','1')} {dt}: gu {np.median(ts):.3f} ms", flush=True)
finally:
    ttnn.close_device(dev)
