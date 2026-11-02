"""Per-site ternary prefill stages (wall ms, median of 5): rotate, dequant (bfp8/bfp4), matmul, and the whole
prefill_linear, for every TernaryLinear site at S in argv[1]."""
import os, sys, time
import numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native.a3 import linear as al, prefill as pf
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
D = ttnn.DRAM_MEMORY_CONFIG
def timed(f, n=5):
    r = f(); [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]; ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
        [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]
    return float(np.median(ts))
try:
    sh = Shared(dev); cache = os.path.join(os.environ["TT_CACHE_PATH"], "a3_linear"); rot = pf.rotator(sh)
    sites = [("gdn_in", al.TernaryLinear(sh, "blk.0.gdn_in", cache, rows=lambda: al.gdn_in_rows(sh, 0), shape=al.gdn_in_shape(sh, 0))),
             ("gdn_out", al.TernaryLinear(sh, "blk.0.ssm_out.weight", cache)),
             ("attn_qkv", al.TernaryLinear(sh, "blk.3.attn_qkv_fused", cache, rows=lambda: al.attn_qkv_rows(sh, 3), shape=al.attn_qkv_shape(sh, 3))),
             ("attn_o", al.TernaryLinear(sh, "blk.3.attn_output.weight", cache))]
    for s in [int(v) for v in sys.argv[1].split(",")]:
        for name, tl in sites:
            x = ttnn.from_torch(torch.randn(1, s, tl.k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
            xr = rot(x)
            w8 = pf.dequant(tl.w, tl.s, tl.k, tl.n); w4 = pf.dequant(tl.w, tl.s, tl.k, tl.n, dtype=ttnn.bfloat4_b)
            pc = pf.wide_pc(s, tl.k, tl.n)
            r = {"rot": timed(lambda: rot(x)),
                 "dq8": timed(lambda: pf.dequant(tl.w, tl.s, tl.k, tl.n)),
                 "dq4": timed(lambda: pf.dequant(tl.w, tl.s, tl.k, tl.n, dtype=ttnn.bfloat4_b)),
                 "mm8": timed(lambda: ttnn.linear(xr, w8, compute_kernel_config=pf._LOFI, memory_config=D, program_config=pc)),
                 "mm8_auto": timed(lambda: ttnn.linear(xr, w8, compute_kernel_config=pf._LOFI, memory_config=D)),
                 "mm4": timed(lambda: ttnn.linear(xr, w4, compute_kernel_config=pf._LOFI, memory_config=D, program_config=pc)),
                 "total": timed(lambda: pf.prefill_linear(tl, x))}
            print(f"S={s} {name} [{tl.k}->{tl.n}]: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
            for t in (x, xr, w8, w4): ttnn.deallocate(t)
finally:
    ttnn.close_device(dev)
