"""gdn_out/attn_o prefill matmul [S,6144]@[6144,5120] bfp8 scratch: my wide_pc vs auto vs the dense gdn_out config
(tpc.create_prefill_mlp_matmul_program_config, max_cols = decode grid width, TP=1 prefill tuning). Wall ms, median of 5."""
import os, sys, time
import numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
from models.demos.blackhole.qwen36.tt import tp_common as tpc
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
D = ttnn.DRAM_MEMORY_CONFIG
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
g = dev.compute_with_storage_grid_size()
for k, n in ((6144, 5120), (17408, 5120), (5120, 16384), (5120, 14336), (5120, 17408)):
    w = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    for s in [int(v) for v in sys.argv[1].split(",")]:
        x = ttnn.from_torch(torch.randn(1, 1, s, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
        r = {}
        wp = pf.wide_pc(s, k, n)
        r["wide_or_auto"] = timed(lambda: ttnn.linear(x, w, compute_kernel_config=pf._LOFI, memory_config=D, program_config=wp))
        try:
            pc = tpc.create_prefill_mlp_matmul_program_config(s, k, n, max_cols=g.x, tuning=tpc.prefill_tuning(1))
            r["dense_cfg11"] = timed(lambda: ttnn.linear(x, w, compute_kernel_config=pf._LOFI, memory_config=D, program_config=pc))
        except Exception as e:
            r["dense_cfg11_FAIL"] = 0
        print(f"[{k}->{n}] S={s}: " + "  ".join(f"{a} {b:.3f}" for a, b in r.items()), flush=True)
        ttnn.deallocate(x)
    ttnn.deallocate(w)
ttnn.close_device(dev)
