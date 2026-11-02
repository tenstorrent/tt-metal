"""prefill_mlp vs the dense Qwen36MLP prefill op sequence, as called in-model (x [1,S,5120] DRAM, down with
qwen36 prefill_progcfg). Dense weights = the dequantized layer-0 weights typecast to the dense dtypes
(w1/w3 bfp4, w2 bfp8). Wall ms, median of 5."""
import os, sys, time
import numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
from models.demos.blackhole.qwen36.tt import tp_common as tpc
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
try:
    sh = Shared(dev); mlp = TernaryMLP(sh, 0, os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp"))
    wg, wu = pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True)
    wd = pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120)
    w1, w3 = ttnn.typecast(wg, ttnn.bfloat4_b), ttnn.typecast(wu, ttnn.bfloat4_b)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False)
    pcf = lambda q, k, n: tpc.create_prefill_matmul_program_config(q, k, n, grid_size=tpc.prefill_grid_default(), tuning=tpc.prefill_tuning(1))
    for s in [int(v) for v in sys.argv[1].split(",")]:
        mc = ttnn.L1_MEMORY_CONFIG if s <= 512 else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.from_torch(torch.randn(1, s, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dpc = pcf(s, 17408, 5120)
        def dense():
            a = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            b = ttnn.linear(x, w3, compute_kernel_config=ckc, memory_config=mc)
            h = ttnn.mul(a, b, memory_config=mc); ttnn.deallocate(a); ttnn.deallocate(b)
            y = ttnn.linear(h, wd, compute_kernel_config=ckc, memory_config=mc, program_config=dpc); ttnn.deallocate(h)
            return y
        r = {"dense": timed(dense), "ternary": timed(lambda: pf.prefill_mlp(mlp, x, mc, dpc))}
        os.environ["BONSAI_PF_GU"] = "bfp4"; r["ternary_gu_bfp4"] = timed(lambda: pf.prefill_mlp(mlp, x, mc, dpc)); del os.environ["BONSAI_PF_GU"]
        print(f"S={s}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
        ttnn.deallocate(x)
finally:
    ttnn.close_device(dev)
