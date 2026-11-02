"""Rotation variants at S rows (wall ms, median of 5; pcc vs the current Rotator):
  cur   : x^T -> batched [nb,1024,1024]@[nb,1024,S] (auto) -> transpose back
  bigM  : x*d -> x^T -> [nb,1024,S] -> per-batch transpose [nb,S,1024] -> [nb*S,1024] @ H (2D mcast) -> back"""
import os, sys, time
import numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared
from models.demos.blackhole.bonsai2.native.hadamard import sylvester
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
D = ttnn.DRAM_MEMORY_CONFIG
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
try:
    sh = Shared(dev); rot = pf.rotator(sh)
    H = ttnn.from_torch(torch.from_numpy(sylvester(1024).astype(np.float32)).reshape(1, 1, 1024, 1024), dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    for s in [int(v) for v in sys.argv[1].split(",")]:
        for k in (5120, 6144, 17408):
            nb = k // 1024
            d = ttnn.from_torch(torch.from_numpy(sh.had.signs[k]).reshape(1, 1, 1, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
            x = ttnn.from_torch(torch.randn(1, 1, s, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
            m = nb * s
            pcm = -(-m // 32 // 10); pcn = 4
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(8, 10), in0_block_w=4, out_subblock_h=1,
                    out_subblock_w=4, out_block_h=1 if pcm > 8 else pcm, out_block_w=pcn, per_core_M=pcm, per_core_N=pcn, transpose_mcast=False,
                    fused_activation=None, fuse_batch=False)
            def bigm(pc=pc, ckc=pf._HIFI4):
                xd = ttnn.multiply(x, d, memory_config=D)
                xt = ttnn.transpose(xd, -2, -1, memory_config=D); ttnn.deallocate(xd)
                xb = ttnn.transpose(ttnn.reshape(xt, [1, nb, 1024, s]), -2, -1, memory_config=D); ttnn.deallocate(xt)
                y = ttnn.matmul(ttnn.reshape(xb, [1, 1, m, 1024]), H, compute_kernel_config=ckc, memory_config=D, dtype=ttnn.bfloat16, program_config=pc)
                ttnn.deallocate(xb)
                yt = ttnn.transpose(ttnn.reshape(y, [1, nb, s, 1024]), -2, -1, memory_config=D); ttnn.deallocate(y)
                out = ttnn.transpose(ttnn.reshape(yt, [1, 1, k, s]), -2, -1, memory_config=D); ttnn.deallocate(yt)
                return out
            ref = ttnn.to_torch(rot(x)).float(); got = ttnn.to_torch(bigm(pc, pf._LOFI)).float()
            pcc = np.corrcoef(ref.flatten().numpy(), got.flatten().numpy())[0, 1]
            same = (ref == got).float().mean().item()
            r = {"cur": timed(lambda: rot(x)), "bigM": timed(bigm), "bigM_lofi": timed(lambda: bigm(pc, pf._LOFI)), "bigM_lofi_auto": timed(lambda: bigm(None, pf._LOFI))}
            print(f"S={s} K={k}: " + "  ".join(f"{a} {b:.3f}" for a, b in r.items()) + f"  lofi vs cur: pcc {pcc:.7f} bit-same {same*100:.2f}%", flush=True)
            ttnn.deallocate(x); ttnn.deallocate(d)
finally:
    ttnn.close_device(dev)
