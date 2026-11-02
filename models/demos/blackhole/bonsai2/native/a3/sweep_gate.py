"""Sweep 2D mcast configs for the prefill gate/up matmul [S,5120] @ [5120,17408] bfp8 (wall ms, median of 5)."""
import math, os, sys, time
import numpy as np, torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=24576)
LOFI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False)
w = ttnn.from_torch(torch.randn(1, 1, 5120, 17408) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
for s in [int(v) for v in sys.argv[1].split(",")]:
    x = ttnn.from_torch(torch.randn(1, 1, s, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print(f"S={s} auto {timed(lambda: ttnn.linear(x, w, compute_kernel_config=LOFI, memory_config=ttnn.DRAM_MEMORY_CONFIG)):.3f}", flush=True)
    for gx, gy in ((8, 10),):
        pcm = math.ceil(s // 32 / gy); pcn = math.ceil(544 / gx)
        for ibw, bh, bw, sw in ((4, 1, pcn, 4), (2, 1, pcn, 4), (8, 1, pcn, 4), (4, pcm, pcn // 2, 2), (2, pcm, pcn // 4, 1), (4, 2 if pcm % 2 == 0 else 1, pcn, 4)):
            if pcm % bh or pcn % bw or bw % sw: continue
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw,
                out_subblock_h=1, out_subblock_w=sw, out_block_h=bh, out_block_w=bw, per_core_M=pcm, per_core_N=pcn,
                transpose_mcast=False, fused_activation=None, fuse_batch=False)
            try:
                t = timed(lambda: ttnn.linear(x, w, compute_kernel_config=LOFI, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc))
                print(f"S={s} grid {gx}x{gy} ibw {ibw} bh {bh} bw {bw} sw {sw}: {t:.3f}", flush=True)
            except Exception as e:
                print(f"S={s} grid {gx}x{gy} ibw {ibw} bh {bh} bw {bw} sw {sw}: FAIL {str(e).splitlines()[0][:120]}", flush=True)
    ttnn.deallocate(x)
ttnn.close_device(dev)
