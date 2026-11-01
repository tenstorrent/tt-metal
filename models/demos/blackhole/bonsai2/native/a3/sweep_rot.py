"""Sweep MatmulMultiCoreReuseProgramConfig for the rotation bmm [1,nb,1024,1024] @ [1,nb,1024,S] (wall ms, median of 5)."""
import time, sys
import numpy as np, torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
D = ttnn.DRAM_MEMORY_CONFIG
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
g = dev.compute_with_storage_grid_size(); print("grid", g.x, g.y)
for s, nb in ((128, 17), (128, 6), (128, 5)):
    a = ttnn.from_torch(torch.sign(torch.randn(1, nb, 1024, 1024)) / 32, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, nb, 1024, s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=HIFI4, memory_config=D, dtype=ttnn.bfloat16))
    print(f"S={s} nb={nb} auto {timed(lambda: ttnn.matmul(a, b, compute_kernel_config=HIFI4, memory_config=D, dtype=ttnn.bfloat16)):.3f}", flush=True)
    for fid_name, cfg in (("hifi4", HIFI4),):
        for gx, gy in ((g.x, g.y),):
            for ibw, pcm, pcn, sh, sw in ((4, 32, 4, 4, 1), (4, 16, 4, 4, 1), (4, 8, 4, 2, 2), (4, 4, 4, 1, 4), (8, 8, 4, 2, 2), (8, 4, 4, 1, 4), (8, 2, 4, 1, 4)):
                if pcn > s // 32: continue
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw,
                        out_subblock_h=sh, out_subblock_w=sw, per_core_M=pcm, per_core_N=pcn)
                try:
                    f = lambda: ttnn.matmul(a, b, compute_kernel_config=cfg, memory_config=D, dtype=ttnn.bfloat16, program_config=pc)
                    ok = torch.equal(ttnn.to_torch(f()), ref)
                    print(f"S={s} nb={nb} grid {gx}x{gy} ibw {ibw} pcm {pcm} pcn {pcn} sub {sh}x{sw}: {timed(f):.3f} same={ok}", flush=True)
                except Exception as e:
                    print(f"S={s} nb={nb} grid {gx}x{gy} ibw {ibw} pcm {pcm} pcn {pcn}: FAIL {str(e).splitlines()[0][:100]}", flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)
ttnn.close_device(dev)
