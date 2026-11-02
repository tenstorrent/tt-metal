"""GDN a|b prefill matmul [S,5120] @ [5120,96] bf16: ttnn auto vs a row-split 1D config (wall ms, median of 5)."""
import sys, time
import numpy as np, torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
D = ttnn.DRAM_MEMORY_CONFIG
cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
def timed(f, n=5):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(r)
    return float(np.median(ts))
w = ttnn.from_torch(torch.randn(1, 1, 5120, 96) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
g = dev.compute_with_storage_grid_size()
for s in [int(v) for v in sys.argv[1].split(",")]:
    x = ttnn.from_torch(torch.randn(1, s, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=cfg, memory_config=D))
    out = {"auto": timed(lambda: ttnn.linear(x, w, compute_kernel_config=cfg, memory_config=D))}
    mt = s // 32
    for ncores in (mt, min(mt, 64), min(mt, 32)):
        pcm = -(-mt // ncores); nc = -(-mt // pcm)
        for ibw in (4, 8, 16):
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(g.x, g.y), in0_block_w=ibw,
                    out_subblock_h=1, out_subblock_w=3, per_core_M=pcm, per_core_N=3, fuse_batch=True, fused_activation=None, mcast_in0=False)
            try:
                f = lambda: ttnn.linear(x, w, compute_kernel_config=cfg, memory_config=D, program_config=pc)
                same = torch.allclose(ttnn.to_torch(f()).float(), ref.float(), atol=1e-2, rtol=1e-2)
                out[f"1d pcm{pcm} ibw{ibw}"] = timed(f)
                if not same: out[f"1d pcm{pcm} ibw{ibw} MISMATCH"] = 0
            except Exception as e:
                print(f"S={s} pcm{pcm} ibw{ibw} FAIL {str(e).splitlines()[0][:100]}")
    print(f"S={s}: " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
    ttnn.deallocate(x)
ttnn.close_device(dev)
