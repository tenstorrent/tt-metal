# Fused-SwiGLU FF1+FF3 minimal_matmul block / subblock sweep at one batch size, the model's operands (in0 bfp8 in L1,
# packed gate/up weight [2560, 19456] bfp4 DRAM interleaved, bfp8 out in DRAM, LoFi, fuse_swiglu). One device session,
# traced timing per config; configs the op rejects (validation, L1 clash) are reported and skipped.
# Usage: bench_ff13_sweep.py <batch> [top_n]   SWEEP="M,K,N,sh,sw;..." replaces the default grid; FF13_FP32=1 uses
#        fp32 dest accumulation (as bs1's FF1/FF3 fidelity may).
import itertools
import os
import statistics
import sys
import time

import torch

import ttnn

B = int(sys.argv[1])
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
M, K, N = B * 512, 2560, 19456


def grid():
    if os.getenv("SWEEP"):
        return [tuple(int(v) for v in c.split(",")) for c in os.getenv("SWEEP").split(";")]
    ms = (1, 2) if B == 1 else (4, 8, 16)
    cfgs = []
    for mb, kb, nb in itertools.product(ms, (5, 8, 10, 16, 20), (4, 8, 16)):
        for sh, sw in ((1, 2), (1, 4), (2, 2), (1, 8), (2, 4), (4, 2)):
            if sw % 2 or nb % sw or mb % sh or sh * sw > 8:
                continue
            cfgs.append((mb, kb, nb, sh, sw))
    return cfgs


def main():
    torch.manual_seed(0)
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    try:
        fp32 = os.getenv("FF13_FP32", "0") == "1"
        ckc = ttnn.init_device_compute_kernel_config(
            D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32, packer_l1_acc=True
        )
        w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=D)
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=D,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        results = []
        cfgs = grid()
        print(f"RES B{B}: {len(cfgs)} configs, fp32_dest_acc={fp32}", flush=True)
        for c in cfgs:
            mb, kb, nb, sh, sw = c
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
            )  # fmt: skip
            fn = lambda: ttnn.experimental.minimal_matmul(
                x, w, compute_kernel_config=ckc, config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b, fuse_swiglu=True,
            )  # fmt: skip
            try:
                for _ in range(2):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(D)
                n = 4
                tid = ttnn.begin_trace_capture(D, cq_id=0)
                outs = [fn() for _ in range(n)]
                ttnn.end_trace_capture(D, tid, cq_id=0)
                ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / n * 1e6)
                ttnn.release_trace(D, tid)
                [ttnn.deallocate(o) for o in outs]
                t = statistics.median(ts)
                results.append((t, c))
                print(f"CFG {c} {t:8.1f} us", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CFG {c} FAILED {str(e).splitlines()[0][:110] if str(e) else type(e).__name__}", flush=True)
        results.sort()
        for t, c in results[:TOP]:
            print(
                f"RES B{B} blocks M,K,N={c[:3]} sb={c[3:]}  {t:8.1f} us  {2 * M * K * N / t / 1e6:6.1f} TFLOP/s",
                flush=True,
            )
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
