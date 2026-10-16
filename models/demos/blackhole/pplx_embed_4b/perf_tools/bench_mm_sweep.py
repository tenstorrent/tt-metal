# Plain minimal_matmul block / subblock sweep for one model projection at one batch size, the model's operands
# (bench_mm_ablate.py's presets: weight layout, in0 / out placement). One device session, traced timing per config;
# configs the op rejects (validation, L1 clash) are reported and skipped.
# Usage: bench_mm_sweep.py <preset> <batch> [top_n]   (presets: ff2, wo, qkv, ff13plain; SWEEP="M,K,N,sh,sw;..."
#        replaces the default grid)
import itertools
import os
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_mm_ablate import preset  # noqa: E402

NAME, B = sys.argv[1], int(sys.argv[2])
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
M, K, N, BLOCKS0, SWIGLU, WLAYOUT, OUTP, *REST = preset(NAME, B)
IN0P = REST[0] if REST else "L1"
assert not SWIGLU, "fused SwiGLU: use bench_ff13_sweep.py"


def grid():
    if os.getenv("SWEEP"):
        return [tuple(int(v) for v in c.split(",")) for c in os.getenv("SWEEP").split(";")]
    kt = K // 32
    ks = [k for k in (4, 8, 16, 19, 20) if k <= kt and (kt % k == 0 or k in (8, 16))]
    cfgs = []
    for mb, kb, nb in itertools.product((4, 8, 16), ks, (4, 8, 16)):
        for sh, sw in ((1, 8), (2, 4), (1, 4), (4, 2), (2, 2)):
            if nb % sw or mb % sh or sh * sw > 8:
                continue
            cfgs.append((mb, kb, nb, sh, sw))
    return cfgs


def main():
    torch.manual_seed(0)
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        if WLAYOUT == "sharded":  # DRAM width-sharded over the 8 banks, width padded to a multiple of 8 tiles
            pad = -(-N // 256) * 256
            w_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                    [K, pad // 8],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        else:
            w_mc = ttnn.DRAM_MEMORY_CONFIG
        w = ttnn.from_torch(
            torch.randn(1, 1, K, N) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=w_mc
        )
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=D,
            memory_config=ttnn.L1_MEMORY_CONFIG if IN0P == "L1" else ttnn.DRAM_MEMORY_CONFIG,
        )
        out_mc = ttnn.L1_MEMORY_CONFIG if OUTP == "L1" else ttnn.DRAM_MEMORY_CONFIG
        cfgs = grid()
        print(f"RES {NAME} B{B}: M={M} K={K} N={N} in0={IN0P} out={OUTP} model blocks={BLOCKS0}; {len(cfgs)} configs")
        results = []
        for c in cfgs:
            mb, kb, nb, sh, sw = c
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
            )
            fn = lambda: ttnn.experimental.minimal_matmul(
                x, w, compute_kernel_config=ckc, config=cfg, memory_config=out_mc, dtype=ttnn.bfloat8_b
            )
            try:
                for _ in range(2):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(D)
                n = 4
                tid = ttnn.begin_trace_capture(D, cq_id=0)
                keep = []
                for _ in range(n):
                    o = fn()
                    ttnn.deallocate(o) if OUTP == "L1" else keep.append(o)  # live L1 outputs would not fit
                ttnn.end_trace_capture(D, tid, cq_id=0)
                ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / n * 1e6)
                ttnn.release_trace(D, tid)
                [ttnn.deallocate(o) for o in keep]
                t = statistics.median(ts)
                results.append((t, c))
                print(f"CFG {c} {t:8.1f} us", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CFG {c} FAILED {str(e).splitlines()[0][:110] if str(e) else type(e).__name__}", flush=True)
        results.sort()
        for t, c in results[:TOP]:
            print(
                f"RES {NAME} B{B} blocks M,K,N={c[:3]} sb={c[3:]}  {t:8.1f} us  {2 * M * K * N / t / 1e6:6.1f} TFLOP/s",
                flush=True,
            )
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
