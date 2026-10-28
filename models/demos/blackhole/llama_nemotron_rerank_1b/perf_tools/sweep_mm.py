# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sweep matmul configs for the reranker's weight shapes; time each inside a trace.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/sweep_mm.py --m 512 --top 5

Shapes: QKV 2048x3072, WO 2048x2048, FF1/FF3 2048x8192, FF2 8192x2048.
Candidates: ttnn.linear with its own config (baseline) and minimal_matmul over
block/subblock sizes, at HiFi2 and LoFi, with and without fp32 accumulation.
Each candidate runs REPS times in one trace; the reported time is per call.
"""

import argparse
import itertools
import time

import torch

import ttnn

SHAPES = {
    "qkv": (2048, 3072),
    "wo": (2048, 2048),
    "ff13": (2048, 8192),
    "ff2": (8192, 2048),
    "ff13s": (2048, 16384),
    "ff13w": (2048, 16384),
    "score": (2048, 32),
}
REPS = 10


def timed(device, fn):
    fn()  # compile
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        best = min(best, (time.perf_counter() - t) / REPS * 1e6)
    ttnn.release_trace(device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[512, 4096, 16384])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--wdtype", default="bfloat8_b")
    ap.add_argument("--xdtype", default="bfloat16", help="activation (in0) dtype")
    ap.add_argument(
        "--l1-reserve-mb", type=float, default=0, help="hold this much interleaved L1 (e.g. a resident residual)"
    )
    ap.add_argument("--odtype", default="bfloat16", help="output dtype")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--fid", nargs="+", default=["HiFi2"])
    ap.add_argument("--acc", nargs="+", type=int, default=[1, 0])
    ap.add_argument("--mb", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--kb", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--nb", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--sb", nargs="+", default=["2x2", "1x4", "2x4", "1x8"])
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    grid = dev.compute_with_storage_grid_size()
    reserve = None
    if args.l1_reserve_mb:
        rows = int(args.l1_reserve_mb * 2**20 / (2048 * 2)) // 32 * 32
        reserve = ttnn.from_torch(
            torch.zeros(1, 1, rows, 2048),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    wdt = getattr(ttnn, args.wdtype)
    ckc = {
        (f, acc): ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=True
        )
        for f in [getattr(ttnn.MathFidelity, n) for n in args.fid]
        for acc in [bool(a) for a in args.acc]
    }
    try:
        for M in args.m:
            for name in args.shapes:
                K, N = SHAPES[name]
                x = ttnn.from_torch(
                    torch.randn(1, 1, M, K), dtype=getattr(ttnn, args.xdtype), layout=ttnn.TILE_LAYOUT, device=dev
                )
                w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
                res = []
                for (f, acc), c in ckc.items():
                    tag = f"{str(f).split('.')[-1]}{'/fp32' if acc else ''}"
                    try:
                        if name in ("ff13s", "ff13w", "score"):
                            raise ValueError("no linear-auto for the fused SwiGLU shape")
                        us = timed(
                            dev,
                            lambda: ttnn.linear(
                                x,
                                w,
                                compute_kernel_config=c,
                                core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
                                dtype=ttnn.bfloat16,
                            ),
                        )
                        res.append((us, f"linear-auto {tag}"))
                        print(f"     .. M={M} {name} {us:8.1f} us linear-auto {tag}", flush=True)
                    except Exception as e:
                        res.append((1e9, f"linear-auto {tag} ERR {str(e)[:60]}"))
                    for mb, kb, nb, (sh, sw) in itertools.product(
                        args.mb, args.kb, args.nb, [tuple(map(int, v.split("x"))) for v in args.sb]
                    ):
                        if sh * sw > (4 if acc else 8) or nb % sw or mb % sh or (K // 32) % kb:
                            continue
                        cfg = ttnn.MinimalMatmulConfig(
                            M_block_size=mb,
                            K_block_size=kb,
                            N_block_size=nb,
                            subblock_h=sh,
                            subblock_w=sw,
                            compute_with_storage_grid_size=grid,
                        )
                        try:
                            us = timed(
                                dev,
                                lambda: ttnn.experimental.minimal_matmul(
                                    x,
                                    w,
                                    config=cfg,
                                    compute_kernel_config=c,
                                    dtype=getattr(ttnn, args.odtype),
                                    fuse_swiglu=name == "ff13s",
                                ),
                            )
                            res.append((us, f"minimal M{mb} K{kb} N{nb} sb{sh}x{sw} {tag}"))
                            print(
                                f"     .. M={M} {name} {us:8.1f} us minimal M{mb} K{kb} N{nb} sb{sh}x{sw} {tag}",
                                flush=True,
                            )
                        except Exception as e:
                            pass
                res.sort()
                flops = 2 * M * K * N
                print(f"== M={M} {name} {K}x{N} ({flops/1e9:.1f} GFLOP)")
                for us, d in res[: args.top]:
                    print(f"   {us:9.1f} us  {flops/us/1e6:7.1f} TFLOPS  {d}")
                for us, d in res:
                    if d.startswith("linear-auto HiFi2/fp32"):
                        print(f"   baseline {us:9.1f} us  {d}")
                ttnn.deallocate(x)
                ttnn.deallocate(w)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
