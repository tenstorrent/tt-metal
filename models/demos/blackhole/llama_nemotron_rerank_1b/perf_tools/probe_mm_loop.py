# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The math thread's matmul ceiling: minimal_matmul's compute loop on tiles already in L1 (no data
movement, no CB handshakes) on every core, as % of the FPU peak (4096 FLOP/cycle LoFi, /2 HiFi2, /4 HiFi4).

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_mm_loop.py --block 4,16,8,1,8 --fid HiFi2 --bf16-acc
"""
import argparse
import os
import time

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops import _mm_grid as mg

KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "mm_loop_compute.cpp")
PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", nargs="+", default=["4,16,8,1,8"], help="M_block,K_block,N_block,sh,sw")
    ap.add_argument("--fid", nargs="+", default=["HiFi2"])
    ap.add_argument("--bf16-acc", action="store_true")
    ap.add_argument("--dtype", default="bfloat8_b", help="in0 / in1 dtype")
    ap.add_argument("--cycles", type=float, default=20e6, help="ideal cycles per run")
    ap.add_argument("--mhz", type=float, default=1350)
    ap.add_argument("--keep-dst", action="store_true", help="accumulate all reps in DST (no per-K-block pack)")
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=10_000_000)
    grid = dev.compute_with_storage_grid_size()
    cs = mg.cores(0, 0, grid.x - 1, grid.y - 1)
    dt = getattr(ttnn, args.dtype)
    dummy = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
    for blk in args.block:
        mb, kb, nb, sh, sw = map(int, blk.split(","))
        for fid in args.fid:
            per_rep = mb * kb * nb * 16 * PASSES[fid]  # ideal cycles (16 per tile-mult at LoFi)
            reps = max(1, int(args.cycles / per_rep))
            cbs = [
                mg.cb(0, mb * kb, dt, cs),
                mg.cb(1, kb * nb, dt, cs),
                mg.cb(3, mb * nb, ttnn.bfloat16 if args.bf16_acc else ttnn.float32, cs),
            ]
            kern = ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cs,
                compile_time_args=[reps, mb, kb, nb, sh, sw],
                runtime_args=[],
                defines=[] if args.keep_dst else [("PACK_EVERY_K", "1")],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=getattr(ttnn.MathFidelity, fid),
                    math_approx_mode=False,
                    fp32_dest_acc_en=not args.bf16_acc,
                ),
            )
            prog = ttnn.ProgramDescriptor(kernels=[kern], cbs=cbs, semaphores=[])
            run = lambda: ttnn.generic_op([dummy, dummy], prog)
            run()
            ttnn.synchronize_device(dev)
            best = 1e9
            for _ in range(5):
                t = time.perf_counter()
                run()
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t)
            ideal_s = reps * per_rep / (args.mhz * 1e6)
            print(
                f"block {blk} {fid} {'bf16' if args.bf16_acc else 'fp32'} acc {'keep-dst' if args.keep_dst else 'pack/K'}: "
                f"{best*1e3:.2f} ms vs ideal {ideal_s*1e3:.2f} ms -> {100*ideal_s/best:.1f}% of peak @ {args.mhz:.0f} MHz",
                flush=True,
            )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
