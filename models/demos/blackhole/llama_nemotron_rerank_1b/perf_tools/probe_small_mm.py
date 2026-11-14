# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small-row matmuls (single requests): the model's current minimal_matmul config vs ttnn.linear with a 1D
multicast-in0 program config, where every core reads its own N slice of the weight from DRAM (weights read once,
by all cores in parallel) and the small activation block is multicast to all cores.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_small_mm.py --rows 512 2048

Prints traced time per call and the max abs difference between the two outputs.
"""
import argparse
import itertools
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import mm_block_config

SHAPES = {
    "qkv": (2048, 3072, "HiFi2"),
    "wo": (2048, 2048, "HiFi2"),
    "ff13": (2048, 16384, "HiFi2"),
    "ff2": (8192, 2048, "LoFi"),
}
REPS = 10


def timed(dev, fn):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t) / REPS * 1e6)
    ttnn.release_trace(dev, tid)
    out = ttnn.to_torch(outs[0]).float()
    for o in outs:
        ttnn.deallocate(o)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int, default=[512, 2048])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    grid = dev.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    torch.manual_seed(0)
    for rows, name in itertools.product(args.rows, args.shapes):
        K, N, fid = SHAPES[name]
        f = getattr(ttnn.MathFidelity, fid)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        x = ttnn.from_torch(torch.randn(1, 1, rows, K), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch(torch.randn(K, N) / K**0.5, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        mem = ttnn.L1_MEMORY_CONFIG if rows * N * 2 <= 8_500_000 else ttnn.DRAM_MEMORY_CONFIG
        kind = "ff13" if name == "ff13" else name
        mb, kb, nb, sh, sw = mm_block_config(kind if kind != "qkv" else "qkv", rows)
        ref_us, ref = timed(
            dev,
            lambda: ttnn.experimental.minimal_matmul(
                x,
                w,
                compute_kernel_config=ckc,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=min(kb, 16),
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=grid,
                ),
                dtype=ttnn.bfloat16,
                memory_config=mem,
            ),
        )
        print(f"{rows:5d} rows {name:5s} {K}x{N} {fid}: minimal_matmul {ref_us:7.1f} us", flush=True)
        mt, nt, kt = rows // 32, N // 32, K // 32
        per_core_n = -(-nt // ncores)
        best = None
        for in0_bw in (1, 2, 4, 8):
            if kt % in0_bw:
                continue
            # fp32 accumulation: a subblock holds at most 4 tiles; one tile row, as wide as divides the slice.
            sw1 = max(d for d in range(1, min(per_core_n, 4) + 1) if per_core_n % d == 0)
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=in0_bw,
                out_subblock_h=1,
                out_subblock_w=sw1,
                per_core_M=mt,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            try:
                us, out = timed(
                    dev,
                    lambda: ttnn.linear(
                        x, w, program_config=pc, compute_kernel_config=ckc, dtype=ttnn.bfloat16, memory_config=mem
                    ),
                )
            except Exception as e:
                print(f"      1D mcast in0_block_w {in0_bw}: failed {str(e).splitlines()[0][:120]}", flush=True)
                continue
            diff = (out - ref).abs().max().item()
            print(f"      1D mcast in0_block_w {in0_bw}: {us:7.1f} us (max diff vs minimal {diff:.3f})", flush=True)
            best = min(best or 1e9, us)
        if best:
            print(f"   -> best 1D {best:7.1f} us vs minimal {ref_us:7.1f} us ({ref_us / best:.2f}x)", flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
