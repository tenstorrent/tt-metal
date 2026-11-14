# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmasked bidirectional SDPA (32 q / 8 kv heads of 64) time per Q / K / V dtype; the output takes Q's dtype.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_sdpa_dtype.py --shape 32x512
"""
import argparse
import itertools
import time

import torch

import ttnn

REPS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="32x512")
    ap.add_argument("--q-chunk", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--fid", default="HiFi4")
    ap.add_argument("--once", action="store_true", help="one untraced call per config (device profiling)")
    args = ap.parse_args()
    B, S = map(int, args.shape.split("x"))
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    grid = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    th = {n: torch.randn(B, h, S, 64) for n, h in (("q", 32), ("k", 8), ("v", 8))}
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    for qdt, kvdt, qc in itertools.product(("bfloat16", "bfloat8_b"), ("bfloat16", "bfloat8_b"), args.q_chunk):
        t = {
            n: ttnn.from_torch(x, dtype=getattr(ttnn, qdt if n == "q" else kvdt), layout=ttnn.TILE_LAYOUT, device=dev)
            for n, x in th.items()
        }
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=qc,
            k_chunk_size=S,
            matmul_math_fidelity=getattr(ttnn.MathFidelity, args.fid),
            max_cores_per_head_batch=16,
        )
        fn = lambda: ttnn.transformer.scaled_dot_product_attention(
            t["q"],
            t["k"],
            t["v"],
            is_causal=False,
            scale=0.125,
            compute_kernel_config=ckc,
            program_config=pc,
            output_concat_heads=True,
        )
        if args.once:
            ttnn.synchronize_device(dev)
            ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            print(f"q {qdt} kv {kvdt} q_chunk {qc}: ran once", flush=True)
            for x in t.values():
                ttnn.deallocate(x)
            continue
        fn()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        outs = [fn() for _ in range(REPS)]
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            best = min(best, (time.perf_counter() - t0) / REPS * 1e6)
        ttnn.release_trace(dev, tid)
        print(f"q {qdt} kv {kvdt} q_chunk {qc}: {best:.1f} us (out {outs[0].dtype})", flush=True)
        for o in outs:
            ttnn.deallocate(o)
        for x in t.values():
            ttnn.deallocate(x)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
