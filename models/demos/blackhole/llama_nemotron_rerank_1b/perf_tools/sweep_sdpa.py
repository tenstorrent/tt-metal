# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sweep bidirectional SDPA configs for the reranker's attention shape (32 q heads, 8 kv heads, head_dim 64).

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/sweep_sdpa.py --shapes 1x512 32x512

Per (batch, seq): q/k chunk sizes, fidelity, exp approximation, with the
[B, 1, S, S] padding mask and without it. Prints time per call (traced, 10
reps) and the max abs difference against the HiFi4 / no-approx / masked output.
"""

import argparse
import itertools
import time

import torch

import ttnn

REPS = 10


def timed(device, fn):
    out = fn()
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
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["1x512", "32x512"])
    ap.add_argument("--chunks", nargs="+", type=int, default=[64, 128, 256, 512])
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    grid = dev.compute_with_storage_grid_size()
    try:
        for shape in args.shapes:
            B, S = map(int, shape.split("x"))
            torch.manual_seed(0)
            mk = lambda h, dt=ttnn.bfloat16: ttnn.from_torch(
                torch.randn(B, h, S, 64), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev
            )
            q, k, v = mk(32), mk(8), mk(8)
            m = torch.zeros(B, 1, S, S)
            m[..., S - S // 4 :] = -1e9
            mask = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ref = None
            rows = []
            for fid, approx, use_mask, qc, kc in itertools.product(
                ("HiFi4", "HiFi2", "LoFi"), (False, True), (True, False), args.chunks, args.chunks
            ):
                if qc > S or kc > S:
                    continue
                ckc = ttnn.init_device_compute_kernel_config(
                    dev.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, fid),
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
                pc = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=approx
                )
                fn = lambda: ttnn.transformer.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=False,
                    attn_mask=mask if use_mask else None,
                    scale=0.125,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )
                try:
                    us, out = timed(dev, fn)
                except Exception as e:
                    print(f"   {shape} {fid} approx={approx} mask={use_mask} q{qc} k{kc} ERR {str(e)[:80]}", flush=True)
                    continue
                o = ttnn.to_torch(out).float()
                ttnn.deallocate(out)
                if ref is None and use_mask:
                    ref = o
                diff = (o - ref).abs().max().item() if (ref is not None and use_mask) else float("nan")
                rows.append((us, f"{fid} approx={approx} mask={use_mask} q{qc} k{kc} maxdiff={diff:.3g}"))
                print(f"   .. {shape} {us:9.1f} us {rows[-1][1]}", flush=True)
            rows.sort()
            print(f"== {shape} best:")
            for us, d in rows[:8]:
                print(f"   {us:9.1f} us  {d}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
