# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check SDPA's on-device windowed mask (cu_window_seqlens) for packed variable-length sequences.

Packs sequences of the given lengths into one row of S tokens (the tail after the
last sequence is one extra window), runs non-causal windowed SDPA with 32 q /
8 kv heads of 64, and compares each sequence's rows against torch attention over
that sequence alone. Also times windowed vs unmasked SDPA on the same shape, and
checks that repeated (zero-length) boundaries are accepted, so a fixed-size
boundary tensor can describe any number of sequences.
"""

import argparse
import time

import torch

import ttnn


def timed(device, fn, reps=10):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(reps)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    t = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t) / reps * 1e6
    ttnn.release_trace(device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--lens", type=int, nargs="+", default=[300])
    ap.add_argument("--pad-boundaries", type=int, default=0, help="extra repeated boundaries at the end")
    ap.add_argument("--q", type=int, default=128)
    ap.add_argument("--k", type=int, default=256)
    args = ap.parse_args()
    S = args.S
    bounds = [0]
    for n in args.lens:
        bounds.append(bounds[-1] + n)
    assert bounds[-1] <= S
    if bounds[-1] < S:
        bounds.append(S)
    bounds += [S] * args.pad_boundaries
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    try:
        torch.manual_seed(0)
        q, k, v = torch.randn(1, 32, S, 64), torch.randn(1, 8, S, 64), torch.randn(1, 8, S, 64)
        tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for t in (q, k, v))
        cu = ttnn.from_torch(
            torch.tensor(bounds, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            q_chunk_size=args.q,
            k_chunk_size=args.k,
            exp_approx_mode=False,
        )
        win = lambda: ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, cu_window_seqlens=cu, scale=0.125, compute_kernel_config=ckc, program_config=pc
        )
        out = ttnn.to_torch(win()).float()
        kr, vr = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
        worst = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b == a:
                continue
            ref = torch.softmax(q[:, :, a:b] @ kr[:, :, a:b].transpose(-1, -2) * 0.125, dim=-1) @ vr[:, :, a:b]
            worst = max(worst, (out[:, :, a:b] - ref).abs().max().item())
        plain = lambda: ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, scale=0.125, compute_kernel_config=ckc, program_config=pc
        )
        print(f"S={S} windows={bounds} max|err| vs per-sequence torch {worst:.4f}")
        print(f"windowed {timed(dev, win):.1f} us   unmasked {timed(dev, plain):.1f} us")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
