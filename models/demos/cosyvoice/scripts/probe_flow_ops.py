# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which op class is Wormhole's flow penalty in?

The flow decoder is the model's most architecture-penalised stage -- ~1.8x Blackhole on
Wormhole -- and an earlier probe closed off the obvious explanation: it is **not**
parallelism-shaped.
The estimator at batch 2 costs 0.90x batch 1, so its cores are already idle at the size it
runs; splitting the two classifier-free-guidance rows across chips would buy nothing. That
leaves per-op cost, which is what this measures.

The estimator is 16 ResnetBlock1D + 64 BasicTransformerBlock + 6 convolutions per call, ten
Euler steps per utterance. Rather than a flat profile, time each **class** at the exact
shapes the real forward uses and multiply by its count, so the per-class totals reconstruct
the stage and the Wormhole/Blackhole ratio is attributable:

    transformer @ T=282   x  8      down[0], up[1]
    transformer @ T=141   x 56      down[1], mid x12, up[0]
    resnet      @ 4 configs         320->256 @282, 256->256 @141 x13, 512->256 @141, @282
    conv1d / conv_transpose1d       the down/up path
    time_embedding        x  1

Nothing is re-implemented here: the modules are pulled off a built `TtConditionalDecoder`,
so the shapes and the compute-kernel configs are the ones production uses. A class whose
ratio is far above the stage's overall 1.8x is where the Wormhole work is.

The transformer block is broken down further -- layer_norm, fused QKV, SDPA, out-projection,
feed-forward -- because at 64 calls per step it dominates by count, and "attention is slow"
and "the feed-forward is slow" imply completely different fixes.

    python3 models/demos/cosyvoice/scripts/probe_flow_ops.py [--reps 20]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")

T0, T1 = 282, 141  # the captured utterance's lengths, before and after down[0]
BATCH = 2  # classifier-free guidance: conditioned + unconditioned
STEPS = 10  # Euler steps per utterance


def bench(fn, reps: int, device) -> float:
    """Best-of-`reps` wall time for one call, in ms, with the device drained each time.

    Best-of rather than mean: the interesting quantity is the cost of the work, and the
    tail here is host scheduling noise, not device variance. Idle `synchronize_device`
    measures ~0.012 ms on both parts, so a ~1 ms block is distorted by ~1%.
    """
    out = fn()
    if out is not None:
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
        if out is not None:
            ttnn.deallocate(out)
    return best * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    from models.demos.cosyvoice.tt.flow.estimator import TtConditionalDecoder
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(os.path.join(GOLDEN, "flow_weights.npz")).sub("decoder").sub("estimator")
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        est = TtConditionalDecoder(device, bag)
        torch.manual_seed(0)

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def act(t, c):
            return dev(torch.randn(BATCH, t, c) * 0.1)

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  estimator, batch {BATCH}, T {T0} -> {T1}, best of {args.reps}\n")

        rows = []  # (label, ms_per_call, count_per_step)

        # ---- the whole thing, for the reconciliation at the bottom --------------
        x, mu = act(T0, 80), act(T0, 80)
        t_in = dev(torch.full((BATCH, 1, 1), 0.5))
        spks, cond = dev(torch.randn(BATCH, 1, 80) * 0.1), act(T0, 80)
        whole = bench(lambda: est(x, mu, t_in, spks=spks, cond=cond, batch=BATCH), args.reps, device)

        # ---- transformer blocks -------------------------------------------------
        tb0 = est.down[0][1][0]  # T=282
        tb1 = est.mid[0][1][0]  # T=141
        h0, h1 = act(T0, 256), act(T1, 256)
        rows.append(("transformer @ T=282", bench(lambda: tb0(h0), args.reps, device), 8))
        rows.append(("transformer @ T=141", bench(lambda: tb1(h1), args.reps, device), 56))

        # ---- inside one transformer block, at the shape 56 of the 64 blocks use --
        fine = []
        n1 = ttnn.layer_norm(h1, weight=tb1.g1, bias=tb1.b1, epsilon=tb1.eps)
        fine.append(
            (
                "layer_norm",
                bench(lambda: ttnn.layer_norm(h1, weight=tb1.g1, bias=tb1.b1, epsilon=tb1.eps), args.reps, device),
            )
        )
        at = tb1.attn
        fine.append(
            (
                "attn: qkv linear",
                bench(lambda: ttnn.linear(n1, at.wqkv, bias=at.bqkv, compute_kernel_config=at.cc), args.reps, device),
            )
        )
        qkv = ttnn.linear(n1, at.wqkv, bias=at.bqkv, compute_kernel_config=at.cc)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=at.h, transpose_key=False)

        def _split():
            a, b, c = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=at.h, transpose_key=False)
            ttnn.deallocate(b)
            ttnn.deallocate(c)
            return a

        fine.append(("attn: split heads", bench(_split, args.reps, device)))
        fine.append(
            (
                "attn: sdpa",
                bench(
                    lambda: ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, is_causal=False, scale=1.0, compute_kernel_config=at.cc
                    ),
                    args.reps,
                    device,
                ),
            )
        )
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0, compute_kernel_config=at.cc
        )
        fine.append(("attn: concat heads", bench(lambda: ttnn.transformer.concatenate_heads(ctx), args.reps, device)))
        ctx2 = ttnn.transformer.concatenate_heads(ctx)
        fine.append(
            (
                "attn: out linear",
                bench(lambda: ttnn.linear(ctx2, at.wo, bias=at.bo, compute_kernel_config=at.cc), args.reps, device),
            )
        )
        fine.append(
            (
                "ff: linear 256->1024",
                bench(lambda: ttnn.linear(n1, tb1.wf1, bias=tb1.bf1, compute_kernel_config=tb1.cc), args.reps, device),
            )
        )
        ff1 = ttnn.linear(n1, tb1.wf1, bias=tb1.bf1, compute_kernel_config=tb1.cc)
        fine.append(
            ("ff: gelu (exact)", bench(lambda: ttnn.gelu(ff1, fast_and_approximate_mode=False), args.reps, device))
        )
        fine.append(
            (
                "ff: linear 1024->256",
                bench(lambda: ttnn.linear(ff1, tb1.wf2, bias=tb1.bf2, compute_kernel_config=tb1.cc), args.reps, device),
            )
        )
        for t in (n1, qkv, q, k, v, ctx, ctx2, ff1):
            ttnn.deallocate(t)

        # ---- resnet blocks ------------------------------------------------------
        t_emb = est.time_embedding(t_in)
        for label, blk, tlen, cin, cnt in (
            ("resnet 320->256 @282", est.down[0][0], T0, 320, 1),
            ("resnet 256->256 @141", est.mid[0][0], T1, 256, 13),
            ("resnet 512->256 @141", est.up[0][0], T1, 512, 1),
            ("resnet 512->256 @282", est.up[1][0], T0, 512, 1),
        ):
            a = act(tlen, cin)
            rows.append(
                (label, bench(lambda a=a, blk=blk, tlen=tlen: blk(a, t_emb, tlen, BATCH), args.reps, device), cnt)
            )
            ttnn.deallocate(a)

        # ---- inside one resnet: which of conv / groupnorm / mish costs ----------
        rblk = est.mid[0][0]
        a = act(T1, 256)
        b1 = rblk.block1
        fine2 = [
            ("conv1d k3 256->256 @141", bench(lambda: b1.conv(a, T1, BATCH)[0], args.reps, device)),
        ]
        cv = b1.conv(a, T1, BATCH)[0]
        fine2.append(("groupnorm(8) @141", bench(lambda: b1.norm(cv), args.reps, device)))
        gn = b1.norm(cv)
        fine2.append(("mish @141", bench(lambda: ttnn.mish(gn), args.reps, device)))
        fine2.append(("res conv1d k1 @141", bench(lambda: rblk.res(a, T1, BATCH)[0], args.reps, device)))
        for t in (cv, gn, a):
            ttnn.deallocate(t)

        # ---- the down/up convolutions ------------------------------------------
        d0 = act(T0, 256)
        rows.append(("down conv1d k3 s2 @282", bench(lambda: est.down[0][2](d0, T0, BATCH)[0], args.reps, device), 1))
        d1 = act(T1, 256)
        rows.append(("down conv1d k3 s1 @141", bench(lambda: est.down[1][2](d1, T1, BATCH)[0], args.reps, device), 1))
        rows.append(("up convT1d s2 @141", bench(lambda: est.up[0][2](d1, T1, BATCH)[0], args.reps, device), 1))
        rows.append(("up conv1d k3 @282", bench(lambda: est.up[1][2](d0, T0, BATCH)[0], args.reps, device), 1))
        rows.append(("final block @282", bench(lambda: est.final_block(d0, T0, BATCH), args.reps, device), 1))
        fb = est.final_block(d0, T0, BATCH)
        rows.append(("final proj k1 @282", bench(lambda: est.final_proj(fb, T0, BATCH)[0], args.reps, device), 1))
        rows.append(("time_embedding", bench(lambda: est.time_embedding(t_in), args.reps, device), 1))

        # ---- report -------------------------------------------------------------
        print(f"  {'block':<26}{'ms/call':>10}{'x/step':>9}{'ms/step':>10}{'% step':>9}{'s/utt':>9}")
        print("  " + "-" * 73)
        total = sum(ms * n for _, ms, n in rows)
        for label, ms, n in sorted(rows, key=lambda r: -r[1] * r[2]):
            per_step = ms * n
            print(
                f"  {label:<26}{ms:>10.3f}{n:>9}{per_step:>10.3f}{100*per_step/total:>8.1f}%{per_step*STEPS/1e3:>9.3f}"
            )
        print("  " + "-" * 73)
        print(f"  {'sum of parts':<26}{'':>10}{'':>9}{total:>10.3f}{100.0:>8.1f}%{total*STEPS/1e3:>9.3f}")
        print(f"  {'measured whole estimator':<26}{whole:>10.3f}{1:>9}{whole:>10.3f}{'':>9}{whole*STEPS/1e3:>9.3f}")
        print(f"  (parts/whole = {total/whole:.2f}x -- above 1.0 is per-call sync and launch overhead)")

        print(f"\n  inside one transformer block @ T=141 (x56 per step, x{56*STEPS} per utterance)")
        tsum = sum(v for _, v in fine)
        for label, ms in fine:
            print(f"    {label:<24}{ms:>9.4f} ms{100*ms/tsum:>8.1f}%   {ms*56*STEPS/1e3:>7.3f} s/utt")
        print(f"    {'sum':<24}{tsum:>9.4f} ms{100.0:>8.1f}%   {tsum*56*STEPS/1e3:>7.3f} s/utt")

        print(f"\n  inside one resnet block @ T=141 (2 blocks of conv+norm+mish each)")
        for label, ms in fine2:
            print(f"    {label:<24}{ms:>9.4f} ms")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
