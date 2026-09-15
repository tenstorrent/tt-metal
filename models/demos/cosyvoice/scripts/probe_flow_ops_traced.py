# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The flow estimator's per-block cost with dispatch removed -- the only version that can
locate Wormhole's penalty.

`probe_flow_ops.py` timed each block class with a `synchronize_device` around it and found
Wormhole **faster than Blackhole on every one**:

    whole estimator, untraced    Blackhole 85.80 ms    Wormhole 70.80 ms
    transformer @ T=141                    0.707                 0.601
    resnet 256->256 @141                   2.331                 1.925

while the *traced* stage is 0.683 s on Wormhole against 0.375 s on Blackhole -- 1.82x the
other way. Both facts are consistent, and PERF.md already records why for the decode step:
**untraced, this model is host-dispatch-bound and the accelerator barely participates**, so
a per-call wall time measures the host. The architecture gap only exists once tracing
removes dispatch.

So capture each block class in its own trace, replay it, and divide. `REPS` calls per trace
amortise the one `execute_trace` command, so what is left is device time.

Two properties of the measurement worth stating, because they decide what it is good for:

  - the block is called `REPS` times on the **same** input, and each output is deallocated
    inside the traced body. That is a valid graph, and it measures the block rather than a
    chain, which is what a per-class attribution needs.
  - trace capture bakes tensor addresses, so every buffer is pre-allocated before
    `begin_trace_capture` -- the constraint CLAUDE.md sec.7 records.

If every class comes back at roughly the same ratio, there is no per-op target on this
stage and the gap is the core count (130 vs 64 = 2.03x). If one class is far worse than the
rest, that class is the work.

    python3 models/demos/cosyvoice/scripts/probe_flow_ops_traced.py [--reps 16]
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

T0, T1 = 282, 141
BATCH = 2
STEPS = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)  # calls inside one trace
    ap.add_argument("--iters", type=int, default=8)  # replays, best-of
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

        t_in = dev(torch.full((BATCH, 1, 1), 0.5))
        t_emb = est.time_embedding(t_in)

        def traced_ms(body) -> float:
            """ms per single call of `body`, measured inside a replayed trace."""
            for _ in range(2):  # warm the program cache; a compile inside capture is an error
                body()
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                for _ in range(args.reps):
                    body()
            finally:
                ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm the replay
            best = 1e9
            for _ in range(args.iters):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                best = min(best, time.perf_counter() - t0)
            ttnn.release_trace(device, tid)
            return best * 1e3 / args.reps

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  traced replay, {args.reps} calls per trace, best of {args.iters}\n")

        rows = []

        def add(label, count, body):
            try:
                rows.append((label, traced_ms(body), count))
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: capture failed -- {str(exc)[:90]}")

        h0, h1 = act(T0, 256), act(T1, 256)
        tb0, tb1 = est.down[0][1][0], est.mid[0][1][0]
        add("transformer @ T=282", 8, lambda: ttnn.deallocate(tb0(h0)))
        add("transformer @ T=141", 56, lambda: ttnn.deallocate(tb1(h1)))

        for label, blk, tlen, cin, cnt in (
            ("resnet 320->256 @282", est.down[0][0], T0, 320, 1),
            ("resnet 256->256 @141", est.mid[0][0], T1, 256, 13),
            ("resnet 512->256 @141", est.up[0][0], T1, 512, 1),
            ("resnet 512->256 @282", est.up[1][0], T0, 512, 1),
        ):
            a = act(tlen, cin)
            add(label, cnt, lambda a=a, blk=blk, tlen=tlen: ttnn.deallocate(blk(a, t_emb, tlen, BATCH)))

        d0, d1 = act(T0, 256), act(T1, 256)
        add("down conv1d k3 s2 @282", 1, lambda: ttnn.deallocate(est.down[0][2](d0, T0, BATCH)[0]))
        add("down conv1d k3 s1 @141", 1, lambda: ttnn.deallocate(est.down[1][2](d1, T1, BATCH)[0]))
        add("up convT1d s2 @141", 1, lambda: ttnn.deallocate(est.up[0][2](d1, T1, BATCH)[0]))
        add("up conv1d k3 @282", 1, lambda: ttnn.deallocate(est.up[1][2](d0, T0, BATCH)[0]))
        add("final block @282", 1, lambda: ttnn.deallocate(est.final_block(d0, T0, BATCH)))

        # Inside the block that carries 56 of the 64 transformer calls.
        n1 = ttnn.layer_norm(h1, weight=tb1.g1, bias=tb1.b1, epsilon=tb1.eps)
        at = tb1.attn
        qkv = ttnn.linear(n1, at.wqkv, bias=at.bqkv, compute_kernel_config=at.cc)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=at.h, transpose_key=False)
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0, compute_kernel_config=at.cc
        )
        ctx2 = ttnn.transformer.concatenate_heads(ctx)
        ff1 = ttnn.linear(n1, tb1.wf1, bias=tb1.bf1, compute_kernel_config=tb1.cc)

        fine = []

        def addf(label, body):
            try:
                fine.append((label, traced_ms(body)))
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: {str(exc)[:80]}")

        def _split():
            a, b, c = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=at.h, transpose_key=False)
            for t in (a, b, c):
                ttnn.deallocate(t)

        addf("layer_norm", lambda: ttnn.deallocate(ttnn.layer_norm(h1, weight=tb1.g1, bias=tb1.b1, epsilon=tb1.eps)))
        addf(
            "attn: qkv linear",
            lambda: ttnn.deallocate(ttnn.linear(n1, at.wqkv, bias=at.bqkv, compute_kernel_config=at.cc)),
        )
        addf("attn: split heads", _split)
        addf(
            "attn: sdpa",
            lambda: ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=1.0, compute_kernel_config=at.cc
                )
            ),
        )
        addf("attn: concat heads", lambda: ttnn.deallocate(ttnn.transformer.concatenate_heads(ctx)))
        addf(
            "attn: out linear",
            lambda: ttnn.deallocate(ttnn.linear(ctx2, at.wo, bias=at.bo, compute_kernel_config=at.cc)),
        )
        addf(
            "ff: linear 256->1024",
            lambda: ttnn.deallocate(ttnn.linear(n1, tb1.wf1, bias=tb1.bf1, compute_kernel_config=tb1.cc)),
        )
        addf("ff: gelu (exact)", lambda: ttnn.deallocate(ttnn.gelu(ff1, fast_and_approximate_mode=False)))
        addf(
            "ff: linear 1024->256",
            lambda: ttnn.deallocate(ttnn.linear(ff1, tb1.wf2, bias=tb1.bf2, compute_kernel_config=tb1.cc)),
        )

        # Inside a resnet, where GroupNorm was the surprise untraced.
        rblk, a = est.mid[0][0], act(T1, 256)
        b1 = rblk.block1
        cv = b1.conv(a, T1, BATCH)[0]
        addf("resnet: conv1d k3 @141", lambda: ttnn.deallocate(b1.conv(a, T1, BATCH)[0]))
        addf("resnet: groupnorm(8) @141", lambda: ttnn.deallocate(b1.norm(cv)))
        addf("resnet: mish @141", lambda: ttnn.deallocate(ttnn.mish(cv)))
        addf("resnet: res conv1d k1 @141", lambda: ttnn.deallocate(rblk.res(a, T1, BATCH)[0]))

        total = sum(ms * n for _, ms, n in rows)
        print(f"  {'block':<26}{'ms/call':>10}{'x/step':>9}{'ms/step':>10}{'% step':>9}{'s/utt':>9}")
        print("  " + "-" * 73)
        for label, ms, n in sorted(rows, key=lambda r: -r[1] * r[2]):
            per = ms * n
            print(f"  {label:<26}{ms:>10.4f}{n:>9}{per:>10.3f}{100*per/total:>8.1f}%{per*STEPS/1e3:>9.3f}")
        print("  " + "-" * 73)
        print(f"  {'sum of parts':<26}{'':>10}{'':>9}{total:>10.3f}{100.0:>8.1f}%{total*STEPS/1e3:>9.3f}")

        print(f"\n  inside one transformer block @ T=141, and one resnet")
        for label, ms in fine:
            print(f"    {label:<28}{ms:>9.4f} ms")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
