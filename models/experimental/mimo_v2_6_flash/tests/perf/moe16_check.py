# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The 16-user MoE decode (FusedStream with two row halves, moe_stream.py) against the 8-user one on Galaxy row 0,
one layer, BFP4 experts, real activations (moe_decode_common).

Correctness: every 16-slot step's rows 0-7 and 8-15 must match the 8-user layer run on those 8 rows alone (the same
experts in the same order, the same arithmetic: PCC >= 0.99999 per active row, every output row finite). Shapes:
four / three / one requests of 4 consecutive tokens (groups of 4 slots), half 1 alone, eight requests of 2 tokens,
16 distinct tokens.

Timing (``--bench``): one trace per layer replayed with each shape's tokens, as kagent_moe_verify_bench.py, for the
16-user shapes and their 8-slot counterparts (the K = 1 steps of 3 and 4 requests, the K = 3 step of 2).

    MOE16 <shape> us=<median per call>

Prints MOE16_CORRECTNESS_PASS when every check passed.
"""

import argparse
import statistics
import time

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common

CALLS = 20
REPLAYS = 7


def windows(cfg):
    starts, lengths = common.prompt_starts(cfg)
    return starts, lengths, [(p, o) for p, n in enumerate(lengths) for o in range(0, n - 3, 4)]


def steps16(cfg, steps):
    """(name, row indices [16] (-1 idle)) of 16-slot steps, and (name, [8]) of the 8-slot counterparts."""
    starts, lengths, wins = windows(cfg)
    out16, out8 = [], []
    for s in range(steps):
        reqs = []
        for r in range(8):
            p, o = wins[(s * 7 + r * 11 + 3) % len(wins)]
            reqs.append([starts[p] + o + j for j in range(4)])
        idle4 = [-1] * 4
        out16.append(("four_requests", reqs[0] + reqs[1] + reqs[2] + reqs[3]))
        out16.append(("three_requests", reqs[0] + reqs[1] + reqs[2] + idle4))
        out16.append(("one_request", reqs[0] + idle4 * 3))
        out16.append(("half1_only", idle4 * 2 + reqs[3] + idle4))
        out16.append(("eight_requests_k1", [i for r in reqs for i in r[:2]]))
        distinct = [starts[p] + (3 * s + 5 * j + 1) % lengths[p] for j, p in enumerate((s + k) % 8 for k in range(16))]
        out16.append(("plain16", distinct))
        out8.append(("two_requests_k3", reqs[0] + reqs[1]))
        out8.append(("four_requests_k1", [i for r in reqs[:4] for i in r[:2]]))
        out8.append(("three_requests_k1", [i for r in reqs[:3] for i in r[:2]] + [-1, -1]))
    return out16, out8


def run(row, moe, inputs, idx):
    x, mask = common.step_inputs(row, inputs, idx)
    out = moe.forward_decode(x, mask)
    full = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0]
    for t in (x, mask, out):
        t.deallocate(True)
    return full


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def bench(row, moe, inputs, shapes):
    """Median us per forward_decode of each (name, idx), one trace replayed with each shape's inputs."""
    name0, idx0 = shapes[0]
    x, mask = common.step_inputs(row, inputs, idx0)
    host = {name: common.step_inputs(row, inputs, idx, host=True) for name, idx in shapes}
    for _ in range(2):
        moe.forward_decode(x, mask).deallocate(True)
    ttnn.synchronize_device(row)
    trace_id = ttnn.begin_trace_capture(row, cq_id=0)
    outs = [moe.forward_decode(x, mask) for _ in range(CALLS)]
    ttnn.end_trace_capture(row, trace_id, cq_id=0)
    ttnn.synchronize_device(row)
    times = {name: [] for name, _ in shapes}
    for _ in range(REPLAYS):
        for name, _ in shapes:
            ttnn.copy_host_to_device_tensor(host[name][0], x)
            ttnn.copy_host_to_device_tensor(host[name][1], mask)
            ttnn.synchronize_device(row)
            t0 = time.perf_counter()
            ttnn.execute_trace(row, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(row)
            times[name].append((time.perf_counter() - t0) / CALLS * 1e6)
    ttnn.release_trace(row, trace_id)
    for t in outs + [x, mask]:
        t.deallocate(True)
    return {name: statistics.median(v[1:]) for name, v in times.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    cfg = common.config()
    inputs = common.golden_moe_inputs(cfg, common.checkpoint(cfg))
    s16, s8 = steps16(cfg, args.steps)
    mesh, row = common.open_row(trace_region_size=128 * 1024 * 1024 if args.bench else 0)
    ok = True
    try:
        moe16 = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b, users=16)
        assert moe16.fused_stream is not None and moe16.fused_stream.halves == 2
        moe8 = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b, users=8)
        worst = {}
        for name, idx in s16:
            got = run(row, moe16, inputs, idx)
            finite = bool(torch.isfinite(got).all())
            want = torch.zeros_like(got)
            for h in range(2):
                half = idx[8 * h : 8 * h + 8]
                if any(i >= 0 for i in half):
                    want[8 * h : 8 * h + 8] = run(row, moe8, inputs, half)[:8]
            live = [s for s, i in enumerate(idx) if i >= 0]
            p = min(pcc(got[s], want[s]) for s in live)
            exact = torch.equal(got[live], want[live])
            worst[name] = min(worst.get(name, 1.0), p)
            ok &= p >= 0.99999 and finite
            print(f"{name}: worst row PCC {p:.7f}, bit-exact {exact}, finite {finite}", flush=True)
        if args.bench:
            t16 = bench(row, moe16, inputs, [(n, i) for n, i in s16[:6]])
            t8 = bench(row, moe8, inputs, [(n, i) for n, i in s8[:3]])
            for name, us in list(t16.items()) + list(t8.items()):
                print(f"MOE16 {name} us={us:.2f}", flush=True)
    finally:
        common.close(mesh, row)
    if ok:
        print("MOE16_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("16-user MoE check failed")


if __name__ == "__main__":
    main()
