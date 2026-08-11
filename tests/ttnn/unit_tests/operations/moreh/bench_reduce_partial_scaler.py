# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time benchmark for every op the ReducePartialScaler migration touches.

Build BOTH sides with --disable-profiler (Tracy is on by default and its dispatch overhead swamps the
few-percent effects this measures), then diff the two JSON outputs:

    python tests/ttnn/unit_tests/operations/moreh/bench_reduce_partial_scaler.py out.json

The case list is derived from the changed-file set; see
docs/reduce_partial_scaler_migration/step-13-perf-scope-analysis.md for the mapping from files to cases
and for the baseline choice (the branch tip before the work, not main).

Every case is run in a tile-aligned and a ragged variant, because which one is interesting depends on
the change: the conditional-scaler work (step 7a) only removes work on ALIGNED shapes, while the
restructured max phase (step 9) and the backward mask removal (step 8) act on the ragged ones. Cases
prefixed CTL are untouched code and read out the noise floor.
"""

import json
import sys
import time

import torch
import ttnn

WARMUP = 3
ITERS = 20
REPEATS = 5


def _bench(fn, device):
    """Return the best (minimum) mean-per-iteration seconds over REPEATS batches."""
    for _ in range(WARMUP):
        fn()
    ttnn.synchronize_device(device)

    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        ttnn.synchronize_device(device)
        best = min(best, (time.perf_counter() - t0) / ITERS)
    return best


def _mk(device, shape):
    t = torch.rand(size=shape, dtype=torch.bfloat16)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _pairs(h, w, ragged_axis, batch=(4, 4)):
    """(label, shape) for an aligned and a ragged variant along ragged_axis ('h' or 'w')."""
    aligned = [*batch, h, w]
    ragged = [*batch, h - 1, w] if ragged_axis == "h" else [*batch, h, w - 1]
    return [(f"aligned_{h if ragged_axis == 'h' else w}", aligned), ("ragged", ragged)]


def build_cases(device):
    S = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy
    SB = ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy
    cases = []

    # A1: moreh softmax SMALL - step 7a, aligned shapes stop emitting/selecting a 2nd scaler tile.
    for label, shape in _pairs(512, 512, "h"):
        x = _mk(device, shape)
        cases.append(
            (f"A1.softmax_small_h.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 2, strategy=S.SMALL_H))
        )
    for label, shape in _pairs(512, 512, "w"):
        x = _mk(device, shape)
        cases.append(
            (f"A1.softmax_small_w.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 3, strategy=S.SMALL_W))
        )

    # A2: moreh softmax LARGE - step 9, the max phase is one streaming reduce now.
    for label, shape in _pairs(512, 512, "h"):
        x = _mk(device, shape)
        cases.append(
            (f"A2.softmax_large_h.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 2, strategy=S.LARGE_H))
        )
    for label, shape in _pairs(512, 512, "w"):
        x = _mk(device, shape)
        cases.append(
            (f"A2.softmax_large_w.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 3, strategy=S.LARGE_W))
        )

    # A3/A4: the SOFTMIN and LOG arms of those same kernels.
    for label, shape in _pairs(512, 512, "w"):
        x = _mk(device, shape)
        cases.append(
            (f"A3.softmin_small_w.{label}", lambda x=x: ttnn.operations.moreh.softmin(x, 3, strategy=S.SMALL_W))
        )
        cases.append(
            (f"A4.logsoftmax_small_w.{label}", lambda x=x: ttnn.operations.moreh.logsoftmax(x, 3, strategy=S.SMALL_W))
        )
        cases.append(
            (f"A4.logsoftmax_large_w.{label}", lambda x=x: ttnn.operations.moreh.logsoftmax(x, 3, strategy=S.LARGE_W))
        )

    # A5/A6: moreh softmax_backward SMALL - step 8 removed the mask, the fold and the Ht==1 branch.
    # CTL.softmax_backward_large_w is baseline code (step 8's _large attempt was reverted).
    for label, shape in _pairs(512, 512, "h"):
        y, dy = _mk(device, shape), _mk(device, shape)
        cases.append(
            (
                f"A5.softmax_backward_small_h.{label}",
                lambda y=y, dy=dy: ttnn.operations.moreh.softmax_backward(y, dy, 2, strategy=SB.SMALL_H),
            )
        )
    for label, shape in _pairs(512, 512, "w"):
        y, dy = _mk(device, shape), _mk(device, shape)
        cases.append(
            (
                f"A5.softmax_backward_small_w.{label}",
                lambda y=y, dy=dy: ttnn.operations.moreh.softmax_backward(y, dy, 3, strategy=SB.SMALL_W),
            )
        )
        cases.append(
            (
                f"A6.logsoftmax_backward_small_w.{label}",
                lambda y=y, dy=dy: ttnn.operations.moreh.logsoftmax_backward(y, dy, 3, strategy=SB.SMALL_W),
            )
        )
        cases.append(
            (
                f"CTL.softmax_backward_large_w.{label}",
                lambda y=y, dy=dy: ttnn.operations.moreh.softmax_backward(y, dy, 3, strategy=SB.LARGE_W),
            )
        )

    # B1: ttnn.softmax general small, which shares A1's kernels. rank 3 keeps it off the rank-4
    # attention factory, so dim=-1 lands on GeneralWSmall and dim=-2 on GeneralHSmall.
    for label, shape in [("aligned_512", [16, 512, 512]), ("ragged", [16, 512, 511])]:
        x = _mk(device, shape)
        cases.append((f"B1.ttnn_softmax_general_w_small.{label}", lambda x=x: ttnn.softmax(x, dim=-1)))
    for label, shape in [("aligned_512", [16, 512, 512]), ("ragged", [16, 511, 512])]:
        x = _mk(device, shape)
        cases.append((f"B1.ttnn_softmax_general_h_small.{label}", lambda x=x: ttnn.softmax(x, dim=-2)))

    # B2: ttnn.softmax general large, sharing A2's kernels. W is big enough that the L1 estimate
    # rejects *Small.
    for label, shape in [("aligned_8192", [2, 256, 8192]), ("ragged", [2, 256, 8191])]:
        x = _mk(device, shape)
        cases.append((f"B2.ttnn_softmax_general_w_large.{label}", lambda x=x: ttnn.softmax(x, dim=-1)))

    # B3: near the L1-fit boundary, which this work moved by one to two scaler tiles.
    for label, shape in [("aligned_2048", [4, 256, 2048]), ("ragged", [4, 256, 2047])]:
        x = _mk(device, shape)
        cases.append((f"B3.ttnn_softmax_boundary_w.{label}", lambda x=x: ttnn.softmax(x, dim=-1)))

    # C: moreh sum/mean over H - runtime-arg removal only, expected exactly flat.
    for label, shape in _pairs(1024, 1024, "h"):
        x = _mk(device, shape)
        cases.append((f"C.moreh_sum_h.{label}", lambda x=x: ttnn.operations.moreh.sum(x, 2, keepdim=True)))
        cases.append((f"C.moreh_mean_h.{label}", lambda x=x: ttnn.operations.moreh.mean(x, dim=2, keepdim=True)))

    # D: bias grad over H - step 10. Ragged H spanning three tiles is the configuration that changed.
    # Only the bias grad is requested, so the measurement isolates the kernel this work touched.
    for label, h in [("aligned_96", 96), ("ragged", 95)]:
        inp = _mk(device, [h, 512])
        wt = _mk(device, [1024, 512])
        out_grad = _mk(device, [h, 1024])
        bias = _mk(device, [1, 1024])
        bias_grad = _mk(device, [1, 1024])
        cases.append(
            (
                f"D.bias_backward_h.{label}",
                lambda o=out_grad, i=inp, w=wt, b=bias, bg=bias_grad: ttnn.operations.moreh.linear_backward(
                    o, i, w, are_required_outputs=(False, False, True), bias=b, bias_grad=bg
                ),
            )
        )

    # E: topk_router_gpt - step 7d added an unpacker + packer reconfig. B=32 / N=128 are op requirements.
    for label, K in [("K2880", 2880), ("K4096", 4096)]:
        ti = _mk(device, [32, K])
        tw = _mk(device, [K, 128])
        tb = _mk(device, [32, 128])
        cases.append(
            (
                f"E.topk_router_gpt.{label}",
                lambda ti=ti, tw=tw, tb=tb: ttnn.experimental.topk_router_gpt(
                    ti, weight_tensor=tw, bias_tensor=tb, k=4, num_experts=128
                ),
            )
        )

    # CTL: untouched code paths. If these move, the run is noise.
    for label, w in [("aligned_4096", 4096), ("ragged", 4095)]:
        x = _mk(device, [1, 4, 1024, w])
        cases.append((f"CTL.layernorm.{label}", lambda x=x: ttnn.layer_norm(x)))
    for label, shape in [("aligned_512", [4, 8, 512, 512]), ("ragged", [4, 8, 511, 512])]:
        x = _mk(device, shape)
        cases.append(
            (f"CTL.softmax_large_c.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 1, strategy=S.LARGE_C))
        )

    return cases


def main(out_path, exclude=None):
    """exclude: regex; matching cases are skipped.

    Needed to measure a baseline that predates the step 7b fix: there, the shared softmax reader emits
    two max-scaler tiles unconditionally while the ttnn general *small* factories size that CB at one
    tile, so every B1/B3 case deadlocks the device (and needs tt-smi -r to clear). Exclude them rather
    than lose the rest of the sweep -- and report them as "baseline hangs", which is the honest delta.
    """
    import re

    skip_re = re.compile(exclude) if exclude else None
    device = ttnn.open_device(device_id=0)
    results = {}
    try:
        for name, fn in build_cases(device):
            if skip_re is not None and skip_re.search(name):
                print(f"{name:46s} EXCLUDED by filter", flush=True)
                continue
            try:
                secs = _bench(fn, device)
            except Exception as exc:  # one unrunnable case should not lose the whole sweep
                print(f"{name:46s} SKIPPED ({type(exc).__name__}: {str(exc)[:70]})", flush=True)
                continue
            results[name] = secs * 1e6  # microseconds per op
            print(f"{name:46s} {results[name]:10.2f} us", flush=True)
    finally:
        ttnn.close_device(device)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "bench.json",
        exclude=sys.argv[2] if len(sys.argv) > 2 else None,
    )
