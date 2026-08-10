# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time benchmark for the ops touched by the ReducePartialScaler migration.

Run on the migration branch and on main, then diff the two JSON outputs:

    python tests/ttnn/unit_tests/operations/moreh/bench_reduce_partial_scaler.py out.json

Measures steady-state wall time for a batch of enqueued ops with a single synchronize at the end,
so per-call host overhead is amortised and the number reflects device work plus dispatch.

Each case is run in both a tile-aligned and a ragged shape. The migration only changes the ragged
path's structure (one reduce instead of two plus a mask), so the aligned rows act as a control: they
should move together with any measurement noise, while a real regression or win should show up as a
divergence between the ragged and aligned columns.
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


def main(out_path):
    device = ttnn.open_device(device_id=0)
    results = {}
    try:
        S = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy

        cases = []

        # moreh_sum / moreh_mean over H (the migrated moreh_sum_h / moreh_mean_h kernels).
        for label, shape in [("aligned_1024", [4, 4, 1024, 1024]), ("ragged_1023", [4, 4, 1023, 1024])]:
            x = _mk(device, shape)
            cases.append((f"moreh_sum_h.{label}", lambda x=x: ttnn.operations.moreh.sum(x, 2, keepdim=True)))
            cases.append((f"moreh_mean_h.{label}", lambda x=x: ttnn.operations.moreh.mean(x, dim=2, keepdim=True)))

        # moreh_softmax SMALL_H / SMALL_W (the migrated max phase).
        for label, shape in [("aligned_512", [4, 4, 512, 512]), ("ragged_511", [4, 4, 511, 512])]:
            x = _mk(device, shape)
            cases.append(
                (f"softmax_small_h.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 2, strategy=S.SMALL_H))
            )
        for label, shape in [("aligned_512", [4, 4, 512, 512]), ("ragged_511", [4, 4, 512, 511])]:
            x = _mk(device, shape)
            cases.append(
                (f"softmax_small_w.{label}", lambda x=x: ttnn.operations.moreh.softmax(x, 3, strategy=S.SMALL_W))
            )

        # layernorm small kernel (layernorm.cpp): E[x] and Var[x] reduces.
        for label, w in [("aligned_4096", 4096), ("ragged_4095", 4095)]:
            x = _mk(device, [1, 4, 1024, w])
            cases.append((f"layernorm.{label}", lambda x=x: ttnn.layer_norm(x)))

        # layernorm with weight+bias -> exercises the same reduces with gamma/beta streaming.
        for label, w in [("aligned_4096", 4096), ("ragged_4095", 4095)]:
            x = _mk(device, [1, 4, 1024, w])
            g = _mk(device, [1, 1, 32, w])
            b = _mk(device, [1, 1, 32, w])
            cases.append((f"layernorm_wb.{label}", lambda x=x, g=g, b=b: ttnn.layer_norm(x, weight=g, bias=b)))

        # layernorm with a residual input -> FUSE_PRE_ADD, i.e. the reduce_multi_input path
        # (large-tensor kernel) or the materialised pre-add (small kernel), depending on size.
        for label, w in [("aligned_4096", 4096), ("ragged_4095", 4095)]:
            x = _mk(device, [1, 4, 1024, w])
            r = _mk(device, [1, 4, 1024, w])
            cases.append((f"layernorm_residual.{label}", lambda x=x, r=r: ttnn.layer_norm(x, residual_input_tensor=r)))

        for name, fn in cases:
            secs = _bench(fn, device)
            results[name] = secs * 1e6  # microseconds per op
            print(f"{name:34s} {results[name]:10.2f} us")
    finally:
        ttnn.close_device(device)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench.json")
