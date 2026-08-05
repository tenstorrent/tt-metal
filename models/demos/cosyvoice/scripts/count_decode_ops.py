# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Count the TTNN ops in one AR decode step, by name.

Once tracing has removed host dispatch and `bfloat8_b` weights have shown the step
is nowhere near bandwidth-bound, what is left is per-op cost on one-row tensors --
so the next optimisation is whichever op the step issues most of. That is a
countable fact, not something to estimate from reading the code, and reading the
code gets it wrong: the projections look like the bulk and are not.

Counting needs no device time. `ttnn` functions are module attributes, so wrapping
them with a counter and running one decode body is enough.

    python models/demos/cosyvoice/scripts/count_decode_ops.py
"""
from __future__ import annotations

import collections
import os
import sys

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from models.demos.cosyvoice.tests.perf.test_llm_perf import LLM_WEIGHTS  # noqa: E402
from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder  # noqa: E402
from models.demos.cosyvoice.tt.weights import WeightBag  # noqa: E402

# Ops whose cost is a kernel launch, as opposed to bookkeeping the host resolves.
TRACKED = [
    "linear",
    "matmul",
    "add",
    "subtract",
    "multiply",
    "layer_norm",
    "softmax",
    "relu",
    "silu",
    "permute",
    "reshape",
    "slice",
    "concat",
    "copy",
    "transpose",
    "deallocate",
]


def main() -> int:
    counts: collections.Counter = collections.Counter()
    originals = {}

    def wrap(name):
        fn = getattr(ttnn, name)

        def counted(*a, **kw):
            counts[name] += 1
            return fn(*a, **kw)

        return fn, counted

    device = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        bag = WeightBag.load(LLM_WEIGHTS)
        meta = bag.meta["ar_decoder"]
        dec = TtARDecoder(device, bag.sub("llm"), meta)
        max_len = 256
        traced = TracedDecodeStep(dec, max_len)

        traced._body()  # warm up outside the counter, so JIT paths do not distort it

        for name in TRACKED:
            if hasattr(ttnn, name):
                originals[name], counted = wrap(name)
                setattr(ttnn, name, counted)
        try:
            traced._body()
        finally:
            for name, fn in originals.items():
                setattr(ttnn, name, fn)

        total = sum(counts.values())
        real = total - counts.get("deallocate", 0)
        n_layers = meta["n_layers"]
        print(f"\n  one decode step, {n_layers} layers, max_len={max_len}")
        print(f"  {'op':<14}{'count':>7}{'per layer':>11}{'share':>8}")
        for name, n in counts.most_common():
            print(f"  {name:<14}{n:>7}{n / n_layers:>11.1f}{100 * n / total:>7.1f}%")
        print(f"  {'TOTAL':<14}{total:>7}")
        print(f"  excluding deallocate: {real}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
