# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace capture for the AR decode step -- `02_plan.md` P6's headline lever.

F23 measured the LLM at **81.9 % of end-to-end runtime** and ~124 us per op over
~280 ops per token: dispatch bound, not compute bound. Trace capture replays the
recorded graph with one host command instead of re-issuing every op.

Two things are checked, in this order, because the second is worthless without the
first:

1. **The traced path computes the same thing.** A trace that is fast and wrong is
   the worst possible outcome, and the in-place KV cache it requires is exactly the
   kind of change that can silently corrupt state after a few steps.
2. **How much faster it is**, against the untraced fixed-shape path measured in
   `test_llm_perf.py`.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.cosyvoice.tt.weights import default_weights_path

LLM_WEIGHTS = default_weights_path().replace("hift_", "llm_")

needs_l1_small = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 131072, "trace_region_size": 67108864}], indirect=True
)
needs_weights = pytest.mark.skipif(not os.path.exists(LLM_WEIGHTS), reason="export llm weights first")


def _setup(device, ttnn, prefix_len=209, max_len=384):
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    dec = TtARDecoder(device, bag.sub("llm"), bag.meta["ar_decoder"])
    d = bag.meta["ar_decoder"]["input_size"]
    torch.manual_seed(0)
    prefix = torch.randn(1, prefix_len, d) * 0.1

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ys, caches = dec.forward_chunk_fixed(
        dev(prefix),
        dec.empty_cache(max_len, prefix_len),
        max_len,
        valid=prefix_len,
        mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
    )
    ttnn.deallocate(ys)
    return dec, caches, d, prefix_len, max_len


@needs_weights
@needs_l1_small
def test_device_traced_matches_untraced(device):
    """**Correctness first.** The same 8 decode steps, traced and untraced, must
    produce the same hidden states.

    The in-place cache is what makes this non-obvious: the untraced path allocates
    a fresh cache per step, the traced one writes back into persistent buffers, and
    a mistake there corrupts state progressively rather than immediately. Comparing
    every step, not just the last, is what localises that.
    """
    import ttnn
    from models.demos.cosyvoice.tt.common import pcc
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias

    dec, caches, d, prefix_len, max_len = _setup(device, ttnn)
    torch.manual_seed(1)
    steps = [torch.randn(1, 1, d) * 0.1 for _ in range(8)]

    # --- untraced reference, fixed-shape path
    want = []
    for i, x in enumerate(steps):
        ys, caches = dec.forward_chunk_fixed(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            caches,
            max_len,
            valid=prefix_len + 1 + i,
            mask=ttnn.from_torch(
                right_aligned_bias(max_len, prefix_len + 1 + i, 1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        )
        want.append(ttnn.to_torch(ys).float())
        ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)

    # --- traced, from a fresh prefill so both start from the same state
    dec2, caches2, *_ = _setup(device, ttnn)
    traced = TracedDecodeStep(dec2, max_len).capture()
    traced.seed(caches2)
    TtARDecoder.free_caches(caches2)

    got = []
    for i, x in enumerate(steps):
        ys = traced.step(x, prefix_len + 1 + i)
        ttnn.synchronize_device(device)
        got.append(ttnn.to_torch(ys).float())
    traced.release()

    print(f"\n  traced vs untraced, {len(steps)} decode steps")
    worst = 1.0
    for i, (a, b) in enumerate(zip(got, want)):
        p = pcc(a, b)
        worst = min(worst, p)
        print(f"    step {i}: PCC {p:.10f}  max|d| {(a - b).abs().max():.3e}")
    assert worst >= 0.999, worst


@needs_weights
@needs_l1_small
def test_device_traced_throughput(device):
    """How much the trace is worth. Compared against the untraced fixed-shape path
    on the identical shapes, in the same process."""
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias

    dec, caches, d, prefix_len, max_len = _setup(device, ttnn)
    torch.manual_seed(0)
    x = torch.randn(1, 1, d) * 0.1
    n = 32

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- untraced
    for i in range(2):  # warm
        ys, caches = dec.forward_chunk_fixed(
            dev(x), caches, max_len, prefix_len + 1 + i, dev(right_aligned_bias(max_len, prefix_len + 1 + i, 1))
        )
        ttnn.deallocate(ys)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for i in range(n):
        ys, caches = dec.forward_chunk_fixed(
            dev(x), caches, max_len, prefix_len + 3 + i, dev(right_aligned_bias(max_len, prefix_len + 3 + i, 1))
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(ys)
    untraced_ms = (time.perf_counter() - t0) / n * 1e3
    TtARDecoder.free_caches(caches)

    # --- traced
    traced = TracedDecodeStep(dec, max_len).capture()
    for i in range(2):
        traced.step(x, prefix_len + 1 + i)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for i in range(n):
        traced.step(x, prefix_len + 3 + i)
        # Synchronised per step, not once at the end. Generation is genuinely
        # sequential -- RAS needs the logits on the host before the next token can
        # be embedded -- so pipelined enqueue throughput would be the wrong number.
        ttnn.synchronize_device(device)
    traced_ms = (time.perf_counter() - t0) / n * 1e3
    traced.release()

    speedup = untraced_ms / traced_ms
    print(f"\n  AR decode step, max_len={max_len}, mean of {n}")
    print(f"    untraced   {untraced_ms:7.2f} ms   ({1e3/untraced_ms:6.1f} tok/s)")
    print(f"    traced     {traced_ms:7.2f} ms   ({1e3/traced_ms:6.1f} tok/s)")
    print(f"    speedup    {speedup:7.2f}x")
    print(f"    P4 wants >= 30 tok/s; P6 wants >= 60")
    print(f"    LLM RTF contribution at 50 tok/s of speech: {50 * traced_ms / 1e3:.3f}")
    assert traced_ms > 0
