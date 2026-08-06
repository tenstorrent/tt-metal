# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace capture for the AR decode step -- `02_plan.md` P6's headline lever.

Before any of it was traced, the LLM was **81.9 % of end-to-end runtime** at ~124 us
per op over ~280 ops per token: dispatch bound, not compute bound. That is what made
trace capture the right lever -- it replays the recorded graph with one host command
instead of re-issuing every op. (Post-tracing figures live in PERF.md; the number
here is the *motivation*, so it stays as it was measured.)

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
# The in-place path captures one trace per scratch row plus one for the periodic
# shift, where the moving one captures a single trace -- so it needs a correspondingly
# larger trace region. An exhausted region shows up as an allocation failure during
# capture, not as a wrong answer.
needs_big_trace = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 131072, "trace_region_size": 402653184}], indirect=True
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
@needs_big_trace
def test_device_inplace_matches_untraced(device):
    """The in-place cache, over **enough steps to cross a shift boundary**.

    Eight steps would prove nothing here. The design is that the cache stays still
    for a whole scratch zone of sub-steps and then slides by exactly that much, so
    the interesting failures all live at that seam: an off-by-one in `slot_bias`, a
    `bd_offset` that drifts out of step with the write row, a shift that discards
    rows the mask still thinks are live. Running a scratch zone plus eight steps puts
    the boundary inside the comparison, with steps on both sides of it.

    The reference is the untraced fixed-shape path, which keeps the same window by
    moving the cache instead of the query -- so agreement means the two ways of
    expressing "the last `max_len` tokens" really do line up.

    **The moving cache is also run at the wider buffer**, and that control is the
    point of the test rather than a garnish. The in-place path does not reproduce the
    reference bit-for-bit the way `TracedDecodeStep` does, and there are two possible
    reasons: `update_cache` could be perturbing the cache, or the reduction could
    simply be grouping a different number of tiles -- 14 instead of 12, with the extra
    ones contributing exact zeros in a different order. Those call for opposite
    responses, and only a third measurement tells them apart.

    It is a valid control only while the window is not full: with `valid` below
    `max_len` both widths hold the same live keys and differ solely in how much
    masked padding surrounds them. That holds here (`209 + 72 < 384`) and would stop
    holding if the step count grew past ~175.
    """
    import ttnn
    from models.demos.cosyvoice.tt.common import pcc
    from models.demos.cosyvoice.tt.llm.decoder import (
        TracedDecodeStep,
        TracedDecodeStepInPlace,
        TtARDecoder,
        right_aligned_bias,
    )

    dec, caches, d, prefix_len, max_len = _setup(device, ttnn)
    torch.manual_seed(1)
    shift_at = TracedDecodeStepInPlace.TILE * TracedDecodeStepInPlace.SCRATCH_TILES
    n = shift_at + 8
    steps = [torch.randn(1, 1, d) * 0.1 for _ in range(n)]

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

    dec2, caches2, *_ = _setup(device, ttnn)
    traced = TracedDecodeStepInPlace(dec2, max_len).capture()
    traced.seed(caches2)
    TtARDecoder.free_caches(caches2)

    got = []
    for i, x in enumerate(steps):
        ys = traced.step(x, prefix_len + 1 + i)
        ttnn.synchronize_device(device)
        got.append(ttnn.to_torch(ys).float())
    traced.release()

    # --- the control: same window, same wider buffer, cache still moving
    width = max_len + shift_at
    dec3, caches3, *_ = _setup(device, ttnn, max_len=width)
    wide = TracedDecodeStep(dec3, width).capture()
    wide.seed(caches3)
    TtARDecoder.free_caches(caches3)
    ctrl = []
    for i, x in enumerate(steps):
        ys = wide.step(x, prefix_len + 1 + i)
        ttnn.synchronize_device(device)
        ctrl.append(ttnn.to_torch(ys).float())
    wide.release()

    print(f"\n  in-place vs untraced, {n} decode steps (shift at step {shift_at})")
    worst, worst_at, worst_ctrl = 1.0, -1, 1.0
    for i, (a, b, c) in enumerate(zip(got, want, ctrl)):
        p, q = pcc(a, b), pcc(c, b)
        if p < worst:
            worst, worst_at = p, i
        worst_ctrl = min(worst_ctrl, q)
        if i < 3 or abs(i - shift_at) <= 1 or i == n - 1:
            print(f"    step {i:2d}: in-place {p:.10f}   moving@{width} {q:.10f}   max|d| {(a - b).abs().max():.3e}")
    print(f"    worst: in-place {worst:.10f} (step {worst_at})   moving@{width} {worst_ctrl:.10f}")

    # Gated below the moving path's exact match, because it cannot be exact: the
    # control above shows how much of the gap is the wider reduction alone. What the
    # gate is really protecting against is the structural failure -- a `bd_offset` or
    # write row out of step, which collapses PCC rather than nudging it.
    assert worst >= 0.995, f"worst PCC {worst} at step {worst_at}"
    assert worst >= worst_ctrl - 0.002, (
        f"in-place {worst} is worse than the width alone explains ({worst_ctrl}) -- "
        "suspect the write index or the positional offset, not rounding"
    )


@needs_weights
@needs_big_trace
def test_device_inplace_throughput(device):
    """What the in-place cache is worth -- and, if it is worth nothing, why.

    Two measurements would only say which is faster. Three say *what made it so*,
    because the in-place buffer is `TILE` rows wider than the window it serves and
    that widening is a cost of the design, not of the mechanism. So the moving cache
    is measured twice: once at the window width it needs, and once at the wider
    buffer the in-place version is obliged to carry.

        moving @ max_len   -- the shipped path
        moving @ width     -- the same mechanism, paying only the widening
        in-place @ width   -- the widening plus `update_cache`

    The first gap is the price of the wider tensors; the second is what writing the
    cache in place is actually worth against rebuilding it. Attributing a regression
    to the wrong one of those is how a good idea gets abandoned for someone else's
    reason.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TracedDecodeStepInPlace, TtARDecoder

    torch.manual_seed(0)
    n = 64  # two full shift cycles, so the periodic shift is amortised honestly
    width = 384 + TracedDecodeStepInPlace.TILE * TracedDecodeStepInPlace.SCRATCH_TILES

    def bench(make, max_len):
        dec, caches, d, prefix_len, _ = _setup(device, ttnn, max_len=max_len)
        x = torch.randn(1, 1, d) * 0.1
        step = make(dec, max_len).capture()
        step.seed(caches)
        TtARDecoder.free_caches(caches)
        for i in range(2):
            step.step(x, prefix_len + 1 + i)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(n):
            step.step(x, prefix_len + 3 + i)
            ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) / n * 1e3
        step.release()
        return ms

    moving_ms = bench(TracedDecodeStep, 384)
    wide_ms = bench(TracedDecodeStep, width)
    # The in-place class widens internally, so it is asked for the same 384 window.
    inplace_ms = bench(TracedDecodeStepInPlace, 384)

    print(f"\n  AR decode step, window 384, mean of {n}")
    print(f"    moving   @ 384    {moving_ms:7.2f} ms   ({1e3/moving_ms:6.1f} tok/s)")
    print(f"    moving   @ {width}    {wide_ms:7.2f} ms   ({1e3/wide_ms:6.1f} tok/s)   widening alone")
    print(f"    in-place @ {width}    {inplace_ms:7.2f} ms   ({1e3/inplace_ms:6.1f} tok/s)")
    print(f"    cost of widening      {wide_ms - moving_ms:+7.2f} ms")
    print(f"    value of in-place     {wide_ms - inplace_ms:+7.2f} ms   (at equal width)")
    print(f"    net vs shipped        {moving_ms - inplace_ms:+7.2f} ms   ({moving_ms/inplace_ms:.2f}x)")
    print(f"    LLM RTF contribution at 50 tok/s of speech: {50 * inplace_ms / 1e3:.3f}")
    assert inplace_ms > 0


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
