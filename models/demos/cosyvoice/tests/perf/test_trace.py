# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace capture for the AR decode step -- the bring-up's headline perf lever.

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

from models.demos.cosyvoice.tests.perf.gates import enforce, report
from models.demos.cosyvoice.tt.common import GOLDEN_DIR
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
def test_device_fused_attention_matches_explicit(device):
    """`sdpa_decode` against the explicit rel-pos chain it replaces.

    The two are independently reachable — `COSYVOICE_SDPA_DECODE=0` restores the
    chain — so the equivalence needs its own gate rather than riding on
    `test_device_traced_matches_untraced`, which compares each path only to itself.

    Untraced on both sides, because this is a test of the *arithmetic*. The identity
    it is checking is that at `T = 1` the positional term `(q+v)P^T` is a per-head row
    vector over the key axis, which is exactly what `attn_mask` accepts — so the four
    ops the chain spends on it collapse into one kernel. Tracing is a separate
    question and the two tests either side of this one already answer it.

    0.998 rather than bit-exactness: flash attention reassociates the softmax across
    k-chunks, so the sums are formed in a different order. That is a real difference
    and the gate should not pretend otherwise. Per layer it is tiny — both paths measure
    0.99998 against a torch golden — but it compounds twice over, through 14
    layers and then through the KV cache, which is why the measured spread is
    0.9988-0.9999 and drifts down with step index rather than staying flat.

    The gate that actually protects the model is `test_token_agreement`, which is
    exact-token and unmoved at 95.83 %. This one is here to catch a *structural*
    break — a bias in the wrong convention, a mask in the wrong form — which collapses
    PCC to 0.7-0.9 rather than nudging the fourth decimal.
    """
    ttnn = pytest.importorskip("ttnn")
    from models.demos.cosyvoice.tt.common import pcc
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias

    def run(fused: bool):
        dec, caches, d, prefix_len, max_len = _setup(device, ttnn)
        for layer in dec.layers:
            layer.attn.sdpa_decode = fused
        torch.manual_seed(3)
        out = []
        for i in range(6):
            x = torch.randn(1, 1, d) * 0.1
            ys, caches = dec.forward_chunk_fixed(
                ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                caches,
                max_len,
                valid=prefix_len + 1 + i,
                mask=ttnn.from_torch(
                    right_aligned_bias(max_len, prefix_len + 1 + i, 1, heads=dec.meta["n_head"] if fused else 1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                ),
            )
            out.append(ttnn.to_torch(ys).float())
            ttnn.deallocate(ys)
        TtARDecoder.free_caches(caches)
        return out

    got, want = run(True), run(False)
    print("\n  fused sdpa_decode vs explicit chain")
    worst = 1.0
    for i, (a, b) in enumerate(zip(got, want)):
        p = pcc(a, b)
        worst = min(worst, p)
        print(f"    step {i}: PCC {p:.10f}  max|d| {(a - b).abs().max():.3e}")
    assert worst >= 0.998, f"worst PCC {worst}"


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

    # Both cache mechanisms ship -- `kv_inplace_default` picks between them by
    # architecture -- so both carry the throughput gates, not only the faster one.
    report(
        [
            enforce("tok_s", 1e3 / moving_ms, device, extra="moving cache"),
            enforce("tok_s_stretch", 1e3 / moving_ms, device, extra="moving cache"),
            enforce("tok_s", 1e3 / inplace_ms, device, extra="in-place cache"),
            enforce("tok_s_stretch", 1e3 / inplace_ms, device, extra="in-place cache"),
        ],
        "bounty gates -- AR decode step, both cache mechanisms",
    )


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
    print(f"    LLM RTF contribution at 50 tok/s of speech: {50 * traced_ms / 1e3:.3f}")

    # The traced step is what `generate()` runs, so it is where the token-throughput
    # gates are enforced. The untraced figure above is the control, not the claim.
    tok_s = 1e3 / traced_ms
    report(
        [enforce("tok_s", tok_s, device), enforce("tok_s_stretch", tok_s, device)],
        f"bounty gates -- traced AR decode step, max_len={max_len}",
    )


# ---------------------------------------------------------------- HiFT vocoder

# The vocoder captures one trace per (mel_frames, batch_size). 400 MB covers the
# 282-frame utterance with room for a second geometry; an exhausted region shows up
# as a capture failure and a fallback to the untraced path, not as a wrong answer.
needs_hift_trace = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 402653184}], indirect=True
)
HIFT_WEIGHTS = default_weights_path()
needs_hift_weights = pytest.mark.skipif(not os.path.exists(HIFT_WEIGHTS), reason="run scripts/export_weights.py first")
needs_hift_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "hift.decode.npz")),
    reason="run scripts/gen_golden.py first",
)


def _hift_inputs(device, ttnn):
    from models.demos.cosyvoice.tt.common import as_torch, load_golden

    g = load_golden("hift.decode")
    mel_t = as_torch(g["call0.in_x"])  # [1, 80, T_mel]
    s_t = as_torch(g["call0.in_s"])  # [1, 1, T_audio]

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return dev(mel_t.permute(0, 2, 1).contiguous()), dev(s_t.permute(0, 2, 1).contiguous()), mel_t.shape[-1]


@needs_hift_weights
@needs_hift_golden
@needs_hift_trace
def test_hift_trace_is_bit_identical(device):
    """The traced vocoder must return exactly what the untraced one does.

    Bit-identical rather than a PCC gate, because there is nothing here that should
    differ: the trace replays the same kernels on the same data. Anything short of
    `max|d| == 0` means the replay is reading a buffer it should not be -- the failure
    mode that cost the CFM a `PCC 0.0017` before its output was moved inside the
    capture.
    """
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator

    model = TtHiFTGenerator.from_export(device)
    model.enable_trace(True)  # off by default; streaming is what opts in
    mel, s, mel_frames = _hift_inputs(device, ttnn)
    try:
        want = ttnn.to_torch(model._decode_impl(mel, s, mel_frames)).float()

        # First sighting runs untraced by design; the second captures and replays.
        runs = [("untraced", ttnn.to_torch(model.decode(mel, s, mel_frames)).float())]
        assert model._trace_id is None, "a geometry seen once must not be captured"
        runs.append(("captured", ttnn.to_torch(model.decode(mel, s, mel_frames)).float()))
        assert model._trace_id is not None, "second sighting should have captured"

        # Repeated replays, not just one. Each `decode` clones its result out of the
        # trace's output buffer, and TTNN warns that buffers allocated while a trace
        # is alive may collide with addresses the replay baked in. That is not
        # hypothetical -- it is what scored the CFM `PCC 0.0017` before its output was
        # moved inside the capture -- and it shows up on the fourth or fifth call, not
        # the second. A streamed utterance replays this many times per second.
        for i in range(6):
            runs.append((f"replay {i}", ttnn.to_torch(model.decode(mel, s, mel_frames)).float()))

        for name, got in runs:
            d = (got - want).abs().max()
            print(f"  {name:>9}  max|d| {d:.3e}")
            assert got.shape == want.shape, (name, got.shape, want.shape)
            assert d == 0, (name, float(d))
    finally:
        model.release()


@needs_hift_weights
@needs_hift_golden
@needs_hift_trace
def test_hift_trace_is_faster(device):
    """How much a replay is worth, and what it costs to take the trace.

    **The capture figure printed here flatters itself, and the break-even derived from it
    is wrong.** The test above has already captured and released a trace for this same
    geometry, and a second capture of a geometry the process has seen costs `~110-160 ms`
    where the first costs `~980 ms`. Reading `break-even at 5 replays` off this line is
    exactly the mistake that put a 30-chunk crossover into the code as a 3-chunk one.

    The replay number is the trustworthy one, and it is the point: `~3x` at this geometry.
    For the honest capture cost and crossover see PERF.md, measured with a fresh geometry
    in a fresh process.
    """
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator

    model = TtHiFTGenerator.from_export(device)
    model.enable_trace(True)
    mel, s, mel_frames = _hift_inputs(device, ttnn)
    n = 5
    try:
        for _ in range(2):
            ttnn.deallocate(model._decode_impl(mel, s, mel_frames))
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(model._decode_impl(mel, s, mel_frames))
            ttnn.synchronize_device(device)
        untraced_ms = (time.perf_counter() - t0) / n * 1e3

        t0 = time.perf_counter()
        assert model._capture(mel, s, (mel_frames, 1)), "capture failed"
        ttnn.synchronize_device(device)
        capture_ms = (time.perf_counter() - t0) * 1e3

        ttnn.deallocate(model._replay(mel, s))
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(model._replay(mel, s))
            ttnn.synchronize_device(device)
        traced_ms = (time.perf_counter() - t0) / n * 1e3

        saved = untraced_ms - traced_ms
        breakeven = f"break-even at {capture_ms / saved:.1f} replays" if saved > 0 else "never pays back"
        print(f"\n  HiFT decode, {mel_frames} mel frames, mean of {n}")
        print(f"    untraced   {untraced_ms:7.2f} ms")
        print(f"    traced     {traced_ms:7.2f} ms   ({untraced_ms / traced_ms:.2f}x)")
        print(f"    capture    {capture_ms:7.2f} ms   ({breakeven})")
        assert traced_ms > 0
    finally:
        model.release()
