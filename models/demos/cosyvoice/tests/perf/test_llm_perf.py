# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode throughput for the LLM stage -- the LLM's second exit criterion.

The LLM is the only sequential stage in CosyVoice: the flow decoder and the
vocoder each run once over the whole utterance, but the AR decoder runs once per
*token*, and a second of speech is 50 tokens. So tok/s translates directly into
RTF, and it is the number that decides whether the pipeline can be real time.

Measured, not asserted-and-hoped: the first pass JIT-compiles kernels, so timing
starts after a warm-up, and the cache grows by one position per step, which makes
later steps genuinely slower than earlier ones. Both are reported.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.cosyvoice.tt.weights import default_weights_path

LLM_WEIGHTS = default_weights_path().replace("hift_", "llm_")

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
needs_weights = pytest.mark.skipif(not os.path.exists(LLM_WEIGHTS), reason="export llm weights first")


@needs_weights
@needs_l1_small
def test_device_decode_throughput(device):
    """Prefill once, then time 32 single-token decode steps."""
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, causal_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta["ar_decoder"]
    dec = TtARDecoder(device, bag.sub("llm"), meta)
    d = meta["input_size"]
    prefix_len, n_steps = 209, 32

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch.manual_seed(0)
    prefix = torch.randn(1, prefix_len, d) * 0.1
    step_in = torch.randn(1, 1, d) * 0.1

    # warm-up: compile every kernel variant both shapes need before timing
    ys, caches = dec.forward_chunk(dev(prefix), caches=None, mask=dev(causal_bias(prefix_len)))
    ttnn.deallocate(ys)
    ys, caches = dec.forward_chunk(dev(step_in), caches=caches, mask=None)
    ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    ys, caches = dec.forward_chunk(dev(prefix), caches=None, mask=dev(causal_bias(prefix_len)))
    ttnn.synchronize_device(device)
    prefill_ms = (time.perf_counter() - t0) * 1e3
    ttnn.deallocate(ys)

    def run_steps():
        out = []
        for _ in range(n_steps):
            t = time.perf_counter()
            nonlocal caches
            y, caches = dec.forward_chunk(dev(step_in), caches=caches, mask=None)
            ttnn.synchronize_device(device)
            out.append((time.perf_counter() - t) * 1e3)
            ttnn.deallocate(y)
        return out

    pass1 = run_steps()
    TtARDecoder.free_caches(caches)

    # Second pass over the SAME key sizes. If a growing cache costs a JIT compile
    # per token -- every step has a new attention key size, and TTNN's program
    # cache is keyed on shape -- this pass is dramatically faster, because the
    # shapes are now warm. If the cost were arithmetic, the two passes would match.
    ys, caches = dec.forward_chunk(dev(prefix), caches=None, mask=dev(causal_bias(prefix_len)))
    ttnn.deallocate(ys)
    pass2 = run_steps()
    TtARDecoder.free_caches(caches)

    m1 = sum(pass1) / len(pass1)
    m2 = sum(pass2) / len(pass2)
    print(f"\n  prefill {prefix_len} tokens         {prefill_ms:8.2f} ms")
    print(f"  cold pass, mean of {n_steps}      {m1:8.2f} ms   ({1e3/m1:6.1f} tok/s)")
    print(f"    first / last step        {pass1[0]:8.2f} / {pass1[-1]:8.2f} ms")
    print(f"  warm pass, same shapes      {m2:8.2f} ms   ({1e3/m2:6.1f} tok/s)")
    print(f"    first / last step        {pass2[0]:8.2f} / {pass2[-1]:8.2f} ms")
    print(f"  compile share of the cold pass: {100 * (m1 - m2) / m1:.1f}%")
    print("  50 tok/s is real time at this token rate; the target is >= 30")

    tok_s = 1e3 / m2
    assert tok_s > 0
    # Reported, never xfailed: an unmet target gets a number and an explanation.
    if tok_s < 30:
        print(f"  BELOW the 30 tok/s target -- {tok_s:.1f} warm. Trace capture is the lever.")


@needs_weights
@needs_l1_small
def test_device_decode_throughput_fixed_cache(device):
    """The same measurement with the right-aligned fixed-width cache.

    This is the honest number for real generation. The growing-cache figure only
    looks acceptable on a second pass over sizes that are already compiled; an
    actual utterance visits each key size exactly once, so it pays every compile.
    Holding the width fixed leaves two shapes for the whole utterance, and the
    *first* pass runs at steady-state speed.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta["ar_decoder"]
    dec = TtARDecoder(device, bag.sub("llm"), meta)
    d = meta["input_size"]
    prefix_len, n_steps, max_len = 209, 32, 256

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch.manual_seed(0)
    prefix = torch.randn(1, prefix_len, d) * 0.1
    step_in = torch.randn(1, 1, d) * 0.1

    def prefill():
        caches = dec.empty_cache(max_len, prefix_len)
        ys, caches = dec.forward_chunk_fixed(
            dev(prefix),
            caches,
            max_len,
            valid=prefix_len,
            mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
        )
        ttnn.deallocate(ys)
        return caches

    # one warm-up of each of the two shapes
    caches = prefill()
    ys, caches = dec.forward_chunk_fixed(
        dev(step_in), caches, max_len, valid=prefix_len + 1, mask=dev(right_aligned_bias(max_len, prefix_len + 1, 1))
    )
    ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    caches = prefill()
    ttnn.synchronize_device(device)
    prefill_ms = (time.perf_counter() - t0) * 1e3

    per_step = []
    for i in range(n_steps):
        t = time.perf_counter()
        ys, caches = dec.forward_chunk_fixed(
            dev(step_in),
            caches,
            max_len,
            valid=min(prefix_len + 1 + i, max_len),
            mask=dev(right_aligned_bias(max_len, min(prefix_len + 1 + i, max_len), 1)),
        )
        ttnn.synchronize_device(device)
        per_step.append((time.perf_counter() - t) * 1e3)
        ttnn.deallocate(ys)
    TtARDecoder.free_caches(caches)

    mean_ms = sum(per_step) / len(per_step)
    tok_s = 1e3 / mean_ms
    print(f"\n  fixed cache, max_len={max_len}")
    print(f"  prefill {prefix_len} tokens         {prefill_ms:8.2f} ms")
    print(f"  decode step, mean of {n_steps}     {mean_ms:8.2f} ms   ({tok_s:6.1f} tok/s)")
    print(f"    first / last step        {per_step[0]:8.2f} / {per_step[-1]:8.2f} ms")
    print(f"  target is >= 30 tok/s; 50 tok/s is real time at this token rate")
    assert tok_s > 0


@needs_weights
@pytest.mark.parametrize("device_params", [{"l1_small_size": 131072, "trace_region_size": 67108864}], indirect=True)
def test_device_decode_bfloat8_weights(device):
    """`bfloat8_b` weights against `bfloat16`, on the traced decode step.

    Once tracing removed the dispatch overhead, what is left of a decode step is
    reading the AR decoder's weights out of DRAM to produce a single token. At
    batch 1 there is no reuse to amortise that against -- every matmul is a matrix
    against one row -- so the step is bandwidth-bound and halving the weight width
    is the lever that matches the bottleneck. Activations stay `bfloat16`; only the
    matrices narrow.

    Both halves of the trade get measured, because either one alone is misleading:
    throughput, and the drift the narrower mantissa costs after 14 blocks.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta["ar_decoder"]
    d = meta["input_size"]
    prefix_len, n_steps, max_len = 209, 32, 256

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch.manual_seed(0)
    prefix = torch.randn(1, prefix_len, d) * 0.1
    steps_in = [torch.randn(1, 1, d) * 0.1 for _ in range(n_steps)]

    def measure(weights_dtype):
        decoder = TtARDecoder(device, bag.sub("llm"), meta, weights_dtype=weights_dtype)
        caches = decoder.empty_cache(max_len, prefix_len)
        ys, caches = decoder.forward_chunk_fixed(
            dev(prefix),
            caches,
            max_len,
            valid=prefix_len,
            mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
        )
        ttnn.deallocate(ys)

        traced = TracedDecodeStep(decoder, max_len).capture()
        traced.seed(caches)
        TtARDecoder.free_caches(caches)

        traced.step(steps_in[0], prefix_len + 1)  # warm the replay path
        ttnn.synchronize_device(device)

        per_step, last = [], None
        for i, x in enumerate(steps_in):
            t = time.perf_counter()
            out = traced.step(x, min(prefix_len + 1 + i, max_len))
            ttnn.synchronize_device(device)
            per_step.append((time.perf_counter() - t) * 1e3)
            last = ttnn.to_torch(out).float()
        traced.release()
        return sum(per_step) / len(per_step), last

    bf16_ms, bf16_out = measure(None)
    bf8_ms, bf8_out = measure(ttnn.bfloat8_b)

    a, b = bf16_out.flatten(), bf8_out.flatten()
    pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    speedup = bf16_ms / bf8_ms

    print(f"\n  traced decode step, mean of {n_steps}")
    print(f"    bfloat16 weights {bf16_ms:8.2f} ms   ({1e3 / bf16_ms:6.1f} tok/s)")
    print(f"    bfloat8_b weights{bf8_ms:8.2f} ms   ({1e3 / bf8_ms:6.1f} tok/s)   {speedup:.2f}x")
    print(f"    hidden-state PCC, bf8 vs bf16      {pcc:.10f}")
    print(f"    LLM RTF at 164 tokens / 3.27 s: {bf16_ms * 164 / 3270:.3f} -> {bf8_ms * 164 / 3270:.3f}")

    # The gate is accuracy, not speed: a faster decoder that drifts is not a win,
    # and the speedup is reported rather than asserted because it is the thing
    # being measured.
    assert pcc >= 0.99, f"bfloat8_b weights cost too much accuracy: PCC {pcc}"
