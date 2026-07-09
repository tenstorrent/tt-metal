# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for Time Series Transformer -- KV-cache + N-trace generation.

Bounty Stage 1 targets (verbatim from issue):
  latency:           < 50 ms   (B=1, single sequence)
  throughput:        >= 100 seq/s
  sample generation: 100 samples in < 1 second

These are asserted as hard targets here regardless of which stage's
techniques (caching, fusion, sharding) are needed to hit them. The bounty's
acceptance checklist itself doesn't gate the number on stage -- Stage 2/3
items like KV-cache and fused ops exist because they're required to reach
Stage 1's numbers, not because Stage 1 is exempt from needing them. See
project notes for the measured justification.

XFAIL HISTORY (resolved, kept for record): Tests 1, 2, and 5 previously
conflated S = num_parallel_samples with the measured workload (B=1 S=100
instead of B=1 S=1 for "single sequence latency"; S=10 instead of S=1 for
throughput). That's fixed -- all three now use S=1. The xfail markers that
existed while that bug was open have been removed: the numbers below are
now honest measurements of the correct workload, not placeholders pending
re-measurement, so a real failure should surface as a real failure.

WARMUP FIX (this revision): measured on hardware 2026-07-09, use_2cq=False,
B=1 S=1, post multi-trace correctness fix -- 10 replays immediately after
ctx build, only 1 prior warmup call:
    120.6, 124.6, 120.2, 97.4, 99.2, 98.4, 98.8, 99.1, 99.9, 99.6 (ms)
Three elevated replays (120-125ms) before settling to a 97-99ms steady
state. One warmup call is not enough to clear whatever host-side dispatch
caching resolves on first real replay after trace capture. Warmup bumped
to 5 discarded replays on all latency-sensitive tests below, with margin
past the observed 3-replay settling point.

CURRENT MEASURED BASELINE (2026-07-09, use_2cq=False, B=1 S=1, steady
state after proper warmup): median ~98ms. Target is 50ms. This is EXPECTED
to fail until Stage-2-labeled op fusion (KV write path: slice_write_kv
21.7ms + qkv_linear 7ms + to_layout_kv 6ms + qkv_split 5ms = ~43% of
per-step latency, per _print_op_times()) lands. A failing
test_single_sequence_latency right now is a correct, honest result --
do not re-add xfail to hide it.
"""

import math
import statistics
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction

import ttnn
from models.demos.time_series_transformer.tt.tst_model import load_weights
from models.demos.time_series_transformer.tt.tst_model_cached_additions import (
    build_traced_decoder_context_cached,
    run_traced_generation_cached,
)

TRACE_BYTES_PER_BS_UNIT = 640_615
TRACE_HEADROOM = 1.25
L1_SMALL_SIZE = 24_576

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

# Discarded warmup replays before any timed measurement. See module
# docstring "WARMUP FIX" -- empirically needs to exceed 3 to clear the
# elevated-replay window observed on hardware.
WARMUP_REPLAYS = 5


def _trace_region_size(bs: int) -> int:
    return int(math.ceil(TRACE_BYTES_PER_BS_UNIT * bs * TRACE_HEADROOM))


def _open_device(bs: int, num_cqs: int = 1) -> ttnn.Device:
    return ttnn.open_device(
        device_id=0,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=_trace_region_size(bs),
        num_command_queues=num_cqs,
    )


@pytest.fixture(scope="module")
def hf_model():
    model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    return model


@pytest.fixture(scope="module")
def inputs():
    tensors = {}
    with safe_open(str(REFERENCE_DIR / "inputs.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def _slice_b(inputs: dict, b: int) -> dict:
    return {k: v[:b] for k, v in inputs.items()}


def _unpack_inputs(d: dict):
    return (
        d["input_past_values"],
        d["input_past_time_features"],
        d["input_future_time_features"],
        d["input_past_observed_mask"],
        d["input_static_categorical_features"],
        d["input_static_real_features"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 -- Single-sequence latency (B=1, S=1)
# ─────────────────────────────────────────────────────────────────────────────


def test_single_sequence_latency(hf_model, inputs):
    B, S = 1, 1
    device = _open_device(B * S, num_cqs=1)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            for _ in range(WARMUP_REPLAYS):
                run_traced_generation_cached(ctx, weights, use_2cq=False)
            latencies = []
            for _ in range(10):
                t0 = time.perf_counter()
                run_traced_generation_cached(ctx, weights, use_2cq=False)
                latencies.append((time.perf_counter() - t0) * 1000)
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    median_ms = statistics.median(latencies)
    print(f"\n[INFO] Latency (S=1) -- median: {median_ms:.1f} ms  all: {[f'{v:.1f}' for v in latencies]}")
    assert median_ms < 50.0, f"Latency {median_ms:.1f} ms exceeds 50 ms target."


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 -- Throughput >= 100 seq/s  (S=1)
# ─────────────────────────────────────────────────────────────────────────────


def test_throughput_seqs_per_second(hf_model, inputs):
    B_avail = inputs["input_past_values"].shape[0]
    B = min(4, B_avail)
    S = 1
    device = _open_device(B * S, num_cqs=2)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            for _ in range(WARMUP_REPLAYS):
                run_traced_generation_cached(ctx, weights, use_2cq=True)
            N_RUNS = 10
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                run_traced_generation_cached(ctx, weights, use_2cq=True)
            elapsed = time.perf_counter() - t0
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    total_seqs = N_RUNS * B
    seqs_per_sec = total_seqs / elapsed
    print(f"\n[INFO] Throughput (S=1): {seqs_per_sec:.1f} seq/s ({total_seqs} seqs in {elapsed:.2f}s)")
    assert seqs_per_sec >= 100.0, f"Throughput {seqs_per_sec:.1f} seq/s < 100 seq/s target."


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 -- 100 samples in < 1 second  (unchanged -- S=100 was already correct)
# ─────────────────────────────────────────────────────────────────────────────


def test_sample_generation_under_1s(hf_model, inputs):
    B, S = 1, 100
    device = _open_device(B * S, num_cqs=2)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            run_traced_generation_cached(ctx, weights, use_2cq=True)  # warmup
            t0 = time.perf_counter()
            samples = run_traced_generation_cached(ctx, weights, use_2cq=True)
            elapsed_s = time.perf_counter() - t0
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    print(f"\n[INFO] 100 samples in {elapsed_s*1000:.1f} ms")
    assert elapsed_s < 1.0, f"100 samples took {elapsed_s:.3f}s, target < 1s."


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 -- use_2cq=True output must match use_2cq=False output  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
#
# CONTEXT: run_traced_generation_cached's use_2cq=True path (CQ1 writes,
# CQ0 runs the trace, handoff via record_event/wait_for_event) had never
# been checked for output CORRECTNESS -- test_throughput_seqs_per_second
# and test_sample_generation_under_1s both already exercise use_2cq=True,
# but both assert ONLY on elapsed time, never on the values in `samples`.
# A race condition that silently corrupts values would still pass both of
# those tests as long as it ran fast enough.
#
# A targeted hardware probe (probe_2cq_event_ordering.py, single-decoder
# -layer harness, not committed here) found exact (0.000000 diff) agreement
# between use_2cq=True and use_2cq=False across 8 steps with deliberate
# cross-step marker values, with no cross-step contamination -- meaning
# ttnn.record_event(device, CQ_COMPUTE) issued right after a non-blocking
# ttnn.execute_trace(...) DOES correctly gate the writer queue on this
# hardware/ttnn version, even though this exact pattern (event recorded
# immediately after execute_trace, rather than after a separate untraced
# "consumer op") isn't shown verbatim in the tt-metal tech report's own
# combined trace+2CQ examples. That probe used a minimal single-layer
# harness with forced per-step synchronization, though -- it did not prove
# the FULL model is correct under realistic (pipelined, non-forced-sync)
# 2CQ execution. This test closes that gap by comparing full end-to-end
# output (not just one decoder layer's K-cache) between use_2cq=True and
# use_2cq=False, with the latter as ground truth (CQ_WRITE == CQ_COMPUTE
# == 0, fully blocking, no cross-queue race possible by construction).
#
# Two SEPARATE ctx instances are built (one per use_2cq value) rather than
# reusing one ctx across two run_traced_generation_cached calls, because
# _zero_kv_caches is only called once, inside build_traced_decoder_context_
# cached -- KV cache reuse safety across repeated calls on one ctx has not
# been separately verified, and this test's whole purpose is establishing
# trust, so it should not rest on an additional unverified assumption.


def test_2cq_matches_single_queue_output(hf_model, inputs):
    B, S = 1, 10  # small S: this test checks correctness, not throughput --
    # keep it cheap. Distributional comparison below is loose for this
    # reason (10 samples is not enough for tight CRPS agreement); the
    # NaN/Inf/scale checks are the primary signal, mean/std is secondary.
    device_1cq = _open_device(B * S, num_cqs=1)
    try:
        weights = load_weights(hf_model, device_1cq)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx_1cq = build_traced_decoder_context_cached(device_1cq, weights, *inp, num_parallel_samples=S)
        try:
            run_traced_generation_cached(ctx_1cq, weights, use_2cq=False)  # warmup
            samples_1cq = run_traced_generation_cached(ctx_1cq, weights, use_2cq=False)
        finally:
            ctx_1cq.release()
    finally:
        ttnn.close_device(device_1cq)

    device_2cq = _open_device(B * S, num_cqs=2)
    try:
        weights = load_weights(hf_model, device_2cq)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx_2cq = build_traced_decoder_context_cached(device_2cq, weights, *inp, num_parallel_samples=S)
        try:
            run_traced_generation_cached(ctx_2cq, weights, use_2cq=True)  # warmup
            samples_2cq = run_traced_generation_cached(ctx_2cq, weights, use_2cq=True)
        finally:
            ctx_2cq.release()
    finally:
        ttnn.close_device(device_2cq)

    # -- Hard correctness checks: these must pass unconditionally --------
    has_nan_1cq = torch.isnan(samples_1cq).any().item()
    has_nan_2cq = torch.isnan(samples_2cq).any().item()
    has_inf_1cq = torch.isinf(samples_1cq).any().item()
    has_inf_2cq = torch.isinf(samples_2cq).any().item()

    assert not has_nan_1cq, "use_2cq=False (ground truth) produced NaN -- fix baseline before trusting this test"
    assert not has_inf_1cq, "use_2cq=False (ground truth) produced Inf -- fix baseline before trusting this test"
    assert not has_nan_2cq, "use_2cq=True produced NaN -- likely a race condition corrupting values"
    assert not has_inf_2cq, "use_2cq=True produced Inf -- likely a race condition corrupting values"
    assert samples_1cq.shape == samples_2cq.shape, (
        f"shape mismatch: use_2cq=False gave {tuple(samples_1cq.shape)}, "
        f"use_2cq=True gave {tuple(samples_2cq.shape)}"
    )

    # -- Distributional comparison (loose -- S=10 is not enough samples for -
    # tight agreement; this is a sanity check, not a precision check) ------
    mean_1cq, std_1cq = samples_1cq.mean().item(), samples_1cq.std().item()
    mean_2cq, std_2cq = samples_2cq.mean().item(), samples_2cq.std().item()

    print(
        f"\n[INFO] use_2cq=False: mean={mean_1cq:.4f} std={std_1cq:.4f}\n"
        f"       use_2cq=True:  mean={mean_2cq:.4f} std={std_2cq:.4f}"
    )

    mean_diff = abs(mean_1cq - mean_2cq)
    mean_scale = max(abs(mean_1cq), abs(mean_2cq), 1e-6)
    assert mean_diff / mean_scale < 0.5, (
        f"Sample means differ by more than 50% relative ({mean_1cq:.4f} vs {mean_2cq:.4f}) "
        f"-- this is far looser than expected sampling noise alone and suggests use_2cq=True "
        f"is reading stale/corrupted cache or mask state, not just different RNG draws."
    )

    print(
        "\n[INFO] test_2cq_matches_single_queue_output PASSED -- use_2cq=True output is "
        "NaN/Inf-free, correctly shaped, and distributionally consistent with the "
        "trusted use_2cq=False baseline. Combined with probe_2cq_event_ordering.py's "
        "exact-match hardware result, this gives reasonable confidence the 2CQ event "
        "choreography is safe on this hardware/ttnn version -- though neither check is "
        "a substitute for the full CRPS/NLL precision the e2e suite already provides "
        "for use_2cq=False."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 -- Single-sequence latency with 2CQ  (B=1, S=1)
# ─────────────────────────────────────────────────────────────────────────────


def test_single_sequence_latency_2cq(hf_model, inputs):
    B, S = 1, 1
    device = _open_device(B * S, num_cqs=2)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            for _ in range(WARMUP_REPLAYS):
                run_traced_generation_cached(ctx, weights, use_2cq=True)
            latencies = []
            for _ in range(10):
                t0 = time.perf_counter()
                run_traced_generation_cached(ctx, weights, use_2cq=True)
                latencies.append((time.perf_counter() - t0) * 1000)
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    median_ms = statistics.median(latencies)
    print(f"\n[INFO] Latency (2CQ, S=1) -- median: {median_ms:.1f} ms  all: {[f'{v:.1f}' for v in latencies]}")
    assert median_ms < 50.0, f"Latency {median_ms:.1f} ms exceeds 50 ms target."
