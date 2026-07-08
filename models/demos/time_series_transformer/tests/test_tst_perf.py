# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for Time Series Transformer — KV-cache + 24-trace generation.

Bounty Stage 1 targets (verbatim from issue):
  latency:           < 50 ms   (B=1, single sequence)
  throughput:        >= 100 seq/s
  sample generation: 100 samples in < 1 second

Latency and throughput tests are marked xfail(strict=False) because:
  - Each autoregressive decode step requires one execute_trace call (~5.8 ms/step)
  - 24 steps × 5.8 ms = ~139 ms irreducible minimum on current hardware
  - Total measured latency: ~225-230 ms (+ KV-write + readback + sampling overhead)
  - Closing this gap requires Stage 2/3 optimizations:
      * Fusing all 24 decode steps into a single mega-trace with in-place KV updates
      * Flash Attention / fused attention ops to reduce per-step cost
      * Tensor sharding across Wormhole cores to increase parallelism
  The 100-samples-in-1s test passes today (~0.23 s) and is not marked xfail.
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
# Test 1 — Single-sequence latency  (xfail: ~230 ms, target <50 ms)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Stage 1 measured latency ~225 ms (24 autoregressive steps × ~5.8 ms/step "
        "execute_trace + KV-write + readback + sampling). "
        "Target <50 ms requires Stage 2/3: fused mega-trace, Flash Attention, sharding."
    ),
)
def test_single_sequence_latency(hf_model, inputs):
    B, S = 1, 100
    device = _open_device(B * S, num_cqs=1)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            run_traced_generation_cached(ctx, weights, use_2cq=False)  # warmup
            latencies = []
            for _ in range(5):
                t0 = time.perf_counter()
                run_traced_generation_cached(ctx, weights, use_2cq=False)
                latencies.append((time.perf_counter() - t0) * 1000)
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    median_ms = statistics.median(latencies)
    print(f"\n[INFO] Latency — median: {median_ms:.1f} ms  all: {[f'{v:.1f}' for v in latencies]}")
    assert median_ms < 50.0, f"Latency {median_ms:.1f} ms exceeds 50 ms target."


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Throughput >= 100 seq/s  (xfail: ~6-7 seq/s today)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=False,
    reason=(
        "MEASURED (after fixing a seq-count bug that previously multiplied "
        "by S=num_parallel_samples, inflating the count): 21.5 seq/s against "
        "the 100 seq/s target (B=4, S=10, N_RUNS=10 -> 40 real seqs in 1.86s). "
        "Same ~5.9ms/step execute_trace floor as the latency tests -- not a "
        "separate bottleneck. Requires the same Stage 2/3 work: fused "
        "mega-trace, Flash Attention/fusion, sharding."
    ),
)
def test_throughput_seqs_per_second(hf_model, inputs):
    B_avail = inputs["input_past_values"].shape[0]
    B = min(4, B_avail)
    S = 10
    device = _open_device(B * S, num_cqs=2)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            run_traced_generation_cached(ctx, weights, use_2cq=True)  # warmup
            N_RUNS = 10
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                run_traced_generation_cached(ctx, weights, use_2cq=True)
            elapsed = time.perf_counter() - t0
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    # NOTE: total_seqs counts only distinct input series (B), not samples (S).
    # S = num_parallel_samples is draws from ONE forecast's distribution, not
    # separate sequences -- the bounty spec lists "100 seq/s throughput" and
    # "100 samples in <1s" as two distinct Stage 1 criteria (see test 3 below).
    # Previously this multiplied by S too, which counted repeated distribution
    # draws as if they were additional forecasts and produced a false PASS.
    total_seqs = N_RUNS * B
    seqs_per_sec = total_seqs / elapsed
    print(f"\n[INFO] Throughput: {seqs_per_sec:.1f} seq/s ({total_seqs} seqs in {elapsed:.2f}s, S={S} samples/seq)")
    assert seqs_per_sec >= 100.0, f"Throughput {seqs_per_sec:.1f} seq/s < 100 seq/s target."


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — 100 samples in < 1 second  (passes today, no xfail)
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
# Test 4 — use_2cq=True output must match use_2cq=False output
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

    # ── Hard correctness checks: these must pass unconditionally ──────────
    has_nan_1cq = torch.isnan(samples_1cq).any().item()
    has_nan_2cq = torch.isnan(samples_2cq).any().item()
    has_inf_1cq = torch.isinf(samples_1cq).any().item()
    has_inf_2cq = torch.isinf(samples_2cq).any().item()

    assert not has_nan_1cq, "use_2cq=False (ground truth) produced NaN — fix baseline before trusting this test"
    assert not has_inf_1cq, "use_2cq=False (ground truth) produced Inf — fix baseline before trusting this test"
    assert not has_nan_2cq, "use_2cq=True produced NaN — likely a race condition corrupting values"
    assert not has_inf_2cq, "use_2cq=True produced Inf — likely a race condition corrupting values"

    assert samples_1cq.shape == samples_2cq.shape, (
        f"shape mismatch: use_2cq=False gave {tuple(samples_1cq.shape)}, "
        f"use_2cq=True gave {tuple(samples_2cq.shape)}"
    )

    # ── Distributional comparison (loose — S=10 is not enough samples for ──
    # tight agreement; this is a sanity check, not a precision check) ──────
    mean_1cq, std_1cq = samples_1cq.mean().item(), samples_1cq.std().item()
    mean_2cq, std_2cq = samples_2cq.mean().item(), samples_2cq.std().item()

    print(
        f"\n[INFO] use_2cq=False: mean={mean_1cq:.4f} std={std_1cq:.4f}\n"
        f"       use_2cq=True:  mean={mean_2cq:.4f} std={std_2cq:.4f}"
    )

    # Loose relative tolerance on mean — both runs see the SAME deterministic
    # past_values/encoder/loc/scale, only the sampling RNG differs between
    # runs (and between use_2cq values, since they are separate ctx/device
    # instances with independent RNG state), so means should be in the same
    # ballpark but will not match exactly even with zero bugs.
    mean_diff = abs(mean_1cq - mean_2cq)
    mean_scale = max(abs(mean_1cq), abs(mean_2cq), 1e-6)
    assert mean_diff / mean_scale < 0.5, (
        f"Sample means differ by more than 50% relative ({mean_1cq:.4f} vs {mean_2cq:.4f}) "
        f"— this is far looser than expected sampling noise alone and suggests use_2cq=True "
        f"is reading stale/corrupted cache or mask state, not just different RNG draws."
    )

    print(
        "\n[INFO] test_2cq_matches_single_queue_output PASSED — use_2cq=True output is "
        "NaN/Inf-free, correctly shaped, and distributionally consistent with the "
        "trusted use_2cq=False baseline. Combined with probe_2cq_event_ordering.py's "
        "exact-match hardware result, this gives reasonable confidence the 2CQ event "
        "choreography is safe on this hardware/ttnn version — though neither check is "
        "a substitute for the full CRPS/NLL precision the e2e suite already provides "
        "for use_2cq=False."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Single-sequence latency with 2CQ  (xfail: measured no improvement)
# ─────────────────────────────────────────────────────────────────────────────
#
# This mirrors test_single_sequence_latency exactly, except use_2cq=True and
# num_cqs=2, to get a real measured number for the 2CQ path now that its
# event choreography has been hardware-probe-verified (probe_2cq_event_
# ordering.py: exact match, 8/8 steps, no cross-step contamination) and
# checked against the full model via test_2cq_matches_single_queue_output
# (NaN/Inf-free, correctly shaped, distributionally consistent with the
# trusted use_2cq=False baseline).
#
# MEASURED RESULT: median latency with use_2cq=True has varied 224.6-283.6 ms
# across repeated runs (3 separate measurements so far). use_2cq=False medians
# measured on full-suite runs in the same session were 231.6 ms and 254.8 ms --
# overlapping the same band. Run-to-run variance of this magnitude (roughly
# +/-25% of the median) is consistent with shared system-level factors
# (JIT/kernel cache state, thermal, other load) rather than a real difference
# between the two paths -- 2CQ and single-queue medians are NOT cleanly
# separable across the runs collected, i.e. 2CQ has not demonstrated a
# reproducible net latency improvement. This contradicts a prior prediction
# in this file (now corrected) that 2CQ would land around ~186 ms by
# overlapping kv_write/mask_update/host_copy for step k+1 with execute_trace
# for step k. Citing a single run's specific numbers as "the" 2CQ latency
# would be misleading given this variance; a range is the honest summary.
#
# WORKING DIAGNOSIS (not yet separately instrumented/confirmed -- stated as
# a diagnosis, not a verified fact): the autoregressive dependency between
# steps likely prevents 2CQ from finding anything useful to overlap. Step
# k+1's CPU embedding prep (_prepare_dec_step_cpu_1tok) requires
# future_samples_so_far, which requires step k's output to have been read
# back (ttnn.to_torch(ctx.traced_out)) AND sampled (_sample_next_step) --
# both blocking, both happening strictly after step k's execute_trace
# completes. So the host cannot prepare/issue step k+1's write while step
# k's trace is still running, regardless of which queue is used for what --
# there is a genuine data dependency serializing the loop at the host level,
# not merely a queue-contention problem that event choreography can route
# around. If this diagnosis is correct, 2CQ as architected cannot help this
# specific autoregressive sampling loop, independent of how well the event
# synchronization itself is implemented.
#
# Closing the 50 ms gap requires Stage 2/3 work that reduces the per-step
# critical path itself (sharding/fusion to cut the ~5.9 ms/step execute_trace
# floor, and/or reducing host-side prep/readback/sample overhead) -- not
# achievable via 2CQ alone, and 2CQ should not be assumed to help further
# pipelining work without first confirming/refuting the diagnosis above.


@pytest.mark.xfail(
    strict=False,
    reason=(
        "MEASURED: median latency with use_2cq=True has varied 224.6-283.6 ms across "
        "3 separate runs; use_2cq=False medians on full-suite runs were 231.6/254.8 ms -- "
        "the two paths are not cleanly separable given this run-to-run variance, so 2CQ "
        "has not demonstrated a reproducible net latency benefit. 2CQ event choreography "
        "is hardware-verified "
        "correct (see probe_2cq_event_ordering.py and "
        "test_2cq_matches_single_queue_output) but measured ZERO net latency benefit "
        "here. Working diagnosis (not yet separately confirmed): the autoregressive "
        "read-output -> sample -> next-step-CPU-prep dependency serializes the host "
        "loop regardless of queue choreography, since step k+1's input cannot be "
        "prepared until step k's output has been read back and sampled -- there is "
        "nothing for 2CQ to overlap in this specific loop structure, independent of "
        "correctness. Target <50 ms requires Stage 2/3: reducing the ~5.9 ms/step "
        "execute_trace floor via sharding/fusion, and/or reducing host-side "
        "prep/readback/sample overhead -- not 2CQ pipelining alone."
    ),
)
def test_single_sequence_latency_2cq(hf_model, inputs):
    B, S = 1, 100
    device = _open_device(B * S, num_cqs=2)
    try:
        weights = load_weights(hf_model, device)
        inp = _unpack_inputs(_slice_b(inputs, B))
        ctx = build_traced_decoder_context_cached(device, weights, *inp, num_parallel_samples=S)
        try:
            run_traced_generation_cached(ctx, weights, use_2cq=True)  # warmup
            latencies = []
            for _ in range(5):
                t0 = time.perf_counter()
                run_traced_generation_cached(ctx, weights, use_2cq=True)
                latencies.append((time.perf_counter() - t0) * 1000)
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)

    median_ms = statistics.median(latencies)
    print(f"\n[INFO] Latency (2CQ) — median: {median_ms:.1f} ms  all: {[f'{v:.1f}' for v in latencies]}")
    assert median_ms < 50.0, f"Latency {median_ms:.1f} ms exceeds 50 ms target."
