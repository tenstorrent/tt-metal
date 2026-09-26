"""Answers to the two follow-up questions on the traced CFM solver's lifecycle:

1. Is the captured trace cached/reused across DIFFERENT utterance lengths within a
   process, or does every new length pay a fresh capture? (single-slot cache, keyed by
   (t_len, channels) -- see TtCausalConditionalCFM._trace_key_for)
2. What's one-time (per NEW shape) vs steady-state (every solve at an ALREADY-captured
   shape)?

Uses random (not real-checkpoint) weights -- kernel compile cost depends on TENSOR SHAPE,
not weight VALUES, so this is a valid, much faster way to isolate the trace lifecycle
question from checkpoint-loading overhead. Every length used here (333, 400, 510) is
picked to include both a genuinely fresh shape (333, 400 -- never compiled by any earlier
script today) and a shape already compiled by an earlier script today (510, from the
2026-09-22 items 2-4 run) to isolate ttnn's own on-disk kernel-compile cache (persists
across PROCESSES) from this class's own in-process trace-object cache (single slot, does
NOT persist across a release).

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 600 /opt/venv/bin/python trace_lifecycle_experiment.py
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/home/user/tt-metal")

from models.demos.audio.cosyvoice2.tt.flow.decoder import (
    CausalConditionalCFMRef,
    CausalConditionalDecoderRef,
    TtCausalConditionalCFM,
    TtCausalConditionalDecoder,
)

device = ttnn.CreateDevice(0, l1_small_size=32768, trace_region_size=100_000_000)


def sync():
    ttnn.synchronize_device(device)


def make_inputs(t_len, seed):
    torch.manual_seed(seed)
    mu = torch.randn(1, t_len, 80) * 0.1
    cond = torch.randn(1, t_len, 80) * 0.1
    spks = torch.randn(1, 80) * 0.1
    mask = torch.ones(1, t_len, 1)
    return mu, mask, spks, cond


try:
    torch.manual_seed(0)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    cfm_ref = CausalConditionalCFMRef(dec)
    tt_dec = TtCausalConditionalDecoder(device, dec)
    tt_cfm = TtCausalConditionalCFM(device, tt_dec, cfm_ref.rand_noise, cfm_ref)

    print("=== PART 1: does one CFM instance cache across DIFFERENT lengths, or single-slot? ===")
    T_A, T_B = 333, 400  # both never compiled before today
    mu_a, mask_a, spks_a, cond_a = make_inputs(T_A, 1)
    mu_b, mask_b, spks_b, cond_b = make_inputs(T_B, 2)

    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_a, mask_a, 10, spks_a, cond_a, use_trace=True)
    sync()
    t_capture_a = time.perf_counter() - t0
    print(f"[T={T_A}] first capture (genuinely new shape, cold kernel cache): {t_capture_a*1000:8.1f} ms")

    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_a, mask_a, 10, spks_a, cond_a, use_trace=True)
    sync()
    t_replay_a = time.perf_counter() - t0
    print(f"[T={T_A}] SAME length again (single-slot cache HIT -> replay only): {t_replay_a*1000:8.1f} ms")

    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_b, mask_b, 10, spks_b, cond_b, use_trace=True)
    sync()
    t_capture_b = time.perf_counter() - t0
    print(f"[T={T_B}] DIFFERENT length (new shape -> single slot evicted, fresh capture): {t_capture_b*1000:8.1f} ms")
    print(
        "  -> if this pays a full fresh-capture cost (comparable to T_A's), the cache is confirmed "
        "single-slot: a new length always evicts the old one, never held alongside it."
    )

    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_a, mask_a, 10, spks_a, cond_a, use_trace=True)
    sync()
    t_recapture_a = time.perf_counter() - t0
    print(
        f"[T={T_A}] back to the FIRST length (evicted by T_B -> must recapture): {t_recapture_a*1000:8.1f} ms  "
        f"(vs its OWN first-ever capture: {t_capture_a*1000:.1f} ms -- kernels for T={T_A} are now warm from "
        f"the earlier capture, so this isolates recapture-with-warm-kernels cost from cold-compile cost)"
    )
    tt_cfm.release_cfm_trace()

    print("\n=== PART 2: steady-state cost once a shape's trace already exists (many solves, no release) ===")
    T_C = 510  # already compiled by an earlier script today -- disk kernel cache is warm even on first capture here
    mu_c, mask_c, spks_c, cond_c = make_inputs(T_C, 3)
    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_c, mask_c, 10, spks_c, cond_c, use_trace=True)
    sync()
    t_first_c = time.perf_counter() - t0
    print(f"[T={T_C}] first capture THIS INSTANCE (kernels already warm on disk from an earlier script today): {t_first_c*1000:8.1f} ms")

    N = 8
    steady_times = []
    for i in range(N):
        mu_i, mask_i, spks_i, cond_i = make_inputs(T_C, 100 + i)  # fresh conditioning each "utterance"
        sync()
        t0 = time.perf_counter()
        tt_cfm.forward(mu_i, mask_i, 10, spks_i, cond_i, use_trace=True)
        sync()
        steady_times.append(time.perf_counter() - t0)
    print(f"[T={T_C}] {N} more solves at the SAME shape, no release between them (real per-utterance steady state):")
    print("  " + "  ".join(f"{t*1000:.1f}" for t in steady_times) + " ms")
    print(f"  mean: {sum(steady_times)/len(steady_times)*1000:.1f} ms  (= {sum(steady_times)/len(steady_times)/10*1000:.2f} ms/Euler-step)")
    tt_cfm.release_cfm_trace()

    print("\n=== PART 3: eager-only baseline at a genuinely fresh shape, for direct comparison ===")
    T_D = 450  # never used before today, either
    mu_d, mask_d, spks_d, cond_d = make_inputs(T_D, 4)
    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_d, mask_d, 10, spks_d, cond_d, use_trace=False)
    sync()
    t_eager_cold_d = time.perf_counter() - t0
    print(f"[T={T_D}] EAGER (no trace), first-ever call at this shape (cold kernel compile): {t_eager_cold_d*1000:8.1f} ms")

    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_d, mask_d, 10, spks_d, cond_d, use_trace=False)
    sync()
    t_eager_warm_d = time.perf_counter() - t0
    print(f"[T={T_D}] EAGER, second call (kernels now warm): {t_eager_warm_d*1000:8.1f} ms")

    sync()
    t0 = time.perf_counter()
    tt_cfm.forward(mu_d, mask_d, 10, spks_d, cond_d, use_trace=True)
    sync()
    t_traced_cold_d = time.perf_counter() - t0
    print(
        f"[T={T_D}] TRACED first capture at this SAME shape (kernels warm from the eager calls just above): "
        f"{t_traced_cold_d*1000:8.1f} ms"
    )
    tt_cfm.release_cfm_trace()
finally:
    ttnn.CloseDevice(device)
