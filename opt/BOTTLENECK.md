# LTX bottleneck — CORRECTED (2026-07-13). Everything before this is VOID.

## Why every optimization returned zero
The developer's inference was right: attacking the "bottleneck" and seeing no movement means the
attribution was wrong. It was. Two mechanical failures:

**1. The profiler never saw the traced denoise — not once.**
`DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000` (tt_metal/impl/profiler/profiler_state_manager.cpp:19):
the profiler DRAM buffer holds **1000 programs per RISC**. LTX's eager prologue burns all 1000 before the
traced denoise begins, so 100% of the traced section is **silently dropped**. Measured: 0 of 32 devices had a
single row with a non-empty `METAL TRACE ID`, despite 5 replays logged on the host.
Also: `PROGRAM CACHE HIT` is a **structurally broken column — always False, in every run, forever**
(op_profiler_json.cpp:171-178 memoizes the cache-MISS string). The "PROGRAM CACHE HIT=False ⇒ cold" reasoning
was worthless. Use `METAL TRACE REPLAY SESSION ID` as the warm/traced witness.

**2. All historical per-op data was taken in the EAGER regime, which is host-dispatch-bound.**
Eager measures a different machine: work is nearly invisible, only op count matters. Every per-op ranking,
every A/B, every "null" came from it.

## The real bottleneck: a work-independent, OP-COUNT-driven floor
MEASURED, traced steady-state (job 103850-13, LTX_PROFILE_DENOISE_ONLY, STEP_MS timer):

| stage | tokens N | STEP_MS | spread |
|---|---|---|---|
| S1 | 9,690 | **348.3 ms** | σ=0.1 ms (8 steps) |
| S2 | 38,760 (**exactly 4×N1**) | **1092.5 ms** | σ=0.25 ms (3 steps) |

Fit t(N) = a + b·N + c·N² across the two points (N2 = 4·N1):
- pure-linear (c=0):    a = 348.3 − (1092.5−348.3)/3  = **100.2 ms**
- pure-quadratic (b=0): a = 348.3 − (1092.5−348.3)/15 = **298.7 ms**

⇒ **a ∈ [100, 299] ms/step.** Over 8 S1 + 3 S2 = 11 steps: **1.1 – 3.3 s of the 6.62 s denoise is
work-independent** — it does not scale with tokens or FLOPs. It is per-program launch + small-op FW floors +
fixed CCL latency, i.e. **OP COUNT**.

**This is the entire 8.2 → 6.0 gap, and nothing has ever attacked it.** bf8 weights, SDPA chunk, num_links,
RMSNorm stat-gather merge, adaLN fusion — every one attacks b/c (FLOPs/bandwidth). Their zero results were
structurally guaranteed. The bf8 null is *positive evidence* for the a-dominated model (bf8 changes bytes,
not program count).

⇒ **The lever is FEWER, BIGGER PROGRAMS** — real fusion of the ~195 ops/block, fewer CCL invocations —
NOT faster kernels.

## VOID (wrong regime or invalid vehicle) — do not cite these again
- The per-op ranking (CCL-matmul 56.3% / SDPA 29.8% / adaLN 8.7%): cold, EAGER, num_layers=1, random weights.
- "144 × 20.56ms = 2.96s vs 3.5s ⇒ 15% gap ⇒ compute-bound": multiplies an eager 1-layer FW by 144 and compares
  it to a traced wall. Two regimes. Circular.
- VAE-trace "0.19ms DEAD": measured as 10 PIPELINED replays — pipelining hides the per-decode dispatch gap in
  BOTH arms, forcing Δ≈0 regardless of truth. Must be re-measured as a SINGLE drained decode.
- RMSNorm QK-merge "null" (45.08 vs 44.03ms): single-block forward with per-iteration sync (>50% of the number
  is host sync). And fusion's payoff IS `a` — the vehicle structurally cannot see it. **Re-measure in the traced
  regime, scored as Δ(programs/step) and Δ(STEP_MS).** This is the lever most likely to be PROMOTED.
- "71% untracked ⇒ instrumentation blind": that log ran TT_METAL_KERNEL_CAPTURE_ONLY=1 — dispatch is skipped
  entirely (distributed.cpp:115-127), zero device ops. Void as evidence.

## The measurement that works (no tracy, no C++ rebuild)
`ttnn.ReadDeviceProfiler()` + `TT_METAL_PROFILER_CPP_POST_PROCESS=1` → `generated/profiler/.logs/cpp_device_perf_report.csv`
with OP NAME + METAL TRACE ID + REPLAY SESSION + DEVICE FW DURATION + **OP TO OP LATENCY**. Raise
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` (changes the build key ⇒ prewarm first). `LTX_PROFILE_DENOISE_ONLY=1`
skips the VAE/audio warmup (~200s) so the run fits a reservation — it is why 103850-13 is the first
full-pipeline job to ever complete (176.9s).

## ACID TEST (a profile that fails this is lying — stop, do not rank ops)
Per device d, per replay session s, over rows with a non-empty METAL TRACE ID:
  busy = Σ DEVICE FW DURATION;  gaps = Σ OP TO OP LATENCY;  span = busy + gaps
`max_d span` must reconcile with the CONTROL STEP_MS (348.3 / 1092.5) within ±10%.
PREDICTION: a_measured lands in [100, 299] ms and agrees between S1 and S2 within ~20%.
