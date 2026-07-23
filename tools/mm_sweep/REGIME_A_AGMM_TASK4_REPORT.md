# AGMM Task 4 — DRAM-staged phase: measurement & characterization

Fused op: `ttnn.experimental.all_gather_regime_a_matmul_async` (streaming DRAM-staged, Phase A), measured
AFTER the F3 (local-first per-shard readiness) and F5 (production IN1_NEAR + PARETO placement) fixes.
Machine: single Blackhole galaxy, container `trusting_borg`, worktree `resilient-marinating-piglet`.
Harness: `tools/mm_sweep/agmm_profile.py` (per-RISC device profiler; 12 launches/config, warmup dropped,
median + [min-max] over the profiled devices/runs). Single-chip reference: `agmm_sc.py` (regime_a, D=1).

Profiler + watcher are usable here (not blocked, contrary to the first Task-4 pass): pre-create the tracy
dirs + set `TT_METAL_HOME` + tear the submesh down cleanly (reset stall group / clear+remove sub-device
manager / close submesh then parent) so `ReadMeshDeviceProfilerResults` flushes the CSV; run the watcher with
`TT_METAL_WATCHER_DISABLE_ETH=1` (the 19-test suite is watcher-clean).

## Per-RISC A/B, D=4 ring, 1 link / 1 injector, config=None. All PCC ≈ 1.0000.

Total device span = max kernel end − min kernel start over all cores on a device (median over runs [min-max]).
Injector span = the fabric injector's kernel span (gather cost). Single-chip MM = regime_a compute span for
the full [M,K,N] on one device (the pure-compute reference; no gather).

| shape (M,K,N)   | M/N   | stream total µs      | full-gather total µs | injector (gather) µs | single-chip MM µs |
|-----------------|-------|----------------------|----------------------|----------------------|-------------------|
| 32,6144,3072    | 0.010 | **138.4** [133-146]  | 143.3 [134-155]      | 62                   | 91.3              |
| 32,15360,768    | 0.042 | **170.4** [166-179]  | 176.4 [171-185]      | 124                  | 58.0              |
| 128,6144,2304   | 0.056 | **255.6** [249-261]  | 261.1 [256-268]      | 190                  | 76.8              |
| 256,6144,768    | 0.333 | **395.1** [386-412]  | 403.9 [396-416]      | 366                  | 44.2              |

## Findings

1. **Streaming overlap is real and consistent** (after F3 local-first): streaming beats the same-binary
   full-gather diagnostic on every shape by ~5-9 µs (~2-3%). Before F3's per-shard local-first fix the two were
   indistinguishable (global-prefix ordering forced every device to wait for remote shard 0). The win is
   modest because the tail that streaming removes is small relative to the binding stage — see (2).

2. **The op moves from compute-bound to strongly fabric-bound as M/N grows**, exactly as the plan's roofline
   predicts (`T_fabric/T_in1 ≈ (M/N)·(D-1)/D`). At 32x6144x3072 the injector/gather (62 µs) is smaller than the
   single-chip matmul (91 µs) → compute-bound, gather largely hidden. At 256x6144x768 the gather (366 µs) is
   ~8x the single-chip matmul (44 µs) → the op is gather-limited; total span (395 µs) tracks the injector, not
   the compute. So on the fabric-bound shapes the dominant lever is **gather cost** (fewer bytes / direct-L1 /
   more links / neighbor-forward chain), NOT more overlap.

3. **Kernel spans include waits — no isolated compute-only or "direct-L1 ceiling" number is claimed.** The
   fused TRISC "compute" span is dominated by the reader blocking on gather readiness (e.g. 394 µs at
   256x6144x768, mostly the 366 µs gather wait), so it is NOT a clean compute-only measurement and cannot be
   differenced against single-chip MM to isolate the "second in0 DRAM read". (My earlier ~55 µs ceiling was not
   isolated — retracted.) A clean first-compute-latency / DRAM-read-overhead number needs custom in-kernel
   profiler zones (mark first matmul math) or the Phase-B direct-L1 A/B itself; treat that as Task 5/6 input.

4. **Not yet measured** (deferred, needs more harness): standalone `all_gather` and unfused `AG→MM` wall/spans
   for the same shards; effective fabric bandwidth; 4 KiB vs 8 KiB packet A/B (Phase A sends one 2 KiB tile per
   packet, inherent to the interleaved-DRAM gather buffer — a real 4 KiB baseline needs Phase-B L1 slots);
   delivered DRAM BW; one/two-link sweep. These are Task 4-extended / Task 7-8 items.

## Recommendation for Phase B (Task 5)

Direct remote-L1 streaming should target the **gather cost** that binds the fabric-bound shapes (large M/N),
not additional overlap (already near-complete). Measure it with `agmm_profile.py` against the exact staged-DRAM
config, and add custom first-math zones so the compute-vs-wait split is separable. Do not adopt direct-L1
without a stable per-corpus win.
