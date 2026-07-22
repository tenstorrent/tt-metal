# Changelog — Time Series Transformer (TTNN)

Investigation history that used to live inline in module docstrings
(`tt/tst_model_cached_additions.py`, `tests/test_tst_perf.py`) now lives here.
Docstrings describe what the code does today; this file describes how it
got there and why decisions were made, so the history is preserved without
being re-read on every import.

## [Unreleased] — code quality pass

- Removed dated investigation narratives from module docstrings; moved here.
- Removed unconditional `print()` calls from the decode hot loop
  (`run_traced_generation_cached`). Timing output now goes through the
  standard `logging` module at `DEBUG` level, opt-in via `verbose=True`.
- Replaced the module-level mutable `_OP_TIMES` dict with a `StepTimer`
  instance created per call and passed explicitly — no shared global state
  across concurrent test runs or repeated calls.
- Corrected a stale comment in `tests/test_tst_perf.py`'s local
  `load_weights()` wrapper: `tst_weights.py::load_weights()` **does** accept
  a `hot_path_memory_config` parameter. The comment previously claimed it
  didn't; the parameter simply isn't being used in that file because
  Investigation 1 (below) found L1 residency made things worse.
- Documented the retirement plan for the three coexisting decode paths
  (`generate()`, `generate_traced()`, `generate_traced_cached()`) — see
  "Decode path retirement plan" below.
- Documented the PCC threshold rationale (0.99 vs 0.999) that was previously
  unstated — see "PCC threshold policy" below.

## Decode path retirement plan

Three decode implementations currently coexist in `tt/tst_model.py` and
`tt/tst_model_cached_additions.py`:

| Path | File | Role today | Retirement condition |
|---|---|---|---|
| `generate()` | `tst_model.py` | Untraced reference. Ground truth for every PCC/NLL/CRPS/MAE test. | Keep indefinitely — this is the correctness oracle every other path is checked against. Not a candidate for removal. |
| `generate_traced()` | `tst_model.py` | KV-cache-less single fused trace. Correctness gate only (`test_tst_e2e_traced.py`); never benchmarked against Stage 1 targets. | Remove once `generate_traced_cached()` has its own equivalent-strength correctness gate (PCC against `generate()` across multiple steps, not just the two tests it has today) AND once the per-layer-trace latency fix lands, since at that point this path stops earning its keep as a "simpler traced comparison." Tracked as a follow-up, not blocking this pass. |
| `generate_traced_cached()` | `tst_model_cached_additions.py` | Canonical Stage 1 inference path. | N/A — this is the target path. |

Until `generate_traced()` is removed, any change to the shared distribution
head or embedding code must keep both `generate_traced()` and
`generate_traced_cached()` passing — they are not allowed to silently
diverge, since `generate_traced()`'s only value is being a faithful (if
slower) point of comparison.

## PCC threshold policy

- **0.99** is the default threshold for every full end-to-end or per-layer
  comparison (`test_tst_pcc.py`, `test_tst_embedding_pcc.py`): these
  accumulate bfloat16 rounding error across many chained ops (attention,
  layer norm, FFN, multiple matmuls), so 0.99 is the realistic ceiling for
  this stack at this precision.
- **0.999** is used only in `test_tst_dist_head_fusion_pcc.py` and
  `test_tst_dist_head_fusion_traced_pcc.py`: these test a single isolated
  linear-projection head (2-3 matmuls, no attention, no accumulated
  layer-norm rounding), on both host and device from the *same* loaded
  weights. Far less numerical drift is expected at that shallow depth, so a
  looser 0.99 threshold there would hide a real transpose/layout bug rather
  than catch one.
- Rule going forward: threshold is a function of op-chain depth, not a
  per-test guess. New PCC tests should default to 0.99 unless they are
  testing an isolated shallow op (≤3 ops deep) against the same weights on
  both sides, in which case 0.999 applies.

## Investigation 1 — L1-resident hot-path decoder weights (2026-07-13)

**Hypothesis:** `load_weights()` already threads a `hot_path_memory_config`
parameter through every decoder-layer weight builder, but every
`test_tst_perf.py` call used the default (`None` → DRAM). Weight footprint
(`scripts/measure_weight_footprint.py`) is 0.236 MB, 0.3% of pooled L1 on
this grid, and decoder hot-path weights are already tile-aligned
(`PADDED_WIDTH=64`), so footprint and alignment were not the blocker.

**Result: hypothesis refuted.** L1-resident hot-path weights measured
*worse*, not better:

| Metric | DRAM | L1 | Delta |
|---|---|---|---|
| `test_single_sequence_latency` | 98.0ms | 117.6ms | +20% (worse) |
| `test_throughput_seqs_per_second` | 28.2 seq/s | 25.4 seq/s | -10% (worse) |
| `test_single_sequence_latency_2cq` | 104.0ms | 102.3ms | ~flat (noise) |

`qkv_linear` alone went from ~7ms to ~11-17ms per step. Per-replay timing
also showed accumulating degradation unique to this config (fast for ~6
replays, then jumps ~20ms and stays there), consistent with pooled-L1
contention between weights, KV-cache, trace, and mask buffers competing for
on-chip space — a mechanism the static footprint check couldn't have
caught, since it only measures size, not runtime contention.

`HOT_PATH_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG` is kept defined in
`test_tst_perf.py` but unused, as a marker not to re-attempt this without
first profiling actual L1 occupancy/fragmentation during decode.

## Warmup fix (prior revision, 2026-07-09)

10 replays immediately after ctx build, only 1 prior warmup call:
`120.6, 124.6, 120.2, 97.4, 99.2, 98.4, 98.8, 99.1, 99.9, 99.6` (ms).

Three elevated replays (120-125ms) before settling to a 97-99ms steady
state. One warmup call was not enough to clear whatever host-side dispatch
caching resolves on first real replay after trace capture. Warmup bumped to
5 discarded replays (`WARMUP_REPLAYS = 5`) on all latency-sensitive tests,
with margin past the observed 3-replay settling point.

## KV-cache fused op for V, re-enabled (2026-07-1x)

The Stage 2 "Change 4" xfail in `test_tst_stage2.py` previously claimed "no
measured evidence the TILE↔ROW_MAJOR conversion tax nets positive" for
using `ttnn.update_cache` on the V cache. This was incorrect: at BS==1, V
cache is already allocated `TILE_LAYOUT` (see `allocate_kv_cache`), so
there is no conversion tax on this path to begin with. A prior session had
already built and correctness-verified this exact change
(`tests/diagnostics/probe_v_cache_update_cache*.py`: exact match vs.
`slice_write` ground truth, multistep, on real hardware) but it was
reverted without the xfail author consulting those probes.

Re-applied and re-verified on hardware. Correctness unaffected (encoder PCC
0.9999925, decoder PCC 0.9999812, CRPS/NLL/MAE all within 5% threshold).

| Metric | Before | After | Change |
|---|---|---|---|
| `write_prep`/step | ~47ms | ~36ms | -11ms |
| `slice_write_kv`/step | ~27ms | ~20ms | -7ms |
| `test_single_sequence_latency` median | 124.3ms | 111.8ms | -10% |
| `test_single_sequence_latency_2cq` median | 99.6ms | 103.3ms | within noise |

Real, correctness-verified improvement. Does not close the gap to Stage 1
targets (<50ms latency, ≥100 seq/s throughput) on its own — no single
op-level change is expected to, since cost is spread roughly evenly across
`write_prep`, `trace_exec`, and `readback`/`sample`/`cpu_prep`.

## FlashDecode — evaluated and reverted

`ttnn.transformer.scaled_dot_product_attention_decode` was tried in place
of the manual Q@K_cache attend-from-cache. Reverted: it parallelizes one
core (or core-group) per batch element and hits
`TT_FATAL: num_cores_available >= B` once `B = batch * num_parallel_samples`
exceeds the 56-core Wormhole grid — exactly the regime the bounty's
"100 samples in <1s" / "100+ seq/s" targets require. Built for
small-batch/long-context LLM serving; this model's access pattern is the
inverse (large effective batch, short `T_max=24` context). The manual
matmul attend-from-cache is PCC-verified (0.9999927) against the
FlashDecode path it replaces and has no batch-size ceiling.

## Layer-threading correctness bug (root-caused, fixed)

`generate_traced_cached()` originally captured a single fused trace for
both decoder layers. Layer 1's Q/K/V were being projected from the raw
input embedding instead of layer 0's real output, so layer 1 never actually
attended over layer 0's content. Invisible to `test_tst_pcc.py` /
`test_tst_e2e.py` (untraced path only) and to the pre-existing
`test_tst_e2e_traced.py` (KV-cache-less traced path only). Caught via
direct PCC comparison against `generate()`: ~0.31-0.33 vs the ~0.999
expected.

**Fix:** one trace per decoder layer, chained through real device buffers —
layer i's real output feeds layer i+1's real input. Correct, but costs
`num_decoder_layers` separate `execute_trace` calls per autoregressive step
instead of 1. This is the leading known cause of the current latency/
throughput gap to Stage 1 targets and is not yet re-optimized — see
`tt/tst_model_cached_additions.py`'s module docstring for the current
state of that work.

## XFAIL removal (perf tests)

Tests 1, 2, and 5 in `test_tst_perf.py` previously conflated
`S = num_parallel_samples` with the measured workload (B=1 S=100 instead of
B=1 S=1 for "single sequence latency"; S=10 instead of S=1 for
throughput). Fixed — all three now use S=1. The xfail markers that existed
while that bug was open have been removed: a failing perf test now is an
honest measurement against the correct workload, not a placeholder.
