# HW perf counters: capture, sizing, and utilization

Per-zone hardware counters (FPU/SFPU/MATH, unpack/pack, instruction-stall and
semaphore-wait, L1/NoC ports) turn ambiguous per-RISC durations into a real
bound classification. This is the model-agnostic recipe for capturing them
losslessly, sizing the buffer, scoping the capture, and trusting the numbers.

## Counter groups

Enable groups with `--profiler-capture-perf-counters` (sets the
`TT_METAL_PROFILE_PERF_COUNTERS` bitfield):

| Group | Bit | Notes |
|---|---|---|
| `fpu` | 0 | FPU/SFPU/MATH active cycles |
| `sfpu` | 0 | alias for `fpu` (SFPU rides the FPU register path) |
| `pack` | 1 | |
| `unpack` | 2 | |
| `l1_0` | 3 | L1 bank 0 / NOC ring 0 ports |
| `l1_1` | 4 | L1 bank 1 / NOC ring 1 — shares the mux with l1_0, needs a 2nd pass |
| `instrn` | 5 | per-thread stalls, semaphore waits |
| `l1_2`,`l1_3`,`l1_4` | 6,7,8 | Blackhole only |
| `all` | — | `fpu｜pack｜unpack｜l1_0｜instrn` in one pass |

L1 banks share one mux, so only one L1 bank is active per run. The CLI runs a
second pass for `l1_1` and merges by zone key automatically.

## Utilization math — what the numbers mean

FPU util % = `FPU_COUNTER / ref_cnt × 100` = FPU-active cycles / total cycles
(cycle occupancy), **not** a throughput-vs-peak figure directly. They coincide
for a clean matmul: occupancy IS the fraction of the fidelity peak reached.
Two columns land in `ops_perf_results_*.csv`:

- `FPU Util Min/Median/Max/Avg (%)` — per-core occupancy distribution.
- `Avg FPU util on full grid (%)` — Σ over cores / full grid size; comparable
  to the achieved fraction of the chip-wide peak.

`PM FPU UTIL (%)` is a separate, **analytical** estimate from the op's
performance model (`100 × PM COMPUTE / DEVICE KERNEL DURATION`); a HW counter
near the perf-model value corroborates both.

### Trust the numbers first

`tests/ttnn/tracy/test_counter_utilization_sanity.py` runs a known-FLOP matmul
and known-byte eltwise and asserts the grid-normalized FPU counter matches the
achieved fraction of the analytical peak. Run it after any change to the
counter capture or util math. Verified on bh 4x8: FPU counter 21.9% vs achieved
21.5% of the HiFi2 peak; eltwise 77.6% DRAM BW with the FPU idle. (Note the
chip reports 130 functional compute cores; the peak table assumes 140 — the
~7% basis difference is within the sanity tolerance.)

## Buffer sizing — derive it, then pin it

Each enabled group emits one marker per counter per zone on BRISC. Counts are
fixed per arch (`hw_counters.h`); `tools/tracy/perf_counter_sizing.py` mirrors
them:

| Group | Blackhole | Wormhole |
|---|---|---|
| fpu | 3 | 3 |
| pack | 5 | 14 |
| unpack | 22 | 22 |
| l1_0..4 | 16 each | 16,16,0,0,0 |
| instrn | 59 | 59 |

```python
from tracy.perf_counter_sizing import markers_per_zone, recommend_program_support_count
markers_per_zone("blackhole", ["fpu", "instrn"])      # 62 markers/zone
recommend_program_support_count(4400)                  # buffer to fit 4400 ops/device
```

All nine BH groups together are 169 markers/zone — under the 250-slot L1
optional-marker budget, so the per-zone L1 vector never overflows on BH. The
real overflow is the **DRAM program-support count**
(`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`): it must cover the distinct
programs/zones per device across the whole run.

**Pin it.** The count is baked into the kernel build define, so changing it
re-hashes every kernel → full instrumented recompile (~14 min). Pick one value
and reuse it: the second run at the same value hits 100% JIT cache.

## Scope control

- **Compute counters → sample per op-grid.** `TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE=K`
  keeps K cores per op (evenly spaced over the op's own grid, deterministic),
  shrinking the per-core row explosion that drives post-process cost. It
  preserves the per-core util distribution (Min/Median/Max/Avg %); it does
  **not** preserve grid-summed metrics (`Avg ... on full grid`). Sampled from
  each op's recorded grid — a fixed device mask would silently drop ops off it.
- **NoC counters → never sample.** Bandwidth is an across-cores aggregate;
  per-core NoC load is non-uniform, so sampling undercounts BW. Capture all
  NoC-active cores and aggregate per op.
- **Counter-group subset.** Capture only the groups you need — fewer markers.

## One-command capture

```bash
python -m tracy.capture_counters \
  --test "tests/ttnn/tracy/test_counter_utilization_sanity.py" \
  --groups fpu,instrn --programs-per-device 4400 --compute-core-sample 3
```

Pins the buffer to cover the workload, applies the env, runs under tracy, and
archives the CSV + `.tracy` + the exact scope to `~/traces/<ts>/` (the reports
dir is wiped each run). Reuse the same `--programs-per-device` across runs to
keep the kernel hash stable.

## Repeatability

- Pin `PROGRAM_SUPPORT_COUNT` (constant kernel hash → 100% JIT cache).
- Deterministic core sampling (same grid → same cores).
- Profile a warm window; never rank the cold/load sum.
- Re-run `process_ops_logs` on the existing `generated/profiler/.logs` to
  iterate on post-processing without re-running on device.

## Open items (not yet implemented)

- **NoC/DRAM/ETH BW % folded into the per-zone row** from the on-device
  `noc_status_counter` (Phase 3a) — interim source until the noc-trace ethernet
  hang (Phase 3b) is root-caused on device.
- **Per-RISC C++ fast post-process** carrying counters (Phase 2a) and a
  **streaming/bounded-memory** Python path (Phase 2b) for mesh-scale OOM.
- **Tracy GUI zone tooltips** (Phase 4) — a vendored-fork protocol change.
- **Readback-time** core restriction — deferred; correct per-op-grid selection
  needs op→grid association that only exists in post-process.
