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

## NoC bytes per op

When noc traces are captured (`--collect-noc-traces`) the post-process emits a
`NOC BYTES FROM COUNTERS` column: Σ num_bytes over the op's cores, straight from
the profiler's own `noc_trace*.json` — independent of tt-npe (so it populates on
hosts without tt-npe built). BW % = bytes / (DEVICE FW DURATION × peak) is
applied by the analysis layer with the part peak (`tools/tracy/noc_bandwidth.py`).
Verified: an 8192² bf16 eltwise add reports exactly 402,653,184 bytes
(2 reads + 1 write). Single-chip `--collect-noc-traces` works; the documented
ethernet hang is fabric/multi-chip init only.

## C++ fast-path counters (Phase 2a — also the Phase 2b OOM fix)

Counter capture now stays on the C++ fast post-process path; it no longer forces
the legacy `pd.read_csv` of the full per-core device log (the 140 GB OOM at mesh
scale). `impl/profiler/perf_counter_metrics.{hpp,cpp}` ports the full
`tools/tracy/perf_counter_analysis.py::compute_perf_counter_metrics` (~200 columns)
to C++: it reads the per-(op,core,counter) values straight from the in-memory
`id==9090` device markers (`PerfCounter(marker.data, marker.data_high)` →
value/ref_cnt/type), pivots first-wins per (op,core,counter), and emits the
canonical `PERF_COUNTER_CSV_HEADERS` columns into `cpp_device_perf_report.csv`.
Memory stays bounded (only the 9090 markers, not every per-core marker). The CLI
gate that disabled cpp post-process for `--profiler-capture-perf-counters` is
removed (`tools/tracy/__main__.py`).

Verified by `tests/ttnn/tracy/test_counter_utilization_sanity.py` (3/3 on BH
p150b): FPU utilization counter matches achieved FLOPS, eltwise reads
bandwidth-bound, and compute vs bandwidth separate — all on the cpp path.

Build gotcha (cost a full debug loop): an incremental `cmake --build build` links
`libtt_metal.so`/`_ttnncpp.so` into `build_Release/{tt_metal,ttnn}/`, but the
process loads them from `build_Release/lib/` via RUNPATH — only `cmake --install
build` refreshes `lib/`. Build **and** install, or the device silently runs stale
profiler code.

## Counters + noc-traces together (Phase 3b — the "multi-chip hang" was a crash)

Running `--profiler-capture-perf-counters` together with `--collect-noc-traces`
aborted in `coalesceFabricEvents` (`TT_FATAL ... Invalid NoC transfer type`) —
not the device hang the plan assumed. Root cause: perf-counter markers
(`marker_id == PERF_COUNTER_PROFILER_ID`) share the `TS_DATA` marker type with
noc-event markers, so they leaked into `timestamped_datapoints_by_op` and the
fabric coalescer decoded their counter payload as `NocEventMetadata`, producing a
bogus `noc_xfer_type`. Fixed by excluding counter markers from that filter
(`profiler.cpp`, `convertNocTracePacketsToJson`). Verified on BH 2x4 (8-chip,
FABRIC_1D): `fpu,instrn` + `--collect-noc-traces` now runs clean and emits
per-device `noc_trace_*.json` for all 8 chips.

Caveat still open: a full 928-op block overflows the profiler DRAM marker buffer
("buffers were full, markers were dropped") — noc BW is undercounted until the
capture is scoped (fewer ops / `--dump-device-data-mid-run` to drain mid-run).

## Delivered

- **CCL fabric BW % (Phase 3 gate).** A full-grid gather overflows the noc-trace
  marker buffer, so its bytes can't be read back from a trace — but they are exact
  from output shape + `ring_size` + topology. `process_ops_logs.py` computes
  `CCL FABRIC BW [GB/s]`/`UTIL (%)` analytically from the op record and the real
  trained per-link speed, needing NO `--collect-noc-traces`. Device-validated on
  BH 2x4: stage-2 TP all-gather 21 GB/s/link, ~42% of the 400G peak.
- **Phase 4 zone digest.** `profiler.cpp` aggregates each op's counters into a
  compact digest on the op FW zone's `meta_data["zone_summary"]` and logs it
  (`[HW FPU=… MATH=…]`) so it is verifiable from the device log. Surfacing it as a
  GUI hover label is a one-line read in `TracyTTDevice.hpp::PushStartMarker`; that
  vendored-fork change is preserved as `tracy-phase4-zone-digest.patch` (the fork
  remote is not pushable from here, so the submodule pointer stays at upstream to
  keep the branch buildable). Apply the patch + rebuild Tracy to see the tooltip.

## Open items (not yet implemented)

- **Readback-time** core restriction — deferred; correct per-op-grid selection
  needs op→grid association that only exists in post-process.
