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

## LTX per-stage visibility toolchain (`tools/tracy`)

Three pure-Python tools (no build; run against any tree's CSVs) turn the per-op CSV into a
served-latency budget. See each script's module docstring for the full contract.

### `ltx_stage_bottlenecks.py` — per-stage rollup + host rows + e2e budget

Rolls the per-op CSV into the canonical LTX stages, ranks the dominant ops per stage, tags each
compute/bandwidth/dispatch-bound, and checks the total against an e2e budget. **Host phases are
first-class rows**, not footnotes: the pipeline's own wall-clock timers (`Transformer prepare` = DiT
reload ~2.1s, `VAE prepare`, `Latent upsample`, `Video export` ~1.6s, I2V `Image encode`) are parsed
from the generate() stderr (`--log`) and shown next to the device stages, so the dominant served
levers (reload + export) are visible in the same budget. Host stages carry no device op — they show
`—` for device time and count only toward the wall-clock budget.

```bash
python tools/tracy/ltx_stage_bottlenecks.py \
  --csv "Stage 2 denoise=ops_perf_results_*.csv" --log gen.stderr.log \
  --budget 6.0 --stage-budgets "Stage 2 denoise=1.5" --html rollup.html
```

**Budget / regression gate.** `--save-baseline base.json` snapshots per-stage device ms + wall;
a later run with `--baseline base.json` shows a Δ% column and flags any stage whose device time (or
wall, for a host stage) grew past `--regress-tol` (default 5%). `--gate` exits nonzero on any
over-budget or regressed stage, so a loop can detect a per-iteration regression.

### `test_ltx_stage_scoped.py` — stage-segmented capture selector

`LTX_PROFILE_STAGES=vae,audio,upsample` generalizes the single-op `test_ccl_allgather_scoped` to
whole stages: the harness runs each selected decode-tail stage on a synthetic served-shape latent
and **drains the profiler (`ttnn.ReadDeviceProfiler`) at each stage boundary**, so the chosen stages'
perf-counter markers land in the CSV instead of being dropped when a dense block overflows the marker
DRAM buffer. `audio` alone builds an audio-only pipeline (no 22B transformer). The dense denoise
blocks (`s1`/`stage2`) need a transformer forward and overflow even alone — the harness skips them
with a message; capture them from the full gen (drained per stage) or `test_transformer_ltx`.

### `ltx_profile_all.py` — multi-profile merge (whole pipeline from several captures)

`merge` stitches N scoped per-op CSVs (`STAGE=path`) into one whole-pipeline table — rebasing GLOBAL
CALL COUNT into per-stage bands so a stage from one run never collides with another, tagging each row
with its STAGE, last-capture-wins per stage — then feeds it to `ltx_stage_bottlenecks` for a single
rollup. `plan` prints the prewarmed per-segment capture commands that cover the pipeline.

```bash
python tools/tracy/ltx_profile_all.py merge \
  --csv "Stage 1 denoise=s1.csv" --csv "Stage 2 denoise=s2.csv" \
  --csv "VAE decode=vae.csv" --csv "Audio decode=audio.csv" \
  --out merged_pipeline.csv --log gen.stderr.log --html whole_pipeline.html
```

## Segmentation map — which stages fit one perf-counter pass

The counter marker buffer overflows on a full denoise block, so the capture must be scoped. Measured
per-device op/zone counts (bh_2x4sp1tp0, 1088×1920, LTX_FAST) and whether they fit one counter pass:

| Segment | Stages | Ops/device | Fits one pass | How to capture |
|---|---|---:|---|---|
| decode-tail | VAE decode + Audio decode (+ Latent upsample) | see capture | yes (drained per stage) | `test_ltx_stage_scoped.py LTX_PROFILE_STAGES=vae,audio` |
| one transformer block | Stage-1 **or** Stage-2 denoise, single block | 354 | marginal — 1 block only | `test_transformer_ltx` block harness |
| full denoise | Stage-1 or Stage-2, all steps | 354 × N_steps | **no — overflows** | full-gen device-time CSV (no counters), or per-block scoped |

Rule of thumb: one transformer block ≈ 354 device ops (≈ 5.8k noc markers/core for its stage-2
AllGather alone — two overflow the ~11.7k/core buffer). The decode tail is light and, drained per
stage boundary, fits one counter pass; each dense denoise block is captured on its own. The merge
driver stitches a decode-tail counter capture with the denoise device-time capture into one ranking.

## Analytical CCL BW % — all-gather, reduce-scatter, all-reduce

The analytical fabric-BW model (`noc_bandwidth.collective_fabric_bw`, exact from output shape +
`ring_size` + topology, no `--collect-noc-traces`) now covers all three collectives, not just
all-gather. All-gather and reduce-scatter are duals that each move `(N-1)/N` of the full tensor;
all-reduce moves `2·(N-1)/N`. Reduce-scatter's OUTPUT is the `1/N` scattered chunk, so its full
tensor is `output·N` — the kind (`_collective_kind` in `process_ops_logs.py`) is what lets one op
record pick the right formula. Device-validated on a real `ReduceScatterMinimalAsyncDeviceOperation`
(LTX stage-2, 2×4 BH, ring_size 2, 2 links, Linear): 37.6 GB/s/link ≈ 75% of a 50 GB/s link — a
column that was blank before. Unit coverage in `tests/ttnn/tracy/test_noc_bandwidth.py`.

## Host-zone Tracy timeline — proposed, not delivered (needs a serving-tree rebuild)

Wrapping the Python generate() phases in real Tracy HOST zones (dispatch gaps visible alongside
device zones) is feasible but **out of scope for a no-rebuild deliverable**: tt-metal's host-zone
path (`tracy::ScopedZone` / the `TracyCZone` C macros) is C++, reachable from Python only via a
pybind shim that does not exist today, so adding it means new C++ bindings + a rebuild of the LIVE
serving tree. The wall-clock host rows above already surface the same dominant phases (reload,
export) from the existing stderr timers with zero pipeline change; the Tracy host timeline would add
intra-phase dispatch-gap detail only. Deliver it in a dedicated instrumentation change when the
serving tree is next rebuilt.

## Open items (not yet implemented)

- **Readback-time** core restriction — deferred; correct per-op-grid selection
  needs op→grid association that only exists in post-process.
- **Denoise per-op counters in one pass** — a full block overflows; today scoped
  to one block or captured as device-time only. A mid-run drain hook per block
  (like `prof_girl_decode`) would extend counter coverage to full denoise.
