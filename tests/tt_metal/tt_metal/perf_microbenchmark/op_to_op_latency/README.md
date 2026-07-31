# `op_to_op_latency` — back-to-back program latency benchmark

Measures per-core latency from when one program finishes (pack pushed its last
tile) to when the next program starts compute (unpack begins tile 0).

## What the benchmark does

On every Tensix core:

```
reader (NOC1) ──CB_in──▶ compute (3 TRISCs) ──CB_out──▶ writer (NOC0)
   interleaved DRAM       UNPACK / MATH / PACK         write + barrier
```

Compute kernel profiler markers (device CSV):

| TRISC | Role | Marker |
|-------|------|--------|
| TRISC_0 | Unpack | `TILE_IDX` = *i* at tile *i* compute start (`UNPACK(...)`) |
| TRISC_1 | Math | optional `MATH` zones around copy + NOPs |
| TRISC_2 | Pack | `FINISH_LAST_PUSH` once at kernel exit, outside the tile loop (`PACK(...)`) |
| NCRISC (or RISCV_1) | Reader | `NCRISC_GO` / `NCRISC_DONE` at kernel entry/exit; `READ_BEFORE/AFTER_BARRIER` on tile 0 |
| BRISC (or RISCV_0) | Writer | `BRISC_GO` / `BRISC_DONE` at kernel entry/exit; `WRITE_BEFORE/AFTER_BARRIER` on last tile |

All kernels emit `PROG_ID` (0 = warmup / pre-compile, 1..N = timed).

The same `MeshWorkload` is enqueued back-to-back `--num-programs N` times, in
Fast-Dispatch mode or Trace mode (capture once, replay once).

## Build

```bash
# Kernels pick up PROFILE_KERNEL only if TT_METAL_DEVICE_PROFILER=1 is set
# before the process starts. Clear JIT cache after changing profiler macros:
#   rm -rf ~/.cache/tt-metal-cache "$TT_METAL_CACHE"

./build_metal.sh --build-tests
cmake --build build_Release --target test_op_to_op_latency -j
```

## Quick start

```bash
export TT_METAL_HOME=$(pwd)
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_DISPATCH=1   # needed for chip-level dispatch markers used by --use-realtime-profiler

cmake --build build_Release --target test_op_to_op_latency -j

./build_Release/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --use-trace \
  --num-programs 2 \
  --compute-nops 1000 \
  --use-device-profiler \
  --use-realtime-profiler

python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/export_op_to_op_profiler_csv.py \
  --input-file "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
```

### Recommended steady-state configuration

For a representative measurement (per-trid double-buffered reader, writer flushed on back-pressure,
end-of-kernel `noc_async_writes_flushed` instead of full barrier, trace mode with warmup, drop the
first two transitions in the export):

```bash
./build_Release/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
  --num-active-cores 80 \
  --input-cb-depth-tiles 12 --output-cb-depth-tiles 16 \
  --num-pages-per-core 16 --compute-nops 2000 \
  --num-programs 8 --use-trace --trace-warmup-replays 2 \
  --reader-dbuf-trid --reader-trid-in-flight 2 --reader-push-tiles 2 \
  --writer-flush-on-pressure --writer-end-barrier-mode 1 \
  --use-device-profiler --use-realtime-profiler

python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/export_op_to_op_profiler_csv.py \
  --input-file "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv" \
  --min-prog-id 3
```

### Multiple runs (recommended for HW review)

```bash
chmod +x tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/run_op_to_op_multi.sh
./tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/run_op_to_op_multi.sh 5
```

Writes `generated/profiler/op_to_op_runs/run_1/` … `run_5/` (raw logs + exported CSVs) and
`multi_run_device_op_to_op_gap.csv` / `multi_run_dispatch_done_to_go.csv` with min/max/mean/std
printed at the end.

To regather a single run after kernel changes, clear JIT cache then repeat quick start:

```bash
rm -rf ~/.cache/tt-metal-cache "${TT_METAL_CACHE:-}"
```

**Outputs** (`generated/profiler/.logs/`):

| File | Contents |
|------|----------|
| `profile_log_device.csv` | Raw device profiler log |
| `profile_log_device_op_to_op_summary_chip.csv` | All-cores gap stats (mean/median/min/max) |
| `profile_log_device_op_to_op_summary_per_core.csv` | Per-core averages across program transitions |
| `profile_log_device_op_to_op_gaps.csv` | Per core: pack finish prog *k* → unpack tile 0 prog *k+1* |
| `profile_log_device_op_to_op_timeline.csv` | Gaps + read/write before/after barrier + chip `go`/`done`/`gap` |
| `profile_log_device_op_to_op_complete.csv` | Same as timeline (full transition picture for HW) |

Buffering columns on timeline/complete (for DRAM read-before-compute questions):

| Column | Meaning |
|--------|---------|
| `input_cb_depth_tiles` / `input_cb_double_buffered` | CB config (default depth 4 = 2× push chunk) |
| `reader_push_tiles_per_chunk` / `reader_batch_push` | Reader `reserve_back(N)` pattern (default push 2, incremental) |
| `tiles_per_core` | Tiles per core per program (default 4) |
| `per_tile_dram_read_barrier` | Benchmark uses `noc_async_read_barrier` every tile |
| `tile0_dram_buffering_latency_us` | Prog *k+1* tile 0: `READ_AFTER` − `READ_BEFORE` (DRAM read + barrier) |
| `read_after_to_unpack_tile0_us` | Buffer ready → `TILE_IDX` tile 0 (CB wait + handoff to unpack) |
| `device_gap_excluding_tile0_dram_buffer_us` | Full device op-to-op gap minus tile-0 DRAM buffering |
| `tile0_dram_buffer_fraction_of_device_gap` | How much of the gap is tile-0 read+barrier |
| `profile_log_device_op_to_op_rt_programs.csv` | Per program: `go_cycles`, `done_cycles`, `program_duration_ns`, `gap_to_next_go_ns` |
| `profile_log_device_op_to_op_dispatch_gaps.csv` | Chip-level RT: previous `done` → next `go` (needs `--use-realtime-profiler`) |
| `profile_log_device_rt.csv` | Raw RT profiler records (go/done cycles per program) |
| `profile_log_device_op_to_op_tiles.csv` | TRISC_0: per-tile `unpack_start_cycles` |
| `profile_log_device_op_to_op_prog_ids.csv` | `PROG_ID` rows (TRISC_0) |

**Op-to-op gap:** `TRISC_0` `TILE_IDX` (`data`=0) of program *k+1* minus `TRISC_2`
`FINISH_LAST_PUSH` of program *k* (export script; `--min-prog-id 1` skips id 0).

Optional: `export TT_METAL_PROFILER_CPP_POST_PROCESS=1` for kernel-duration summary in the test log.

## CLI flags

| flag | default | meaning |
|---|---:|---|
| `--num-pages-per-core N` | 4 | interleaved DRAM pages per core (≥ 1) |
| `--num-active-cores N` | 0 (full grid) | cap active cores; 0 uses the full worker grid. Used for core-count sweeps |
| `--reader-push-tiles N` | 2 | reader `reserve_back` / push chunk size |
| `--input-cb-depth-tiles N` | 0 (auto 2×push) | input CB depth in tiles |
| `--output-cb-depth-tiles N` | 0 (default 2) | output CB depth in tiles |
| `--reader-batch-push` | off | reader mode 1: reserve N, read all, single barrier, `push_back(N)` |
| `--reader-dbuf-trid` | off | reader mode 2: per-trid double-buffer (TRID A/B alternating across two CB slots). Requires `input_cb_depth ≥ 2 × reader_trid_in_flight × page_size_tiles` |
| `--reader-trid-in-flight N` | 2 | reads in flight per TRID when reader mode 2 is on (bump for higher per-core BW; needs deeper input CB) |
| `--page-size-tiles N` | 1 | DRAM page size in tiles (requires `--reader-dbuf-trid` when > 1) |
| `--compute-nops N` | 0 | `TTI_NOP`s per tile; tune so math work is comfortably > 10 µs |
| `--num-programs N` | 8 | back-to-back enqueues per measurement |
| `--use-trace` | off | capture-once / replay-once instead of FD loop |
| `--trace-warmup-replays N` | 0 | untimed trace replays before the measured replay (reach steady-state trace path; use 2 for steady-state runs) |
| `--use-device-profiler` | off | dump `profile_log_device.csv` after `Finish` |
| `--use-realtime-profiler` | off | log per-chip dispatch done/go after timed section (see below) |
| `--no-warmup` | off | skip untimed warmup enqueue |
| `--trace-region-size N` | 1 MiB | trace buffer (bytes) |
| `--device-id N` | 0 | physical device |
| `--read-only` | off | reader pops, writer skips DRAM writes. `bytes_per_program` drops from `2× tiles× tile_size` to `1×` (reads only) |
| `--writer-flush-on-pressure` | off | writer only calls `noc_async_writes_flushed` when the output CB is about to back-pressure, instead of every tile |
| `--writer-end-barrier-mode N` | 0 | end-of-kernel synchronization: 0 = `noc_async_write_barrier` (waits for DRAM ACK), 1 = `noc_async_writes_flushed` (waits for L1 drain only), 2 = none |
| `--cross-program-dram-offset` | off | each program reads/writes a disjoint DRAM tile slice (host allocates `(num_programs+1)×` size). Use to avoid trace replay reusing warm DRAM rows across programs |
| `--buffer-tune` | off | sweep CB depths for DRAM BW, then run op-to-op at the smallest depth within tolerance of peak BW |
| `--buffer-tune-grid` | off | full input × output CB grid sweep (requires `--buffer-tune-output-depths`) |
| `--buffer-tune-bw-only` | off | exit after the BW sweep, skip the latency phase (useful for chart scripts that drive their own latency runs) |
| `--buffer-tune-input-depths LIST` | `2,4,6,8,12,16,24,32` | comma-separated depths for the input CB sweep |
| `--buffer-tune-output-depths LIST` | (none) | optional output CB depth sweep after input tune (required for `--buffer-tune-grid`) |
| `--buffer-tune-pages-per-core N` | 32 | tiles/core during the BW phase (NOP compute → DRAM bound) |
| `--buffer-tune-bw-tolerance PCT` | 2 | depths within PCT% of peak BW count as "at peak" |

## Buffer tune (`--buffer-tune`)

1. Reader **push-2** (double buffer), **compute-nops 0**, large `--buffer-tune-pages-per-core`.
2. For each `--buffer-tune-input-depths` value: one timed program, log **`dram_pipeline_gbps`** = `bytes_per_program / elapsed_us`.
   - `bytes_per_program = 2 × tiles × tile_size` (reads + writes summed); drops to `1×` when `--read-only` is set.
   - `elapsed_us` is wall-clock from right before `replay_mesh_trace` to right after `Finish(cq)`. With `num_programs=1` in the tune phase there are no inter-program gaps in the window — but trace replay invocation + dispatch + drain are still in there. Not kernel-only zone widths.
3. Pick **smallest** input depth within `--buffer-tune-bw-tolerance` percent of peak BW.
4. If `--buffer-tune-output-depths` is set, repeat for the output CB at the best input depth. With `--buffer-tune-grid`, sweep the full input × output product instead.
5. Re-run with `--num-programs` / `--num-pages-per-core` / `--compute-nops` from the normal CLI (latency phase) and optional profilers. Skip the latency phase with `--buffer-tune-bw-only`.

Parse results: `grep BUFFER_TUNE` in the test log. Helper scripts: `run_buffer_tune.sh`, `run_cb_grid_sweep.sh`, `pick_grid_min_cb.py`.

## Reader modes

| mode | flag | behavior |
|---|---|---|
| 0 (default) | (none) | push-1 incremental: `reserve_back(push_tiles)`, read + `noc_async_read_barrier` + `push_back(1)` per tile |
| 1 | `--reader-batch-push` | reserve `push_tiles`, read all, single barrier, `push_back(push_tiles)` |
| 2 | `--reader-dbuf-trid` | per-trid double-buffer: TRID A and TRID B alternate across two CB slots, with up to `--reader-trid-in-flight` reads per TRID in flight. Highest per-core BW; requires `input_cb_depth ≥ 2 × trid_in_flight × page_size_tiles` |

## Writer modes

| flag | behavior |
|---|---|
| (default) | `noc_async_writes_flushed` after every tile, `noc_async_write_barrier` at kernel end |
| `--writer-flush-on-pressure` | only flush when the output CB is about to back-pressure |
| `--writer-end-barrier-mode 0` | end-of-kernel `noc_async_write_barrier` (default; waits for DRAM ACK) |
| `--writer-end-barrier-mode 1` | end-of-kernel `noc_async_writes_flushed` (waits only for L1 drain; safe when the next consumer tolerates non-ACK'd data) |
| `--writer-end-barrier-mode 2` | no end-of-kernel synchronization (experiment only — next program may overlap with in-flight DRAM writes) |

## Steady-state runs

Trace replay needs a couple of untimed warmup replays before timings stabilize, and the first one
or two program transitions in any replay include trace-start artifacts. The recommended steady-state
recipe is:

- `--use-trace --trace-warmup-replays 2`
- `--num-programs 8` (so there are at least 6 clean transitions)
- pass `--min-prog-id 3` to `export_op_to_op_profiler_csv.py` to skip transitions where `from_prog_id < 3`

`run_op_to_op_validation.sh` and `run_op_to_op_chart_sweep.sh` apply this recipe automatically.

## Program-level gaps (real-time profiler)

`--use-realtime-profiler` records **done → go** at **chip dispatch** (not
per-core): `go_cycles` = program start, `done_cycles` = program end. Device CSV also has
per-core `BRISC_GO`/`BRISC_DONE` (writer) and `NCRISC_GO`/`NCRISC_DONE` (reader) at kernel
entry/exit. Export reports
**`dispatch_gap_cycles` = next `go` − previous `done`** (typically **~550–700 ns** on
Blackhole; run-to-run variation is normal). This is separate from the **~4–5 µs** device
op-to-op gap (pack finish → next unpack tile 0 on all cores in parallel).

The test logs each program’s go/done and writes `profile_log_device_rt.csv`.
`export_op_to_op_profiler_csv.py` emits `*_rt_programs.csv`, `*_dispatch_gaps.csv`, and
merges `chip_dispatch_*` columns into timeline/complete CSVs. Duplicate consecutive
`program_id` rows from trace replay are collapsed/skipped. See
[real_time_profiler/getting-started.md](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/real_time_profiler/getting-started.md).

## Kernel profiler notes

Do **not** use `DeviceZoneScopedMainN` in compute `kernel_main` (breaks `TRISC-KERNEL`
marker pairing). Wrap unpack/pack timestamps in `UNPACK(...)` / `PACK(...)` so each
marker appears only on the intended TRISC.

## Sweep scripts

All sweep scripts write per-config runs under `generated/profiler/op_to_op_runs/<batch>/` and a
`chart_data.csv` per batch. Re-run each phase individually with the `BUILD=0` env var if you've
already compiled the test binary.

| Script | Purpose |
|---|---|
| `run_op_to_op_multi.sh` | Run benchmark `N` times for one config (env-driven: `CONFIG_LABEL`, `EXTRA_ARGS`, `TILES_PER_CORE`, `INPUT_CB_DEPTH`, `READER_PUSH`, `MIN_PROG_ID`); aggregates per-run CSVs |
| `run_op_to_op_validation.sh` | Steady-state validation wrapper: 2 trace warmup replays + 8 programs + `--min-prog-id 3`; reports clean chip dispatch done/go |
| `run_op_to_op_chart_sweep.sh` | Full sweep across `{1, 2, 4, 10, 20, 40, 80, 110}` cores: buffer-tune for smallest CB at peak BW, then steady-state latency at that CB. Produces `chart_data.csv` ready to plot |
| `run_op_to_op_all_configs.sh` | Compare baseline push-1 vs push-2 (incremental & batch) vs push-2 BW-tuned, side-by-side |
| `run_buffer_tune.sh` | End-to-end BW-then-latency recipe (input depth sweep → smallest depth at peak → latency at that depth) |
| `run_cb_grid_sweep.sh` | Standalone grid CB sweep (input × output) |
| `run_extra_cores_chart.sh` | Extra core-count points for denser BW scaling |
| `run_batch2_sweep.sh` | Read-only BW sweep on a dense core grid (`--read-only`) |
| `run_batch3_compute_sweep.sh` | Vary `--compute-nops` to find the compute-bound knee |
| `run_batch4_chart_sweep.sh` | Grid CB tune + writer flush-on-pressure across all core counts |
| `run_batch5_chart_sweep.sh` | Full recipe: grid peak BW, then smallest input then output CB still within tolerance |
| `run_batch6_pareto.sh` | BW/latency Pareto sweep at 80c and 110c (full CB matrix) |
| `run_batch7_dram_offset.sh` | Same as batch 4 but with `--cross-program-dram-offset` so each program reads a disjoint DRAM slice |
| `run_batch7_pareto.sh` | Batch 6 Pareto sweep + `--cross-program-dram-offset` |
| `run_batch8_barrier.sh` | Sweep `--writer-end-barrier-mode {0,1,2}` at BW-optimal CBs (batch 4 picks) |
| `run_batch8b_barrier_b1cb.sh` | Sweep `--writer-end-barrier-mode {0,1,2}` at latency-optimal CBs (batch 1 picks). Use this for the barrier study, not batch 8 |

## Analysis scripts

| Script | Purpose |
|---|---|
| `export_op_to_op_profiler_csv.py` | Main exporter: turns `profile_log_device.csv` (+ `profile_log_device_rt.csv`) into per-transition gap tables, summaries, timeline, and dispatch gaps. Drop trace-start transitions with `--min-prog-id N`. Aggregate across runs with `--aggregate-runs-dir DIR`. |
| `export_op_to_op_gaps_csvlite.py` | Pandas-free fallback for the same per-transition gap CSV. Used by `run_op_to_op_multi.sh` if the pandas exporter fails (e.g. environments without pandas) |
| `pick_grid_min_cb.py` | Parse `BUFFER_TUNE` log lines and pick the smallest input CB then smallest output CB still within tolerance of peak BW |
| `decompose_op_to_op_gap.py` | Split the op-to-op gap into chronological components: writer tail (`pack_finish → brisc_done`), dispatch (`brisc_done → brisc_go`), first DRAM (`brisc_go → read_after_barrier`), DRAM-to-compute (`read_after_barrier → unpack_tile_0`). Parametrized via `BATCH_PREFIX` / `OUT_DIR_NAME` / `MIN_PROG_ID` env vars |
| `build_share_tables.py` | Consolidate all batch result CSVs into a single sheet-ready CSV with per-batch headers |
| `rebuild_batch{2,3,4,5,7}_*.py` | Rebuild comparison tables for a given batch from saved per-run CSVs (no re-run needed) |

## Files

```
test_op_to_op_latency.cpp                 host test binary
kernels/
├── reader_interleaved.cpp                NCRISC reader (modes 0/1/2 + cross-program offset)
├── writer_interleaved.cpp                BRISC writer (flush mode + end-barrier mode + cross-program offset)
└── compute_copy_with_nops.cpp            TRISC copy + tunable NOPs
export_op_to_op_profiler_csv.py           main exporter (pandas)
export_op_to_op_gaps_csvlite.py           pandas-free fallback exporter
pick_grid_min_cb.py                       CB picker from BUFFER_TUNE log lines
decompose_op_to_op_gap.py                 op-to-op gap component breakdown
build_share_tables.py                     consolidate batch CSVs for sharing
run_op_to_op_multi.sh                     run-N-times wrapper
run_op_to_op_validation.sh                steady-state validation wrapper
run_op_to_op_chart_sweep.sh               full BW × latency × cores sweep
run_op_to_op_all_configs.sh               reader-mode comparison sweep
run_buffer_tune.sh                        end-to-end buffer tune recipe
run_cb_grid_sweep.sh                      standalone CB grid sweep
run_extra_cores_chart.sh                  extra core-count BW points
run_batch{2,3,4,5,6,7,7_pareto,8,8b}*.sh  batch-specific sweeps (see Sweep scripts table)
rebuild_batch{2,3,4,5,7}_*.py             rebuild comparison tables from saved runs
```
