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
| `--reader-push-tiles N` | 2 | reader `reserve_back` / push chunk size |
| `--input-cb-depth-tiles N` | 0 (auto 2×push) | input CB depth in tiles (output CB stays 2) |
| `--reader-batch-push` | off | reserve N, read all, `push_back(N)`; default is read+`push_back(1)` per tile in chunk |
| `--compute-nops N` | 0 | `TTI_NOP`s per tile; tune so math work is comfortably > 10 µs |
| `--num-programs N` | 8 | back-to-back enqueues per measurement |
| `--use-trace` | off | capture-once / replay-once instead of FD loop |
| `--use-device-profiler` | off | dump `profile_log_device.csv` after `Finish` |
| `--use-realtime-profiler` | off | log per-chip program gaps after timed section (see below) |
| `--no-warmup` | off | skip untimed warmup enqueue |
| `--trace-region-size N` | 1 MiB | trace buffer (bytes) |
| `--device-id N` | 0 | physical device |
| `--output-cb-depth-tiles N` | 0 (default 2) | output CB depth in tiles |
| `--buffer-tune` | off | sweep input CB depth for DRAM BW, then op-to-op at smallest depth at peak BW |
| `--buffer-tune-input-depths LIST` | `2,4,6,8,12,16,24,32` | comma-separated depths for BW sweep |
| `--buffer-tune-output-depths LIST` | (none) | optional output CB depth sweep after input tune |
| `--buffer-tune-pages-per-core N` | 32 | tiles/core during BW phase (NOP compute → DRAM bound) |
| `--buffer-tune-bw-tolerance PCT` | 2 | depths within PCT% of peak BW count as “at peak” |

## Buffer tune (`--buffer-tune`)

1. Reader **push-2** (double buffer), **compute-nops 0**, large `--buffer-tune-pages-per-core`.
2. For each `--buffer-tune-input-depths` value: one timed program, log **`dram_pipeline_gbps`** (read+write bytes / wall time).
3. Pick **smallest** input depth within tolerance of peak BW.
4. Re-run **`--num-programs`** with `--num-pages-per-core` / `--compute-nops` from the normal CLI (latency phase) and optional profilers.

Parse results: `grep BUFFER_TUNE` in the test log. Helper script: `run_buffer_tune.sh`.

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

## Files

```
test_op_to_op_latency.cpp
export_op_to_op_profiler_csv.py
kernels/reader_interleaved.cpp
kernels/writer_interleaved.cpp
kernels/compute_copy_with_nops.cpp
```
