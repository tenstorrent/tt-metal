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

All launches also emit `PROG_ID` (0 = warmup / pre-compile, 1..N = timed).

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
  --use-device-profiler

python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/export_op_to_op_profiler_csv.py \
  --input-file "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
```

**Outputs** (`generated/profiler/.logs/`):

| File | Contents |
|------|----------|
| `profile_log_device.csv` | Raw device profiler log |
| `profile_log_device_op_to_op_summary_chip.csv` | All-cores gap stats (mean/median/min/max) |
| `profile_log_device_op_to_op_summary_per_core.csv` | Per-core averages across program transitions |
| `profile_log_device_op_to_op_gaps.csv` | Per core: pack finish prog *k* → unpack tile 0 prog *k+1* |
| `profile_log_device_op_to_op_tiles.csv` | TRISC_0: per-tile `unpack_start_cycles` |
| `profile_log_device_op_to_op_prog_ids.csv` | `PROG_ID` rows (TRISC_0) |

**Op-to-op gap:** `TRISC_0` `TILE_IDX` (`data`=0) of program *k+1* minus `TRISC_2`
`FINISH_LAST_PUSH` of program *k* (export script; `--min-prog-id 1` skips id 0).

Optional: `export TT_METAL_PROFILER_CPP_POST_PROCESS=1` for kernel-duration summary in the test log.

## CLI flags

| flag | default | meaning |
|---|---:|---|
| `--num-pages-per-core N` | 2 | interleaved DRAM pages per core (≥ 1) |
| `--compute-nops N` | 0 | `TTI_NOP`s per tile; tune so math work is comfortably > 10 µs |
| `--num-programs N` | 8 | back-to-back enqueues per measurement |
| `--use-trace` | off | capture-once / replay-once instead of FD loop |
| `--use-device-profiler` | off | dump `profile_log_device.csv` after `Finish` |
| `--use-realtime-profiler` | off | log per-chip program gaps after timed section (see below) |
| `--no-warmup` | off | skip untimed warmup enqueue |
| `--trace-region-size N` | 1 MiB | trace buffer (bytes) |
| `--device-id N` | 0 | physical device |

## Program-level gaps (real-time profiler)

`--use-realtime-profiler` logs chip-level **next program start − previous program end**
(nanoseconds) for the timed section only. Separate from the device CSV; not written to
a file by this test. See
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
