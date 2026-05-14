# `op_to_op_latency` — back-to-back program latency benchmark

Measures the latency between when one program's math finishes and when the
next program's math starts on the chip. This gap is firmware tear-down +
dispatch + barrier overhead, and is what the HW team needs to characterize
in order to evaluate optimisations like virtualised init registers and
relaxed end-of-op barriers.

## What the benchmark does

On every Tensix core of the chip, run a textbook three-kernel pipeline:

```
reader (NOC1 / RISCV1) ──CB_in──▶  compute (3 TRISCs) ──CB_out──▶  writer (NOC0 / RISCV0)
   noc_async_read_tile              copy_tile                       noc_async_write_tile
   from interleaved DRAM            + N × TTI_NOP                   + noc_async_write_barrier
                                    profiler (device CSV):
                                      TRISC-KERNEL  (firmware trisck.cc — wraps user kernel)
                                      DeviceZoneScopedN("MATH") + TILE_IDX (per tile, user kernel)
```

The inner `MATH` zone opens **after** `cb_wait_front` and `cb_reserve_back`, so
its start cycle marks the moment data is available to the tensix and there is
output-CB space to write into — i.e. the earliest point at which math can
actually begin.

Pages are interleaved across all DRAM banks; work is split across all
worker cores via `split_work_to_cores`. The same `MeshWorkload` is enqueued
back-to-back `--num-programs N` times, in either Fast-Dispatch mode (host
issues N enqueues) or Trace mode (capture once, replay once with all N
enqueues recorded).

## Build

```bash
# Default Metal build has Tracy ON (`ENABLE_TRACY` defaults ON; do not pass
# `build_metal.sh --disable-profiler`). Device/kernel zones are compiled into
# kernels at JIT time only when `TT_METAL_DEVICE_PROFILER=1` is set **before**
# the process starts (see tt_metal/llrt/rtoptions.cpp). Re-run after exporting
# that env so kernels rebuild with `-DPROFILE_KERNEL=...`; clear the JIT cache
# if an old non-profiler kernel keeps getting picked up:
#   rm -rf ~/.cache/tt-metal-cache "$TT_METAL_CACHE"

./build_metal.sh --build-tests
cmake --build build_Release --target test_op_to_op_latency -j
```

## Run

```bash
TT_METAL_HOME=$(pwd) \
TT_METAL_DEVICE_PROFILER=1 \
  ./build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency \
    --use-trace \
    --num-programs 8 \
    --compute-nops 2000 \
    --use-device-profiler
```

### CLI flags

| flag | default | meaning |
|---|---:|---|
| `--num-pages-per-core N` | 2 | interleaved DRAM pages per core (≥ 1) |
| `--compute-nops N` | 0 | `TTI_NOP`s per tile in compute kernel; tune so the `MATH` zone is comfortably > 10us (use Tracy to confirm) |
| `--num-programs N` | 8 | back-to-back enqueues per measurement |
| `--use-trace` | off | capture-once / replay-once instead of FD-mode loop |
| `--use-device-profiler` | off | dump `profile_log_device.csv` after Finish (Tracy-enabled build — default unless `build_metal.sh --disable-profiler`; export `TT_METAL_DEVICE_PROFILER=1` **before** starting the process) |
| `--use-realtime-profiler` | off | register the [real-time profiler](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/real_time_profiler/getting-started.md) callback for the **timed** enqueue burst; after `Finish`, logs per-chip op-to-op gaps (`next start − previous end`, ns). Inactive on some dispatch setups — check logs if no gaps appear. |
| `--no-warmup` | off | skip the untimed warmup enqueue |
| `--trace-region-size N` | 1 MiB | trace buffer size (bytes); bump if `--num-programs` is very large |
| `--device-id N` | 0 | which physical device |

## Program-level op-to-op (real-time profiler)

Pass **`--use-realtime-profiler`** to register `RegisterProgramRealtimeProfilerCallback`
for the **timed** portion only (after warmup). In FD mode that is the `N` back-to-back
`EnqueueMeshWorkload` calls plus `Finish`. In trace mode it is **replay + `Finish`**
only (capture is outside the callback window so RT records are not polluted by the
capture pass). When the device is supported, the test logs per-chip statistics for
**gap = next program `start_timestamp` − previous program `end_timestamp`** (nanoseconds),
matching Mo’s op2op definition. See
[getting-started.md](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/real_time_profiler/getting-started.md).
This is independent of `--use-device-profiler` (kernel CSV zones).

The benchmark sets **`Program::set_runtime_id(1)`** before enqueue: the RT receiver
ignores pages whose id is **0** (reserved for non-program traffic), and new programs
default to runtime id 0 — without a non-zero id you would see **“got 0 record(s)”**
even when the profiler is active.

**Example observed output** (Blackhole, chip 0, FD, `--num-programs 8`,
`--compute-nops 1000`, `--use-realtime-profiler`; numbers vary by chip,
firmware, and load):

```text
Real-time profiler chip 0: 8 op-to-op gap(s) — min 1265.94 ns, max 78903.69 ns, mean 12339.28 ns
FD back-to-back: 8 programs in 84 us (avg 10.50 us/program)
```

So in that run, device-side **gap = next program start − previous program end**
averaged **~12.3 µs** per chip-0 pair, with one **~79 µs** outlier and a **~1.3 µs**
shortest gap. Host wall time for the same burst was **84 µs** total (**~10.5 µs**
per enqueue including all work), which is a different quantity from the RT gap.

## Extracting op-to-op latency from the CSV

After a `--use-device-profiler` run:

```bash
# raw CSV is here:
ls $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv

# pretty-printed analysis:
$TT_METAL_HOME/tools/tracy/process_device_log.py
```

There are two useful zone families in the CSV:

* **`TRISC-KERNEL`** — emitted by firmware around `run_kernel()` on each TRISC
  (`tt_metal/hw/firmware/src/tt-1xx/trisck.cc`). One begin/end pair per TRISC
  per program-run for the whole user kernel body. Safe for a coarse
  “kernel ran” envelope. Do **not** add `DeviceZoneScopedMainN` in user TRISC
  `kernel_main`: it nests inside `TRISC-KERNEL` and `MainN`’s destructor calls
  `finish_profiler()` while the parent zone is still open, which breaks host
  marker pairing (`TT_FATAL: Start and end marker IDs do not match`).

* **`MATH`** — one per tile per program-run on each TRISC, with a **`TILE_IDX`**
  timestamped-data row for loop index `i`. The inner `MATH` zone opens **after**
  `cb_wait_front` and `cb_reserve_back`, so its start marks when data is in L1
  and output CB space exists.

**Program-level op-to-op between host enqueues** (Mo’s metric) is best taken
from **`--use-realtime-profiler`** (`ProgramRealtimeRecord`), not from nesting
another `MainN` zone in this compute kernel.

For each (core, TRISC) row, sort by cycle and pair `_begin` / `_end`.
Convert cycles to microseconds with the device clock frequency that
`get_tt_npu_clock` reports (the test logs `Clock` for convenience).

Use the `MATH` TRISC row of the CSV for "math TRISC" semantics, or the
`PACK` TRISC row if you want "data committed to L1 ready for the writer"
semantics.

## Files

```
test_op_to_op_latency.cpp          host program
kernels/
  reader_interleaved.cpp           NOC1 reader, TensorAccessor + noc_async_read_tile
  writer_interleaved.cpp           NOC0 writer, noc_async_write_tile + barrier
  compute_copy_with_nops.cpp       copy_tile + N×TTI_NOP; per-tile MATH + TILE_IDX
                                   (no DeviceZoneScopedMainN — see README)
```
