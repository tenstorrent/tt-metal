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
                                    (DeviceZoneScopedMainN("MATH"))
```

Pages are interleaved across all DRAM banks; work is split across all
worker cores via `split_work_to_cores`. The same `MeshWorkload` is enqueued
back-to-back `--num-programs N` times, in either Fast-Dispatch mode (host
issues N enqueues) or Trace mode (capture once, replay once with all N
enqueues recorded).

## Build

```bash
# functional build (markers will be compiled out as no-ops):
cmake --build build --target test_op_to_op_latency -j

# profiler-enabled build (required to actually capture op-to-op timing):
./build_metal.sh --enable-profiler --build-tests
cmake --build build --target test_op_to_op_latency -j
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
| `--use-device-profiler` | off | dump `profile_log_device.csv` after Finish (requires profiler build + `TT_METAL_DEVICE_PROFILER=1`) |
| `--no-warmup` | off | skip the untimed warmup enqueue |
| `--trace-region-size N` | 1 MiB | trace buffer size (bytes); bump if `--num-programs` is very large |
| `--device-id N` | 0 | which physical device |

## Extracting op-to-op latency from the CSV

After a `--use-device-profiler` run:

```bash
# raw CSV is here:
ls $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv

# pretty-printed analysis:
$TT_METAL_HOME/tools/tracy/process_device_log.py
```

Filter the rows for zone name `MATH`. For each (core, TRISC) pair, the
sequence of `MATH_begin` / `MATH_end` cycles gives one pair per program-run.
The op-to-op latency for program `n+1` on a given core is:

```
op_to_op_latency_cycles = MATH_begin[n+1] - MATH_end[n]
```

Convert cycles to microseconds with the device clock frequency that
`get_tt_npu_clock` reports (the test logs `Clock` for convenience).

Use the `MATH` row of the CSV for "math TRISC" semantics, or the `PACK`
row if you want "data committed to L1 ready for the writer" semantics.

## Files

```
test_op_to_op_latency.cpp          host program
kernels/
  reader_interleaved.cpp           NOC1 reader, TensorAccessor + noc_async_read_tile
  writer_interleaved.cpp           NOC0 writer, noc_async_write_tile + barrier
  compute_copy_with_nops.cpp       copy_tile + N×TTI_NOP, wrapped in DeviceZoneScopedMainN("MATH")
```
