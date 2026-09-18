# Hardware Performance Counters

## Quick Links
- Hardware reference (counter inventory, registers, limitations, derived metric catalogue): [tt_metal/tt-llk/docs/performance_counters/hardware_reference.md](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md)
- User guide: [docs/source/ttnn/ttnn/profiling_ttnn_operations.rst](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst)
- Counter names, per-arch select tables, register map (shared with the tt-llk harness): [tt_metal/tt-llk/tools/include/perf_counters/](../../tt_metal/tt-llk/tools/include/perf_counters/)
- Profiler firmware side (record format, group bits, readout): [tt_metal/tools/profiler/perf_counters.hpp](../../tt_metal/tools/profiler/perf_counters.hpp)
- Header parsers and metric formulas (shared with the tt-llk harness): [tt_metal/tt-llk/tools/python/tt_llk_perf/](../../tt_metal/tt-llk/tools/python/tt_llk_perf/)
- Tracy adapter: [tools/tracy/perf_counter_analysis.py](../../tools/tracy/perf_counter_analysis.py)
- CSV writer: [tools/tracy/process_ops_logs.py](../../tools/tracy/process_ops_logs.py)

## Overview

Each Tensix core contains hardware performance counters organized into five banks (FPU, TDMA_UNPACK, TDMA_PACK, INSTRN_THREAD, L1). These counters measure cycle-level events: how many cycles the FPU was active, how many cycles a thread was stalled, how many cycles each L1 memory port had traffic, and so on. Quasar has the first four banks in every NEO and no L1 bank; one l1_client event CSR per NEO stands in for it.

The hardware itself (banks and selects per architecture, the debug registers, what is not reachable from software) and the catalogue of derived metrics are described once, in tt-llk: [hardware_reference.md](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md). This report covers the tt-metal profiler side: how a capture is requested, how the firmware reads the counters out and how the results reach the ops report.

## How It Works

1. **Kernel starts**: TRISC1 calls `start_perf_counter()` which writes the start bit to all enabled counter banks. All counters begin accumulating from zero.

2. **Kernel runs**: While the kernel executes, each counter increments every cycle its input signal is high. All counters within a bank run simultaneously; there is no multiplexing during measurement.

3. **Kernel ends**: TRISC1 calls `stop_perf_counter()` which freezes all counters. The counter values remain latched in the debug registers.

4. **BRISC reads counters**: After all TRISCs complete (`wait_ncrisc_trisc()`), BRISC calls `read_perf_counters()`, which walks the select table of every enabled group (`llk::perf::table_for`), programs each select into the bank's mode register and reads the count (`llk::perf::read_table`). Each counter value is packed into a 16-byte profiler marker (id 9090) tagged with the `PerfCounterType` ordinal and written to BRISC's profiler buffer, which is flushed to DRAM whenever it fills. TRISCs cannot do this because they have no NOC access.

5. **Host reads**: After the kernel completes, the host reads the profiler data from DRAM and decodes each marker into a counter ordinal, value, and reference count.

6. **Python processes**: `tools/tracy/perf_counter_analysis.py` resolves the ordinals to names with `tt_llk_perf.headers.counter_type_names()` (parsed from `types.h`, see below) and computes the derived metrics per operation and core with `tt_llk_perf.metrics` (the same package the tt-llk test harness uses, so both report the same numbers from the same counters). Results are written to CSV and printed to console.

On Quasar the DM0 core does both halves: it calls `start_perf_counter()` before it releases the TRISCs and `stop_perf_counter()` plus `read_perf_counters()` after they finish, looping over the four NEOs through the NoC window (`neo_window(n)`, see the register reference). DM0 has no profiler buffer of its own, so it writes each NEO's records into that NEO's four TRISC profiler buffers; NEOs with no enabled TRISC are skipped. Every record carries the NEO index in a 4-bit `neo` field next to the 16-bit counter type (tt-1xx records are unchanged: type below 256, `neo` 0), and the host reports metrics per NEO with `risc_type` set to `QUASAR_NEO<n>`.

### Where the definitions live

Everything that describes the hardware is in tt-llk, under `tt_metal/tt-llk/tools/include/perf_counters/` (namespace `llk::perf`): `types.h` (the `PerfCounterType` enum, whose ordinal is the wire format, and the `Bank` enum), `blackhole.h`, `wormhole.h` and `quasar.h` (per-bank `{name, select}` tables), `inventory.h` (arch selection and `table_for`), `registers.h` (debug register addresses and bit constants; on Quasar the per-bank offsets, the two address windows and the l1_client CSR pair) and `hw.h` (the register primitives). The Python package `tt_metal/tt-llk/tools/python/tt_llk_perf/` parses names and tables from those headers (`headers.py`) and holds the metric engine (`metrics.py`). `tt_metal/tools/profiler/perf_counters.hpp` adds only the profiler policy: the record format, the `TT_METAL_PROFILE_PERF_COUNTERS` group bits and the emission into the profiler buffer. Adding a counter means appending to the enum and adding a table entry; no host code changes.

### How to Run

```bash
# Capture every counter group; needs several passes, so opt in to the workload replay
python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all \
    -m "pytest your_test.py -x -v"
```

With `--perf-counter-multipass` a request is split into passes (one L1 bank per pass, the non L1 groups ride along) and `all` expands to the architecture's full group set.

Available counter groups for `--profiler-capture-perf-counters`: `fpu`, `pack`, `unpack`, `l1_0`, `l1_1`, `instrn`, `all`. Blackhole also supports `l1_2`, `l1_3`, `l1_4`, `l1_5`; `all` expands to the running architecture's full set. Quasar has no L1 groups: `all` there is `fpu,pack,unpack,instrn` and a request naming an `l1_*` group is rejected. The multipass scheduler still splits those four groups 3 + 1; the environment variable path below takes all four in one run.

One limit forces a request like `all` into several capture passes: the L1 banks share one count-time mux, so at most one L1 bank can count per run. The readout is table driven, so the four other groups fit next to any L1 bank (BRISC firmware with the five group mask: 8664 of 8704 bytes on Blackhole, 7584 of 7712 on Wormhole). `python -m tracy` schedules the passes automatically. A request that fits one pass runs once, exactly as before; a request that does not stops with the printed pass plan unless `--perf-counter-multipass` is given, in which case the workload is replayed once per pass and the per-pass device logs are merged. See the [user guide](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst) for details.

### Environment Variable

`TT_METAL_PROFILE_PERF_COUNTERS` is a bitfield that selects which counter groups are captured. Multiple groups are combined with OR.

| Bit | Value | Group |
|-----|-------|-------|
| `1 << 0` | 1 | FPU |
| `1 << 1` | 2 | PACK |
| `1 << 2` | 4 | UNPACK |
| `1 << 3` | 8 | L1 bank 0 (ring0 NOC, L1 arbitration) |
| `1 << 4` | 16 | L1 bank 1 (ring1 NOC, TDMA extended) |
| `1 << 5` | 32 | INSTRN (instruction thread) |
| `1 << 6` | 64 | L1 bank 2 (BH only: extended unpackers 4-7, ring0 NOC ports 2-3) |
| `1 << 7` | 128 | L1 bank 3 (BH only: ring1 NOC ports 2-3, extended packers 2-5) |
| `1 << 8` | 256 | L1 bank 4 (BH only: extended packers 6-7, tag search, extended unpackers 8-12) |
| `1 << 9` | 512 | L1 bank 5 (BH only: extended unpackers 13-14; the mux wires only two slots here) |

The env-var path selects one pass directly: any set of groups with at most one L1 bank. Example single-pass capture:

```bash
export TT_METAL_PROFILE_PERF_COUNTERS=11   # FPU | PACK | L1 bank 0
```

**L1 bank mutual exclusion:** all L1 banks share the same hardware mux (selected via `MUX_CTRL`), so only one L1 bank may be enabled per run; the env-var path throws if more than one L1 bit is set. For anything that needs several passes, use `python -m tracy --perf-counter-multipass` (above), which schedules the passes and merges the results.

**Quasar mask:** the L1 bits do not exist there. `tt_metal/jit_build/build.cpp` remaps the front end's `all` mask 47 to 39 (`FPU | PACK | UNPACK | INSTRN`) and fails with `TT_FATAL` on any other mask with a bit outside those four; `perf_counters.hpp` has a matching `#error`. The four groups fit the DM0 firmware in one run.

**`TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL`** (Quasar only, default -1 = off) routes the one l1_client event counter of every NEO for the run. The value is `subport * 8 + event`: 37 subports (0-3 TRISC, 4 THCON, 5-24 unpacker read interfaces, 25-36 packer write interfaces) by 8 events, 296 selections of which 256 are valid. Event 0 reads 0 in the RTL and the THCON subport's events 1 to 3 duplicate the TRISC port's SBank 0, so `build.cpp` rejects those (same rule as `llk::perf::l1_client_selection_is_valid` in `quasar.h`, which the firmware `static_assert`s). The CSR has no reference counter; its record carries the wall-clock span between arm and freeze, and the host names the resulting column after the selection (see [L1 client events (Quasar)](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md#l1-client-events-quasar)).

```bash
export TT_METAL_PROFILE_PERF_COUNTERS=39        # FPU | PACK | UNPACK | INSTRN, the Quasar "all"
export TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL=41 # subport 5 (unpacker 0, interface 0), event 1: L1_CLIENT_UNPACK0_IF0_SBANK0_SBANK_POP
```

### Architecture Summary

Moved to the [hardware reference](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md#architecture-summary).

## Derived Metrics Reference

Moved to the [hardware reference](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md#derived-metrics-reference). The formulas live in [tt_metal/tt-llk/tools/python/tt_llk_perf/metrics.py](../../tt_metal/tt-llk/tools/python/tt_llk_perf/metrics.py); the Tracy tool computes them per operation and core and aggregates to Min/Median/Max/Avg across cores.

## Hardware Register Reference

Moved to the [hardware reference](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md#hardware-register-reference).

## Hardware Limitations

Moved to the [hardware reference](../../tt_metal/tt-llk/docs/performance_counters/hardware_reference.md#hardware-limitations).
