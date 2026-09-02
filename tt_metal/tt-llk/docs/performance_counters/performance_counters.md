# LLK Performance Counters Guide

## Quick Links
- Device-side counter HW driver: [tests/helpers/include/counters.h](../../tests/helpers/include/counters.h)
- Test-helper mock functions: [tests/helpers/include/perf.h](../../tests/helpers/include/perf.h)
- Profiler zone macros: [tests/helpers/include/profiler.h](../../tests/helpers/include/profiler.h)
- Host-side counter readback: [tests/python_tests/helpers/counters.py](../../tests/python_tests/helpers/counters.py)
- Host-side derived metrics: [tests/python_tests/helpers/metrics.py](../../tests/python_tests/helpers/metrics.py)
- Test driver: [tests/python_tests/helpers/perf/core.py](../../tests/python_tests/helpers/perf/core.py)
- Test sources: [tests/sources/](../../tests/sources/) (files ending in `_perf.cpp`)
- Pytest CLI registration: [tests/python_tests/conftest.py](../../tests/python_tests/conftest.py)
- Upstream tech report (metal-level): [tech_reports/PerfCounters/perf-counters.md](../../../../tech_reports/PerfCounters/perf-counters.md)

## Overview

This guide documents the LLK test-suite interface for collecting Tensix hardware performance counters. The LLK suite runs bare-metal kernels in `tests/sources/*_perf.cpp` directly on the TRISC cores. There is no firmware or NoC stack involved, so the counter-collection path is implemented entirely inside the test kernels: a C++ macro arms and freezes the hardware counters around a measured scope, writes the snapshot to a fixed L1 buffer, and the host process reads that buffer back from outside the kernel.

Tensix cores contain five hardware performance counter banks. Every bank exposes two values per measurement: total elapsed cycles (`OUT_L`) and an event count for the selected `counter_sel` (`OUT_H`). The Python driver enumerates every per-architecture counter slot, configures the HW, runs the kernel, then iterates each slot to pull both values out into a pandas DataFrame and optionally a CSV. Derived metrics (utilisation %, stall %, backpressure %, composite ratios) are computed in Python on top of the raw counts.

| Bank | Description |
|------|-------------|
| INSTRN_THREAD | Per-thread instruction issue counts, availability, and stall reasons |
| FPU | FPU and SFPU active cycles |
| TDMA_UNPACK | Unpacker busy signals, math pipeline status, srcA/B write port and overwrite signals |
| TDMA_PACK | Packer busy, dest-read availability, math availability |
| L1 | NoC ring transactions and L1 port arbitration (mux-selected) |

## How It Works

### Two builds, one test source

Every test source under `tests/sources/*_perf.cpp` is compiled twice from the same C++ file. The build is selected by two preprocessor flags:

| Build | `LLK_PROFILER` | `PERF_COUNTERS_COMPILED` | Active half of `START_PERF_MEASURE` | What it measures |
|-------|----------------|--------------------------|--------------------------------------|------------------|
| NC (no counters) | defined | undefined | `ZONE_SCOPED` | Per-zone wall-clock cycles (`RISCV_DEBUG_REG_WALL_CLOCK_L`) |
| WC (with counters) | defined | defined | `MEASURE_PERF_COUNTERS` | Per-zone HW counter snapshot |

`START_PERF_MEASURE(name)` expands to `MEASURE_PERF_COUNTERS(name)` + `ZONE_SCOPED(name)`. The two halves are mutually exclusive — only one of them is non-empty in any given build, so wall-clock and counter measurements are never taken simultaneously and cannot perturb each other. The single name keeps NC wall-clock data and WC counter data joinable by zone name in the host driver; the driver runs whichever build is needed and merges the resulting DataFrames on that name.

Source-side, this is the pattern:

```cpp
void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        // ... unpack hw_configure, math_init, pack_init ...
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        for (uint32_t tile = 0; tile < TILE_CNT; ++tile)
        {
            // ... per-tile work ...
        }
    }
}
```

Each zone is registered once at its first encounter (the counter half is RAII-scoped and assigns a stable zone id by hashing the name), so placing `START_PERF_MEASURE` **outside** the loop is preferred — counter start is not a no-op and would dominate per-iteration cost if done on every tile.

### `PerfRunType` and the split arm/freeze model

Each LLK perf test is associated with a `PerfRunType` (declared in `perf.h`):

| Run type | Purpose | Arm thread | Freeze thread |
|----------|---------|-----------|---------------|
| `L1_TO_L1` | End-to-end pipeline cycles, unpack → math → pack | UNPACK | PACK |
| `L1_CONGESTION` | Pipeline cycles under L1 traffic contention, unpack → pack | UNPACK | PACK |
| `UNPACK_ISOLATE` | Unpack-only kernels (no math/pack) | UNPACK | UNPACK |
| `MATH_ISOLATE` | Math/SFPU-only kernels (no unpack/pack) | MATH | MATH |
| `PACK_ISOLATE` | Pack-only kernels (no unpack/math) | PACK | PACK |

The arm thread runs first in the natural pipeline, the freeze thread runs last. For end-to-end measurements (`L1_TO_L1`, `L1_CONGESTION`) the window opens when unpack starts producing and closes when pack stops consuming. For single-thread isolate modes the same thread arms and freezes — the other two threads are idle for the run type and only participate in the entry/exit barrier.

The arm/freeze split is determined at compile time by `is_arm_thread<run_type>()` and `is_freeze_thread<run_type>()` in `counters.h`.

### The `MEASURE_PERF_COUNTERS` macro

Expands to a `perf_counter_scoped<PERF_RUN_TYPE>` RAII object. Its constructor and destructor execute the following sequence (only on the WC build):

1. **Constructor (zone entry).** The **arm thread** writes the rising-edge start bit to `PERF_CNT_ALL` (FPU + INSTRN), `PERF_CNT_TDMA_UNPACK2`, `PERF_CNT_L1_2`, and `PERF_CNT_TDMA_PACK2`, clearing all banks and starting the count. It then posts the entry semaphore (`pc_buf` slot `FPU_SFPU`) twice. The two non-arm threads spinwait on that semaphore, then `semaphore_get` it. The barrier guarantees no thread is inside the measured scope before the arm thread has armed the HW.

2. **Body.** All three threads run the work inside the scope. Counters tick continuously on the shared backend.

3. **Destructor (zone exit).** The **freeze thread** writes the rising-edge stop bit to the same four registers, then walks the shared 200-word config buffer at `0x169000` and reads every valid slot. For each slot it programs the bank's mode register with the `counter_sel`, reads `OUT_H` (the event count), and stores the value in the per-zone data area. The bank's elapsed-cycles value (`OUT_L`) is sampled once per bank from the first slot. After all slots are read it posts the exit semaphore (`pc_buf` slot `UNPACK_TO_DEST`) twice. The two non-freeze threads spinwait then `semaphore_get` the exit semaphore.

Each zone gets its own data block in L1 (see [L1 Layout](#l1-layout-and-zone-buffers)) so multiple measurement scopes in the same kernel produce independent snapshots. The kernel may contain up to `PERF_COUNTERS_MAX_ZONES = 8` distinct zone names; identical names share a zone.

The `pc_buf` semaphores are the cheapest synchronisation primitive available on Tensix — they're consumed by the backend without involving the FPU or unpacker pipelines, so the barrier itself contributes negligible cycles to the measured window.

### Configure-once from BRISC

Before any TRISC kernel runs, BRISC executes `configure_and_arm_from_brisc()` once (called from `brisc.cpp` when the WC build flag is set). This:

- Writes the per-architecture `BUILTIN_COUNTER_CONFIG` (130 slots on WH, 169 on BH) into the shared L1 config buffer at `0x169000`.
- Clears every per-zone data area and sync word.
- Programs each bank's reference-period and mode registers, sets `PERF_CNT_MUX_CTRL` for L1, and does an initial global arm (later overridden by the first `MEASURE_PERF_COUNTERS` zone).

After BRISC releases the TRISCs, the shared config is read-only for the rest of the run.

### Reading results from host

After the kernel completes:

1. The host process reads the per-zone data area back from device L1.
2. `read_counters()` decodes each 32-bit config word (bit 31 valid, bits 7:0 bank, bits 16:8 `counter_sel`, bits 19:17 `l1_mux`), looks up the human-readable counter name from the per-architecture inventory, and pairs every event count with that zone's bank cycle count.
3. The result is a long-format DataFrame: one row per `(zone, bank, counter_id, l1_mux)` tuple with columns `cycles`, `count`, and (optionally) derived metrics from `compute_metrics()`.

Because both wall-clock cycles (NC build, `ZONE_SCOPED` start/end timestamps from `RISCV_DEBUG_REG_WALL_CLOCK_L`) and HW counter cycles (WC build, `OUT_L`) are tagged with the same zone name, the test driver merges them by `(test_variant, zone)`.

## How to Run

The LLK test suite uses a two-phase pytest flow: a compile-producer phase that builds every variant in parallel and a compile-consumer phase that runs them on hardware.

```bash
source setup_testing_env.sh   # required: sets LLK_HOME, PATH, virtualenv
cd $LLK_HOME/tests
export CHIP_ARCH=blackhole   # or wormhole / quasar

# Phase 1 — build all variants (no HW access)
pytest --compile-producer --enable-perf-counters -n 8 -x ./python_tests/perf_eltwise_binary.py

# Phase 2 — run on HW
pytest --compile-consumer --enable-perf-counters -x ./python_tests/perf_eltwise_binary.py
```

The `--enable-perf-counters` flag triggers two things:

1. Test sources are compiled with `-DPERF_COUNTERS_COMPILED` (the WC build). BRISC is compiled with the same flag so it runs `configure_and_arm_from_brisc()` once at startup.
2. The Python driver runs `read_counters()` per variant after the kernel finishes and merges raw counts into the result CSV.

Without the flag the suite still runs the same sources but builds the NC variant, and only `ZONE_SCOPED` wall-clock data is collected.

### CLI flags

| Flag | Implies `--enable-perf-counters` | Effect |
|------|----------------------------------|--------|
| `--enable-perf-counters` | — | Build the WC variant and collect raw counters per zone |
| `--dump-perf-counters` | yes | Export raw counter values to a separate `<test>.counters.csv` alongside the main results CSV |

`--dump-perf-counters` implicitly enables counter collection; you don't need to specify `--enable-perf-counters` separately.

### Output

For each test variant, the WC build emits:

- A row per `(zone, bank, counter_id, l1_mux)` in the main results DataFrame, with raw `cycles` and `count` columns.
- A `*.counters.csv` file if `--dump-perf-counters` was passed.

The NC build emits per-zone wall-clock cycle counts in the same results DataFrame so a single run with both builds (different pytest invocations) can be merged off-line to compare wall-clock cycles against counter-derived cycle counts.

## Architecture Summary

| | Wormhole | Blackhole |
|---|---|---|
| INSTRN_THREAD slots in inventory | 59 | 59 |
| FPU slots | 3 | 3 |
| TDMA_UNPACK slots | 22 | 22 |
| TDMA_PACK slots | 14 | 5 |
| L1 mux positions (Tensix) | 2 | 5 |
| L1 slots in inventory | 32 (16 × 2 mux) | 80 (16 × 5 mux) |
| Total slots in `BUILTIN_COUNTER_CONFIG` | 130 | 169 |
| Total config words in L1 | 200 (rest are zero-padded) | 200 |

**Wormhole** has `PACK_COUNT = 4` (per-engine packer busy signals are live in RTL), so `TDMA_PACK` exposes counters 11–18 for per-engine busy and 267–272 for per-engine dest-read availability and grant counts. The L1 mux is 1-bit wide: position 0 covers NoC Ring 0 plus L1 arbitration, position 1 covers NoC Ring 1 plus TDMA-extended signals.

**Blackhole** has `PACK_COUNT = 1`; per-engine packer busy and dest-read signals for engines 1–3 are tied to constants in RTL and are omitted from the inventory. Only counters 11, 18, 267, 271, 272 remain on the `TDMA_PACK` bank. BH compensates with more L1 mux positions (3 extra) which expose additional NoC rings and miscellaneous L1 ports.

**INSTRN_THREAD bank.** Counters 0–23 are per-thread instruction-type availability (CFG/SYNC/THCON/XSEARCH/MOVE/FPU/UNPACK/PACK, 3 threads each). Counters 24–26 are per-thread total stall cycles. The stall-reason layout differs:

- WH: shared stall reasons (SRCA/B clear/valid) are replicated three times each (counters 27–38), then per-thread stall reasons occupy counters 39–65.
- BH: shared stall reasons occupy single slots (27–30), per-thread stall reasons occupy 31–57.

Bit-8-extended counters 256/264/272 expose `THREAD_INSTRUCTIONS_{0,1,2}` (one per per-thread instance), and 283 exposes `ANY_THREAD_STALL`. The full per-arch inventory is in `BUILTIN_COUNTER_CONFIG[]` inside `counters.h`.

## L1 Layout and Zone Buffers

Counter state lives at a fixed L1 address determined entirely at compile time. No allocator is involved.

```
0x169000 +────────────────────────────────────────────+
         │ Shared config:  200 words × 4 = 800 bytes │  Slot encoding:
         │   bit 31     = valid                      │   bit 31     valid
         │   bits 19:17 = l1_mux                     │   bits 19:17 l1_mux
         │   bits 16:8  = counter_sel                │   bits 16:8  counter_sel
         │   bits 7:0   = bank_id                    │   bits 7:0   bank_id
0x169320 +────────────────────────────────────────────+
         │ Zone 0 cycles (5 words: one per bank)    │
         │ Zone 0 data   (200 words: counter values) │
         │ Zone 0 sync   (40 bytes; SYNC flag + pad) │  = PERF_COUNTERS_ZONE_SIZE
         +────────────────────────────────────────────+
         │ Zone 1 cycles / data / sync               │
         │ ...                                       │
         │ Zone 7 cycles / data / sync               │
         +────────────────────────────────────────────+
         │ Enabled flag (4 bytes)                    │
         │ Bank mask    (4 bytes)                    │
         │ Per-zone valid counts (8 words)           │
         +────────────────────────────────────────────+
```

The layout is bounded by a `static_assert` to stay below `0x16AFF4` (the profiler region boundary). Each zone reserves `PERF_COUNTERS_ZONE_SIZE = (5 + 200) × 4 + 40 = 860` bytes, supporting up to `PERF_COUNTERS_MAX_ZONES = 8` zones per kernel.

The 200-word shared config supplies a single source of truth for which counters are recorded for every zone. There is no per-zone configuration — every zone records the same set of counters but stores its own snapshot.

## Hardware Register Reference

The following addresses are used (offsets from `RISCV_DEBUG_REGS_START_ADDR = 0xFFB12000`):

| Register | Offset | Description |
|----------|--------|-------------|
| `PERF_CNT_INSTRN_THREAD0` | 0x000 | Reference period (mode 1) |
| `PERF_CNT_INSTRN_THREAD1` | 0x004 | Mode + counter_sel |
| `PERF_CNT_INSTRN_THREAD2` | 0x008 | Start/Stop (rising edge) |
| `PERF_CNT_TDMA_UNPACK0..2` | 0x00C–0x014 | Same triplet |
| `PERF_CNT_FPU0..2` | 0x018–0x020 | Same triplet |
| `PERF_CNT_L1_0..2` | 0x030–0x038 | Same triplet |
| `PERF_CNT_ALL` | 0x03C | Global start/stop for FPU + INSTRN_THREAD |
| `PERF_CNT_TDMA_PACK0..2` | 0x0F0–0x0F8 | Same triplet |
| `PERF_CNT_OUT_L_INSTRN_THREAD` | 0x100 | Elapsed cycles for bank |
| `PERF_CNT_OUT_H_INSTRN_THREAD` | 0x104 | Event count for selected `counter_sel` |
| `PERF_CNT_OUT_L_TDMA_UNPACK` | 0x108 | … |
| `PERF_CNT_OUT_H_TDMA_UNPACK` | 0x10C | … |
| `PERF_CNT_OUT_L_TDMA_PACK` | 0x110 | … |
| `PERF_CNT_OUT_H_TDMA_PACK` | 0x114 | … |
| `PERF_CNT_OUT_L_DBG_L1` | 0x118 | … |
| `PERF_CNT_OUT_H_DBG_L1` | 0x11C | … |
| `PERF_CNT_OUT_L_FPU` | 0x120 | … |
| `PERF_CNT_OUT_H_FPU` | 0x124 | … |
| `PERF_CNT_MUX_CTRL` | 0x218 | L1 mux selector (bits 6:4) |

### Mode register (`PERF_CNT_*1`)

| Bits | Field | Description |
|------|-------|-------------|
| 7:0 | mode | 0 = continuous with cycle tracking; 1 = stop after `PERF_CNT_*0` cycles; 2 = continuous without cycle tracking |
| 16:8 | counter_sel | Selects which counter event is routed to `OUT_H` |
| 31:17 | reserved | — |

The macro path always uses mode 0. Mode 1 is unused in the LLK test suite. The `counter_sel` field is rewritten on each slot read so a single bank can multiplex multiple counters into one measurement window.

### Start/Stop register (`PERF_CNT_*2`)

Rising-edge triggered. Bit 0 = start (0→1 also clears the counter), bit 1 = stop. The macro writes `1` then immediately writes `0` on both arm and freeze paths to guarantee the next arm sees a clean 0→1 transition.

### L1 mux (`PERF_CNT_MUX_CTRL`)

The L1 bank has only 8 physical counters, but the hardware exposes 16 (WH) or 40 (BH) client signals. Bits 6:4 of `PERF_CNT_MUX_CTRL` select which group of 8 signals the counters are wired to:

| Mux | WH clients | BH clients |
|-----|------------|------------|
| 0 | unpacker 0, port 1 (pack1+ECC), TDMA bundles, NoC Ring 0 | unpacker 0, port 1 (unpacker1+ECC), TDMA bundles, NoC Ring 0 |
| 1 | ext unpackers, NoC Ring 1, TDMA extended | ext unpackers 1-3, NoC Ring 1, TDMA packer 2 |
| 2 | — | ext unpackers 4-7, NoC Ring 0 secondary channels |
| 3 | — | NoC Ring 1 secondary channels, ext packers 2-5 |
| 4 | — | ext packers 6-7, tag search / packer 1 |

The mux routes signals at **count time**: whichever group is selected when the counters start is the only one measured during the window, so one run captures one L1 group. The harness pins the group at compile time (`LLK_PERF_L1_MUX_GROUP`, default 0) and the host read fails loudly on a snapshot captured with a different group; capturing another group means another run with another build.

## Derived Metrics Reference

The metric formulas live in one shared module, [tools/tracy/perf_metrics_common.py](../../../../tools/tracy/perf_metrics_common.py), used by this harness and by the metal-level Tracy tool, so both report the same numbers from the same counters. The complete catalogue (all 107 metrics, grouped by topic, with formulas) lives in the [PerfCounters tech report](../../../../tech_reports/PerfCounters/perf-counters.md#derived-metrics-reference); this section only covers how the harness presents them.

**Column naming.** The merged CSV carries one column per metric key, prefixed with the run type: a single run produces `<RUN_TYPE>_<key>` (e.g. `L1_TO_L1_fpu_utilization_pct`), several runs produce `<RUN_TYPE>_mean(<key>)` and `<RUN_TYPE>_std(<key>)`. The valid key set is `METRIC_BASES` in [helpers/perf/schema.py](../../tests/python_tests/helpers/perf/schema.py), which is derived from the shared module and therefore cannot drift from it.

**Metric families.** `*_pct` keys are bounded percentages (0-100%). `*_ratio` keys are unbounded raw ratios that can exceed 1.0 by design (cross-domain numerator and denominator, or overlapping sums) and are never clamped.

**N/A semantics.** A metric whose counters do not exist on the running architecture is empty in the CSV, never 0 — the harness computes per-zone snapshots, and any counter absent from the captured groups makes its dependent cross-bank metrics empty too. In particular the Wormhole-only per-engine packer metrics are empty on Blackhole, and the Blackhole-only extended L1 groups are empty on Wormhole.

**Per-zone, not per-op.** Unlike the Tracy tool (which aggregates Min/Median/Max/Avg across cores per operation), this harness measures one core and reports each zone (`INIT`, `TILE_LOOP`) separately, so a metric here answers "what did this zone of this variant do", not "how did the op behave across the grid".

## Notes and Caveats

- **NC vs WC are mutually exclusive.** A given pytest invocation produces one build, so wall-clock and counter data come from separate runs. Merge them off-line by `(test_variant, zone)`.
- **The arm/freeze split shifts zone boundaries slightly.** For `L1_TO_L1` and `L1_CONGESTION`, the measurement window opens when unpack arms (before unpack issues its first instruction inside the scope) and closes when pack freezes (after pack issues its last). Counter values from these run types are not directly comparable to a hypothetical "all three threads start and stop simultaneously" baseline.
- **`PERF_COUNTERS_MAX_ZONES = 8` per kernel.** Adding a 9th distinct `MEASURE_PERF_COUNTERS("...")` name silently reuses zone 0. Reuse the same name across multiple call sites if you want them in the same bucket.
- **One L1 mux group per run.** The L1 mux routes signals at count time, so a run only ever measures the group `LLK_PERF_L1_MUX_GROUP` selected at compile time; the other groups' counter slots stay empty. Reading with a different group selected does not recover them.
- **BRISC compile flag.** When `--enable-perf-counters` is set, BRISC is rebuilt with `-DPERF_COUNTERS_COMPILED`. Otherwise BRISC does not touch the counter HW at all — this keeps the NC build free of any counter-armed monitoring overhead.
- **Test isolation.** As with every LLK test, counter state at kernel entry is whatever the previous test left behind. The BRISC reset path clears the shared config and zone buffers, so each test starts from a known L1 state, but HW counter registers themselves may carry residual values until the first `MEASURE_PERF_COUNTERS` rising-edge clear.
