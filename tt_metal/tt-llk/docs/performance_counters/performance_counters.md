# LLK Performance Counters Guide

## Quick Links
- Device-side counter HW driver: [tests/helpers/include/counters.h](../../tests/helpers/include/counters.h)
- Test-helper mock functions: [tests/helpers/include/perf.h](../../tests/helpers/include/perf.h)
- Profiler zone macros: [tests/helpers/include/profiler.h](../../tests/helpers/include/profiler.h)
- Host-side counter readback: [tests/python_tests/helpers/counters.py](../../tests/python_tests/helpers/counters.py)
- Host-side derived metrics: [tests/python_tests/helpers/metrics.py](../../tests/python_tests/helpers/metrics.py)
- Test driver: [tests/python_tests/helpers/perf.py](../../tests/python_tests/helpers/perf.py)
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
pytest --compile-producer --enable-perf-counters -n 8 -x ./python_tests/perf_eltwise_binary_fpu.py

# Phase 2 — run on HW
pytest --compile-consumer --enable-perf-counters -x ./python_tests/perf_eltwise_binary_fpu.py
```

The `--enable-perf-counters` flag triggers two things:

1. Test sources are compiled with `-DPERF_COUNTERS_COMPILED` (the WC build). BRISC is compiled with the same flag so it runs `configure_and_arm_from_brisc()` once at startup.
2. The Python driver runs `read_counters()` per variant after the kernel finishes and merges raw counts into the result CSV.

Without the flag the suite still runs the same sources but builds the NC variant, and only `ZONE_SCOPED` wall-clock data is collected.

### CLI flags

| Flag | Implies `--enable-perf-counters` | Effect |
|------|----------------------------------|--------|
| `--enable-perf-counters` | — | Build the WC variant and collect raw counters per zone |
| `--dump-raw-counters` | yes | Print raw HW counter values to the console per variant |
| `--dump-raw-metrics` | yes | Print derived efficiency metrics (utilisation, stall, BP %) to the console |
| `--dump-csv-counters` | yes | Export raw counter values to a separate `<test>.counters.csv` alongside the main results CSV |

Any of `--dump-raw-counters`, `--dump-raw-metrics`, or `--dump-csv-counters` implicitly enables counter collection; you don't need to specify `--enable-perf-counters` separately.

### Output

For each test variant, the WC build emits:

- A row per `(zone, bank, counter_id, l1_mux)` in the main results DataFrame, with raw `cycles` and `count` columns.
- A `*.counters.csv` file if `--dump-csv-counters` was passed.
- A merged metrics summary (Min / Median / Max / Avg of every derived metric across variants) if `--dump-raw-metrics` was passed.

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

The L1 bank has 8 `counter_id` values (0–7) but hardware exposes 16 (WH) or 40 (BH) distinct signals. Bits 6:4 of `PERF_CNT_MUX_CTRL` select which set of 8 signals is routed:

| Mux | WH meaning | BH meaning |
|-----|------------|------------|
| 0 | NoC Ring 0 + L1 arbitration + unpacker | NoC Ring 0 |
| 1 | NoC Ring 1 + TDMA extended | NoC Ring 1 |
| 2 | — | NoC Ring 2 |
| 3 | — | NoC Ring 3 |
| 4 | — | Misc L1 ports |

The mux is latched at counter-start time. The macro path re-writes it before each slot read during freeze, so a single zone snapshot can contain counters from multiple mux positions interleaved.

## Derived Metrics Reference

Derived metrics are computed in `tests/python_tests/helpers/metrics.py` from the raw counter DataFrame. The metric set mirrors the metal-level [PerfCounters tech report](../../../../tech_reports/PerfCounters/perf-counters.md) — the same catalogue applies to **both Wormhole and Blackhole** (architecture differences are confined to a few WH-only or BH-only counters, called out per-metric). The LLK driver operates on per-zone snapshots rather than per-op aggregates, so all derived values appear in the merged CSV and the `--dump-raw-metrics` console output.

> **Full catalogue.** Metrics #1–#47 in `tech_reports/PerfCounters/perf-counters.md` are the authoritative list. The sections below document the ones the LLK driver surfaces directly; raw counters for every other upstream metric are present in the per-zone CSV, so any upstream formula can be re-evaluated on LLK data without code changes.

---

### Compute Utilisation

**1. FPU Utilisation**

Fraction of elapsed cycles the FPU was executing an instruction.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU |

```
FPU Util = FPU_INSTRUCTION / FPU_OUT_L * 100
```

- **High value (>20%)**: FPU is the active compute unit. Expected for matmul, eltwise multiply.
- **Low value (~0%)**: FPU is idle. Expected for SFPU-only or pure data-movement zones.

**Use case:** Primary indicator of compute utilisation for FPU-path kernels.

---

**2. SFPU Utilisation**

Fraction of cycles the SFPU was active.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU |

```
SFPU Util = SFPU_INSTRUCTION / FPU_OUT_L * 100
```

- **High value (>20%)**: SFPU-heavy kernel (sqrt, gelu, exp).
- **Low value (~0%)**: SFPU unused (FPU-path or data movement).

**Use case:** Confirms whether a zone exercises the SFPU pipeline.

---

**3. Math Utilisation**

Combined FPU+SFPU active cycles. Counter 257 is the OR of both unit-active signals.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU |

```
Math Util = FPU_OR_SFPU_INSTRN / FPU_OUT_L * 100
```

**Use case:** Single-number compute utilisation across FPU and SFPU.

---

### Pipeline Efficiency

**4. Packer Efficiency**

Fraction of packer-busy cycles where destination data was available.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_PACK |

```
Packer Efficiency = PACKER_DEST_READ_AVAILABLE / PACKER_BUSY * 100
```

- **High value (~100%)**: Packer never waits for math output (no dest-read stalls).
- **Low value (<80%)**: Packer is busy but math has not finished writing the destination — math is the bottleneck.

**Use case:** Detects destination-register stalls indicating the math stage cannot keep up.

---

**5. Math Pipeline Utilisation**

Math-instruction issue rate relative to availability.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
Math Pipeline Utilisation = MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE * 100
```

- **High value (>95%)**: Math instructions issue as fast as they arrive at the front end.
- **Low value (<60%)**: Math is stalled in flight (scoreboard, dest port, fidelity).

**Use case:** Distinguishes "math has nothing to do" from "math is stalled while busy".

---

**6. FPU Execution Efficiency**

FPU active cycles relative to math-instruction availability on the math thread.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU + INSTRN_THREAD |

```
FPU Execution Efficiency = FPU_INSTRUCTION / FPU_INSTRN_AVAILABLE_1 * 100
```

- **High value (>80%)**: FPU executes whenever math work is pending — compute-bound.
- **Low value (<30%)**: Math instructions queued but FPU not running — pipeline-stall-bound.

**Use case:** Compute-bound vs stall-bound discriminator on the math path.

---

**7. Math-to-Pack Handoff Ratio**

Ratio of math-output availability to packer consumption.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_PACK |

```
Math-to-Pack Handoff = AVAILABLE_MATH / PACKER_BUSY * 100
```

- **>100%**: Math produces output faster than packer can consume (packer is the bottleneck).
- **~100%**: Balanced.
- **<100%**: Packer is stalled waiting on math — math is the bottleneck.

**Use case:** Identifies math-vs-pack pipeline imbalance.

---

**8. Unpacker-to-Math Data Flow**

Backpressure from math stage onto the unpackers.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
Unpacker-to-Math = avg(SRCA_WRITE_AVAILABLE, SRCB_WRITE_AVAILABLE) /
                   avg(UNPACK0_BUSY_THREAD0, UNPACK1_BUSY_THREAD0) * 100
```

- **High value (>80%)**: Unpackers always have somewhere to write — no backpressure.
- **Low value (<30%)**: Source-register buffers are full; math is not consuming fast enough.

**Use case:** Cross-check against per-unpacker write efficiency to isolate math-driven backpressure from DMA port contention.

---

### Thread Analysis

**9. Thread N Stall Rate**

Fraction of cycles each TRISC thread was stalled.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
Thread N Stall Rate = THREAD_STALLS_N / INSTRN_OUT_L * 100
```

Thread mapping: T0 = UNPACK, T1 = MATH, T2 = PACK.

- **High value (>30%)**: Thread waits on resources most of the time.
- **Low value (<5%)**: Thread keeps the issue pipeline full.

**Use case:** First-pass localisation of which thread is losing time. Follow up with the stall-reason breakdown.

---

**10. Thread N Issue Rate**

Average instructions issued per cycle, per thread.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
TN Issue Rate = THREAD_INSTRUCTIONS_N / INSTRN_OUT_L
```

- **High (~1.0)**: Thread issues an instruction nearly every cycle.
- **Low (<0.1)**: Thread is idle or blocked.

**Use case:** Detects threads that look "busy" by stall-rate but actually never issue work.

---

### Pipeline Wait Metrics

**11. SrcA/SrcB Valid Wait**

Cycles math waited for unpack to provide source data.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
SrcA Valid Wait = WAITING_FOR_SRCA_VALID / INSTRN_OUT_L * 100
SrcB Valid Wait = WAITING_FOR_SRCB_VALID / INSTRN_OUT_L * 100
```

- **High (>5%)**: Math is starved by the unpacker.
- **Low (~0%)**: Unpack keeps pace with math.

**Use case:** Detects unpack-side data starvation.

---

**12. SrcA/SrcB Clear Wait**

Cycles unpack waited for math to release a source register before overwriting.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
SrcA Clear Wait = WAITING_FOR_SRCA_CLEAR / INSTRN_OUT_L * 100
SrcB Clear Wait = WAITING_FOR_SRCB_CLEAR / INSTRN_OUT_L * 100
```

- **High (>10%)**: Math holds source registers longer than unpack can wait. Common for SFPU-heavy SrcA reuse.
- **Low (~0%)**: No register pressure between unpack and math.

**Use case:** Identifies source-register contention.

---

**13. Math / Pack / Unpack Idle Wait**

Cycles each thread waited for its primary HW unit to become idle.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
Math Idle Wait T1 = WAITING_FOR_MATH_IDLE_1 / INSTRN_OUT_L * 100
Pack Idle Wait T2 = WAITING_FOR_PACK_IDLE_2 / INSTRN_OUT_L * 100
Unpack Idle Wait T0 = WAITING_FOR_UNPACK_IDLE_0 / INSTRN_OUT_L * 100
```

**Use case:** Pinpoints which HW unit is the bottleneck.

---

### Semaphore Waits

**14. Semaphore Zero/Full Wait per Thread**

Cycles each thread spent blocked on a semaphore.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
Zero Wait TN = WAITING_FOR_NONZERO_SEM_N / INSTRN_OUT_L * 100
Full Wait TN = WAITING_FOR_NONFULL_SEM_N / INSTRN_OUT_L * 100
```

- **Zero Wait high**: Thread waits for a producer to signal.
- **Full Wait high**: Thread waits for a consumer to drain — downstream backpressure.

**Use case:** Identifies producer/consumer imbalance across threads.

---

### TDMA Stall Metrics

**15. Data Hazard Stall Rate**

Fraction of math-valid cycles stalled by dest-to-src hazards (MOVD2A / MOVD2B).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
Data Hazard Stall = (MATH_INSTRN_AVAILABLE - DATA_HAZARD_STALLS_MOVD2A) /
                    MATH_INSTRN_AVAILABLE * 100
```

`DATA_HAZARD_STALLS_MOVD2A` is `math_instrn_valid & ~dest2src_post_stall` — cycles math was available *and not* D2A-stalled. Subtracting from `MATH_INSTRN_AVAILABLE` gives the stall count.

**Use case:** Surfaces dest-to-src register-movement overhead.

---

**16. SrcA/SrcB Write Port Blocked**

Fraction of srcA/B DMA write attempts blocked by port unavailability.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
SrcA Port Blocked = (SRCA_WRITE_AVAILABLE - SRCA_WRITE_NOT_BLOCKED_PORT) /
                    SRCA_WRITE_AVAILABLE * 100
SrcB Port Blocked = (SRCB_WRITE_AVAILABLE - SRCB_WRITE_NOT_BLOCKED_PORT) /
                    SRCB_WRITE_AVAILABLE * 100
```

**Use case:** Isolates DMA-port contention from overwrite contention.

---

**17. SrcA/SrcB Write Overwrite Blocked**

Fraction of write attempts blocked because math has not consumed the previous value.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
SrcA Overwrite Blocked = (SRCA_WRITE_AVAILABLE - SRCA_WRITE_NOT_BLOCKED_OVR) /
                         SRCA_WRITE_AVAILABLE * 100
SrcB Overwrite Blocked = (SRCB_WRITE_AVAILABLE - SRCB_WRITE) /
                         SRCB_WRITE_AVAILABLE * 100
```

- **High (>30%)**: Math is slow to consume — typical for SFPU-heavy SrcA reuse.

**Use case:** Separates math-consumer bottleneck from DMA arbitration bottleneck.

---

**18. Dest Read Backpressure**

Fraction of packer dest-read requests not granted.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_PACK |

```
Dest Read BP = (PACKER_DEST_READ_AVAILABLE - DEST_READ_GRANTED_0) /
               PACKER_DEST_READ_AVAILABLE * 100
```

**Use case:** Detects math-to-pack register handoff stalls.

---

**19. Math Scoreboard Stall Rate**

Fraction of math-available cycles stalled by FPU data-hazard scoreboard.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_PACK |

```
Math Scoreboard Stall = (MATH_INSTRN_AVAILABLE - AVAILABLE_MATH) /
                        MATH_INSTRN_AVAILABLE * 100
```

**Use case:** Identifies FPU pipeline hazards (RAW / WAW) blocking math issue.

---

### Instruction Availability

**20. Per-type Instruction Availability**

Fraction of cycles each instruction type was queued in its primary thread's buffer.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
TYPE Avail Rate = TYPE_INSTRN_AVAILABLE_N / INSTRN_OUT_L * 100
```

Types: CFG, SYNC, THCON, MOVE (T0), FPU/MATH (T1), UNPACK (T0), PACK (T2), XSEARCH.

**Use case:** Shows which instruction types dominate the scheduling pipeline.

---

### Write Port Analysis

**21. SrcA Write Actual Efficiency**

Fraction of srcA write attempts that succeeded at the DMA port.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
SrcA Write Actual Efficiency = SRCA_WRITE / SRCA_WRITE_AVAILABLE * 100
```

- **High (100%)**: Every srcA write succeeds — no write-port blocking.
- **Low (<80%)**: A significant fraction of srcA writes are blocked.

**Use case:** Measures effective srcA write throughput; low values indicate port contention.

---

**22. SrcB Write Actual Efficiency**

Fraction of srcB write attempts that succeeded at the DMA port.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
SrcB Write Actual Efficiency = SRCB_WRITE / SRCB_WRITE_AVAILABLE * 100
```

Mirrors metric 21 for srcB; both archs expose this independently.

**Use case:** Effective srcB write throughput. Compare with srcA for asymmetric blocking.

---

**23. Unpacker N Write Efficiency**

Fraction of unpacker-busy cycles that actually completed a write.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
Unpacker0 Write Eff = SRCA_WRITE / UNPACK0_BUSY_THREAD0 * 100
Unpacker1 Write Eff = SRCB_WRITE / UNPACK1_BUSY_THREAD0 * 100
```

**Use case:** Identifies whether unpacker stalls are from port contention or overwrite blocking — compare with metrics 16 and 17.

---

### L1 Memory and NoC

**24. L1 Port / NoC Ring Utilisation**

Per-port and per-NoC-channel utilisation.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 |

```
L1 Port Util  = L1_PORT_REQ / L1_OUT_L * 100
NoC Ring Util = avg(NOC_RINGN_CHANNEL_0, NOC_RINGN_CHANNEL_1) / L1_OUT_L * 100
```

**Use case:** Identifies which L1 ports and NoC rings carry traffic in the measured zone. BH exposes additional rings via `l1_mux = 2..4`.

---

**25. L1 Backpressure**

Fraction of requested cycles where L1 did not grant.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 |

```
L1 BP = (REQ - GRANT) / REQ * 100
```

Computed per port and per NoC channel.

**Use case:** Detects L1 port contention. Compare unpacker port BP with Thread 0 stall rate to confirm L1 is the bottleneck.

---

### Composite

**26. Stall Cause Overlap Factor per Thread**

Ratio of summed per-thread stall reasons to total thread stalls.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
Stall Overlap TN = sum(all WAITING_FOR_*_N) / THREAD_STALLS_N
```

- **~1.0×**: Single dominant stall reason at any given cycle.
- **>2.0×**: Multiple HW units busy simultaneously when the thread stalls.

**Use case:** Tells you whether to chase a single bottleneck or a set of interacting ones.

---

**27. Compute-to-Unpack Ratio**

Whether the zone is compute-bound or memory-bound.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU + TDMA_UNPACK |

```
Compute-to-Unpack = FPU_OR_SFPU_INSTRN / (UNPACK0_BUSY_THREAD0 + UNPACK1_BUSY_THREAD0) * 100
```

- **>100%**: Compute-bound.
- **<20%**: Memory-bound (unpackers busier than math).

**Use case:** Quick one-number diagnostic per zone.

---

### Fidelity

**28. Fidelity Stall Rate**

Fraction of math-valid cycles spent in a fidelity phase (multi-HF-cycle math instruction).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
Fidelity Stall Rate = MATH_FIDELITY_STALL / MATH_INSTRN_AVAILABLE * 100
```

- **0%**: Pure LoFi (every math instruction completes in 1 HF cycle).
- **>0%**: HiFi2 or HiFi4 active — multi-cycle math contributes to wall time.

> **Known issue:** On HiFi variants this metric can exceed 100% because the formula's numerator counts every HF cycle of multi-HF instructions while the denominator counts only the issued instructions. The math is being re-calibrated; treat values >100% as "fidelity is the dominant cost" rather than a literal percentage.

**Use case:** Detects whether fidelity is contributing to the cycle budget.

---

**29. HiFi Fraction**

Fraction of issued math instructions that took more than 1 HF cycle.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
HiFi Fraction = (MATH_INSTRN_HF_2_CYCLE + MATH_INSTRN_HF_4_CYCLE) /
                (MATH_INSTRN_HF_1_CYCLE + MATH_INSTRN_HF_2_CYCLE + MATH_INSTRN_HF_4_CYCLE) * 100
```

**Use case:** Quick check of fidelity mix in a workload. 0% = pure LoFi, 100% = pure HiFi.

---

**30. Avg HF Cycles Per Instrn**

Weighted average of HF cycles per issued math instruction (1 for LoFi, 2 for HiFi2, 4 for HiFi4).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_UNPACK |

```
Avg HF Cycles = (HF_1 + 2*HF_2 + 4*HF_4) / (HF_1 + HF_2 + HF_4)
```

**Use case:** Single-number summary of fidelity impact on math execution.

---

### TDMA / Math Stall (cont.)

**31. Math Dest Write Port Stall Rate**

Fraction of math cycles stalled by destination register write port contention.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | TDMA_PACK |

```
Math Dest Write Port Stall = (MATH_INSTRN_AVAILABLE - MATH_NOT_STALLED_DEST_WR_PORT) /
                             MATH_INSTRN_AVAILABLE * 100
```

- **High value (>10%)**: Math is stalled waiting for write port to destination register.
- **Low value (~0%)**: No write port stalls.

The metric is skipped when `MATH_NOT_STALLED_DEST_WR_PORT` reads 0 across the whole zone (would otherwise read a misleading 100%). Common on BH for workloads that don't drive the write-port path.

**Use case:** Detects destination register write contention from the math side.

---

### Additional Idle Waits

**32. MMIO / SFPU / THCON / MOVE Idle Wait**

Fraction of total cycles each thread spent waiting for specific hardware units.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN_THREAD |

```
MMIO Idle Wait T0  = WAITING_FOR_MMIO_IDLE_0  / INSTRN_OUT_L * 100
SFPU Idle Wait T1  = WAITING_FOR_SFPU_IDLE_1  / INSTRN_OUT_L * 100
THCON Idle Wait T0 = WAITING_FOR_THCON_IDLE_0 / INSTRN_OUT_L * 100
MOVE Idle Wait T0  = WAITING_FOR_MOVE_IDLE_0  / INSTRN_OUT_L * 100
```

- **High value (>5%)**: Significant time waiting on this unit. MOVE wait at ~3% for tilize is expected.
- **Low value (~0%)**: HW unit never bottlenecks the thread. THCON and MMIO are typically ~0%.

**Use case:** Absolute (not relative to total stalls) measure of time lost to each HW unit.

---

### L1 Memory and NoC (cont.)

**33. L1 TDMA Bundle Util**

Average utilisation of the two TDMA/RISC L1 ports.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
L1 TDMA Bundle Util = avg(L1_0_TDMA_BUNDLE_0_RISC, L1_0_TDMA_BUNDLE_1_TRISC) / L1_OUT_L * 100
```

**Use case:** Measures firmware + TDMA data movement overhead through L1.

---

**34. NoC Ring 0/1 Outgoing/Incoming Util**

Average utilisation of NoC channels per ring direction.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (Ring 0 on mux 0, Ring 1 on mux 1) |

```
NoC Ring 0 Outgoing Util = avg(L1_0_NOC_RING0_OUTGOING_0, L1_0_NOC_RING0_OUTGOING_1) / L1_OUT_L * 100
NoC Ring 0 Incoming Util = avg(L1_0_NOC_RING0_INCOMING_0, L1_0_NOC_RING0_INCOMING_1) / L1_OUT_L * 100
```

**Use case:** Per-ring, per-direction NoC bandwidth utilisation. Compare outgoing vs incoming for data-flow direction.

---

**35. RISC Core L1 Util**

RISC core L1 memory access utilisation.

| | |
|---|---|
| **Architectures** | Blackhole only |
| **Counter group** | L1 (mux 1) |

```
RISC Core L1 Util = L1_1_RISC_CORE / L1_OUT_L * 100
```

- **High (>10%)**: RISC core is actively touching L1 — firmware memory overhead.
- **Low (~0%)**: Minimal RISC L1 traffic.

**Use case:** Quantifies firmware memory access overhead on BH. Requires the L1 mux-1 slot enabled.

---

### L1 Backpressure

**36. NoC Ring 0/1 Outgoing/Incoming Backpressure**

Fraction of NoC transaction cycles where L1 was not ready.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 |

```
NoC Ring 0 Outgoing BP = (req0 + req1 - grant0 - grant1) / (req0 + req1) * 100
```

- **High (>15%)**: NoC is stalled by L1 contention.
- **Low (<5%)**: NoC traffic flows freely.

**Use case:** High outgoing BP means produced data can't leave the core fast enough.

---

**37. L1 Unpacker / Packer Port Backpressure**

L1 port contention for unpacker and packer.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
L1 Unpacker BP   = (L1_0_UNPACKER_0 - L1_0_UNPACKER_0_GRANT) / L1_0_UNPACKER_0 * 100
L1 Packer Port BP = (L1_0_PORT1     - L1_0_PORT1_GRANT)      / L1_0_PORT1 * 100
```

- **Unpacker BP high (>80%)**: L1 is busy when unpacker wants in — common (other ports compete).
- **Packer Port BP low (<5%)**: Normal.

**Blackhole note:** Unpacker grant counter can exceed request on some cores (signal-semantics difference); the metric is suppressed rather than reported as a meaningless negative.

**Use case:** Investigate only when combined with high Thread 0 stall rate.

---

### L1 Composite

**38. L1 Total Bandwidth Util**

Overall L1 bandwidth saturation across all 8 mux-0 ports.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
L1 Total BW Util = sum(all 8 port req counts) / (8 * L1_OUT_L) * 100
```

- **High (>30%)**: L1 heavily utilised — possible bottleneck.
- **Medium (10-20%)**: Moderate.
- **Low (<5%)**: Underutilised.

**Use case:** Single-number L1 saturation indicator.

---

**39. L1 Read vs Write Ratio**

Balance between read and write traffic on L1.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
L1 R/W Ratio = (Unpacker + NoC_Out) / (Unpacker + NoC_Out + Packer + NoC_In) * 100
```

Read ports: unpacker, NoC outgoing. Write ports: packer, NoC incoming.

- **~50%**: Balanced (matmul).
- **>70%**: Read-heavy.
- **<30%**: Write-heavy.

**Use case:** Diagnoses the dominant L1 data-flow direction.

---

**40. NoC Ring Asymmetry**

Balance between outgoing and incoming NoC traffic.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
NoC Asymmetry = NoC_Outgoing / (NoC_Outgoing + NoC_Incoming) * 100
```

- **~50%**: Balanced send/receive.
- **>70%**: Send-heavy.
- **<30%**: Receive-heavy.

**Use case:** Surfaces directional NoC imbalance pointing to data-placement issues.

---

**41. L1 Contention Index**

Average backpressure across active L1 ports.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
L1 Contention Index = avg(BP of Unpacker, NoC Out 0, NoC Out 1, NoC In 0, NoC In 1)
```

- **High (>40%)**: Significant L1 stress.
- **Medium (15-30%)**: Moderate.
- **Low (<10%)**: Minimal.

**Use case:** Single-number L1 stress level — easier to compare across zones than individual port BPs.

---

**42. Unpacker L1 Efficiency**

When the unpacker is busy, how often does L1 actually serve it.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) + TDMA_UNPACK |

```
Unpacker L1 Efficiency = L1_0_UNPACKER_0_GRANT / UNPACK0_BUSY_THREAD0 * 100
```

- **High (>50%)**: L1 serves unpacker requests efficiently.
- **Low (<5%)**: L1 is the data-delivery bottleneck.

**Use case:** Low values combined with high unpacker BP confirm L1 is starving the unpacker.

---

**43. Packer L1 Efficiency**

When the packer is busy, how often does L1 serve it.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) + TDMA_PACK |

```
Packer L1 Efficiency = L1_0_PORT1_GRANT / PACKER_BUSY * 100
```

- **High (>100%)**: L1 port has headroom (shared with ECC/other clients).
- **Low (<50%)**: Packer is L1-starved.

**Use case:** Confirms whether L1 write-back is the limiting factor for pack.

---

**44. NoC vs Compute Balance**

Whether the operation is NoC-bound or compute-bound.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) + FPU |

```
NoC vs Compute = (NoC_Out + NoC_In) / (FPU_COUNTER + NoC_Out + NoC_In) * 100
```

- **>60%**: NoC-bound.
- **~50%**: Balanced.
- **<40%**: Compute-bound (FPU/SFPU is the bottleneck).

**Use case:** Quick optimisation-target picker (compute kernels vs NoC routing).

---

**45. TDMA vs NoC L1 Share**

Fraction of L1 bandwidth used by RISC/TDMA versus NoC.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1 (mux 0) |

```
TDMA vs NoC = (TDMA_Bundle_0 + TDMA_Bundle_1) / (TDMA + NoC_Out + NoC_In) * 100
```

- **High (>20%)**: Firmware uses significant L1 bandwidth.
- **Low (<5%)**: NoC dominates L1 traffic.

**Use case:** Spots firmware L1 overhead worth optimising.

---

### Wormhole-only (per-engine packer)

**46. Packer Load Imbalance**

Spread between the most and least utilised packer engines.

| | |
|---|---|
| **Architectures** | Wormhole only (`PACK_COUNT=4`) |
| **Counter group** | TDMA_PACK |

```
Packer Load Imbalance = (max(BUSY_0..3) - min(BUSY_0..3)) / max(BUSY_0..3) * 100
```

- **Low (<10%)**: Even distribution across engines.
- **High (>25%)**: Some engines idle while others saturate — suboptimal tile packing.

**Use case:** WH only — per-engine busy signals are tied to 0 in BH RTL (`PACK_COUNT=1`).

---

**47. Packer Engine N Util**

Per-engine packer utilisation.

| | |
|---|---|
| **Architectures** | Wormhole only |
| **Counter group** | TDMA_PACK |

```
Packer Engine N Util = PACKER_BUSY_N / TDMA_PACK_OUT_L * 100
```

**Use case:** WH only. On BH the per-engine `PACKER_BUSY_N` signals are tied off; only the aggregate `PACKER_BUSY` is exposed.

---

## Notes and Caveats

- **NC vs WC are mutually exclusive.** A given pytest invocation produces one build, so wall-clock and counter data come from separate runs. Merge them off-line by `(test_variant, zone)`.
- **The arm/freeze split shifts zone boundaries slightly.** For `L1_TO_L1` and `L1_CONGESTION`, the measurement window opens when unpack arms (before unpack issues its first instruction inside the scope) and closes when pack freezes (after pack issues its last). Counter values from these run types are not directly comparable to a hypothetical "all three threads start and stop simultaneously" baseline.
- **`PERF_COUNTERS_MAX_ZONES = 8` per kernel.** Adding a 9th distinct `MEASURE_PERF_COUNTERS("...")` name silently reuses zone 0. Reuse the same name across multiple call sites if you want them in the same bucket.
- **L1 mux mutual exclusion is handled inside `MEASURE_PERF_COUNTERS`.** The freeze path re-programs `PERF_CNT_MUX_CTRL` before each L1 slot read, so a single zone snapshot contains counters from multiple mux positions without per-zone configuration changes.
- **BRISC compile flag.** When `--enable-perf-counters` is set, BRISC is rebuilt with `-DPERF_COUNTERS_COMPILED`. Otherwise BRISC does not touch the counter HW at all — this keeps the NC build free of any counter-armed monitoring overhead.
- **Test isolation.** As with every LLK test, counter state at kernel entry is whatever the previous test left behind. The BRISC reset path clears the shared config and zone buffers, so each test starts from a known L1 state, but HW counter registers themselves may carry residual values until the first `MEASURE_PERF_COUNTERS` rising-edge clear.
