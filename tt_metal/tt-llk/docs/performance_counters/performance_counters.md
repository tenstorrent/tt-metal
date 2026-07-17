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

| Build | `LLK_PROFILER` | `PERF_COUNTERS_COMPILED` | Active half(s) of `START_PERF_MEASURE` | What it measures |
|-------|----------------|--------------------------|--------------------------------------|------------------|
| NC (no counters) | defined | undefined | `ZONE_SCOPED` (timing) + `MEASURE_PERF_COUNTERS` (barrier only) | Per-zone wall-clock cycles (`RISCV_DEBUG_REG_WALL_CLOCK_L`) |
| WC (with counters) | defined | defined | `MEASURE_PERF_COUNTERS` **and** `ZONE_SCOPED` | Per-zone HW counter snapshot **and** wall-clock cycles |

`START_PERF_MEASURE(name)` expands to `MEASURE_PERF_COUNTERS(name)` + `ZONE_SCOPED(name)`. In the NC build, `MEASURE_PERF_COUNTERS` performs a no-op cross-thread rendezvous and `ZONE_SCOPED` records the per-zone wall-clock timestamps. In the WC build **both** are live: the counter scope performs the rendezvous *and* arms/freezes the HW counters, while `ZONE_SCOPED` records the per-zone wall-clock timestamps without adding another rendezvous. A single WC run therefore yields both counter and wall-clock data per zone under the same name; the host driver keys everything by `(test_variant, zone)` and can merge NC and WC results (or use the WC wall-clock directly).

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

### `PerfRunType` and the single-thread arm/freeze model

Each LLK perf test is associated with a `PerfRunType` (declared in `perf.h`): `L1_TO_L1` and `L1_CONGESTION` run the full unpack → math → pack pipeline; `UNPACK_ISOLATE` / `MATH_ISOLATE` / `PACK_ISOLATE` exercise a single stage. The run type only selects which thread(s) do real work inside `TILE_LOOP`.

**A single fixed thread — pack — arms and freezes the counters for every run type.** `is_perf_actor_thread()` in `counters.h` returns true only on pack. This is run-type-independent and correct because:

- The perf counters are **global hardware** driven by shared debug registers (`PERF_CNT_ALL` and the per-bank `*2` command registers), so any RISC can arm/freeze them — the identity of the issuing thread does not change what is counted.
- The rendezvous is an **actor-wait barrier** (`sync_point`, see below): the arm thread waits for *all* threads to arrive before arming, and the freeze thread waits for *all* threads to finish before freezing. So the counter window brackets every thread's work regardless of which thread issues the arm/freeze — verified across all perf tests, including `L1_CONGESTION`, where the last-exiting thread is workload-dependent.

Pack is chosen because a measurement sweep found it is the actual last-exiting thread in the majority of cases, minimising the (out-of-window) idle spin before the freeze. An earlier revision used a per-run-type source/sink split (unpack arms, pack freezes for pipeline modes); the barrier makes that distinction unnecessary, so it was collapsed to one fixed thread.

### The `MEASURE_PERF_COUNTERS` macro

Expands to a `perf_counter_scoped` RAII object. Its constructor and destructor execute the following sequence (only on the WC build):

1. **Constructor (zone entry).** Calls `sync_point(is_perf_actor_thread(), arm_all_counters)`. All three threads rendezvous; the **actor thread (pack)** then writes the rising-edge start bit to `PERF_CNT_ALL` (FPU + INSTRN), `PERF_CNT_TDMA_UNPACK2`, `PERF_CNT_L1_2`, and `PERF_CNT_TDMA_PACK2` — clearing all banks and starting the count — and releases the others.

2. **Body.** All three threads run the work inside the scope. Counters tick continuously on the shared backend.

3. **Destructor (zone exit).** Calls `sync_point(is_perf_actor_thread(), freeze_and_read_all_counters)`. Every thread has already drained its backend (`PROFILER_SYNC`/`tensix_sync`) and written its `ZONE_END` before arriving. The **actor thread (pack)** writes the rising-edge stop bit to the same four registers, then walks the shared 200-word config buffer at `0x169000` and reads every valid slot: for each it programs the bank's mode register with the `counter_sel`, reads `OUT_H` (the event count), and stores it in the per-zone data area (`OUT_L`, the elapsed cycles, is sampled once per bank). It then sets the zone's `SYNC_ZONE_COMPLETE` flag and releases the others.

Each zone gets its own data block in L1 (see [L1 Layout](#l1-layout-and-zone-buffers)) so multiple measurement scopes in the same kernel produce independent snapshots. The kernel may contain up to `PERF_COUNTERS_MAX_ZONES = 8` distinct zone names; identical names share a zone.

#### The `sync_point` rendezvous

The barrier is `llk_profiler::sync_point` — the **same epoch barrier the NC (profiler) build uses at each zone entry**, so both builds share one synchronization primitive. It is an *actor-release* barrier on the profiler's L1 array: every thread announces arrival on its per-thread generation slot; the designated actor (the arm or freeze thread) waits for all arrivals, runs its action (arm / freeze+read / no-op), then bumps a shared `epoch` word to release the waiters. Because the actor's only spin is the arrival wait — *before* its action — it never spins inside the counter window, so the barrier cannot pollute the measurement. It uses only plain L1 loads/stores plus `invalidate_data_cache` (no atomics, no scarce semaphores). An earlier revision used `pc_buf` semaphores for the WC arm/freeze handshake; `sync_point` replaced them so NC and WC use one mechanism.

### Configure-once from BRISC

Before any TRISC kernel runs, BRISC executes `configure_and_arm_from_brisc()` once (called from `brisc.cpp` when the WC build flag is set). This:

- Writes the per-architecture `BUILTIN_COUNTER_CONFIG` (130 slots on WH, 169 on BH) into the shared L1 config buffer at `0x169000`. That array is built at compile time from the canonical metal inventory — see [Counter inventory single source](#counter-inventory-single-source).
- Clears every per-zone data area and sync word.
- Clears `DBG_FEATURE_DISABLE` to `0` — see [DBG_FEATURE_DISABLE scrub](#dbg_feature_disable-scrub) below.
- Programs each bank's reference-period and mode registers, sets `PERF_CNT_MUX_CTRL` for L1, and does an initial global arm (later overridden by the first `MEASURE_PERF_COUNTERS` zone).

After BRISC releases the TRISCs, the shared config is read-only for the rest of the run.

##### `DBG_FEATURE_DISABLE` scrub

`DBG_FEATURE_DISABLE` is a 16-bit debug/chicken-bit register whose bits toggle low-level behaviors — notably randomized L1 arbitration (`DBG_FEAT_L1_ARB_LFSR_EN`, `0x8`), L1 atomic serialization, and L1 read-enable override. It resets to `0` (all normal), but HW register state **leaks between tests** run back-to-back on an un-reset device, so a prior test that set one of these bits would silently perturb — and make nondeterministic — the ~80 L1 counters. BRISC writes `0` here to guarantee a clean baseline regardless of leaked state; the blanket write (rather than clearing one bit) is deliberate because any of the bits, not just LFSR, would skew the measurement. Verified: with a leaked `0x8` present, the L1 metrics jitter 40–98 % run-to-run without this scrub and are byte-identical with it. Note this scrub is WC-only (it lives in the counter path); the NC path has no equivalent.

### Reading results from host

After the kernel completes:

1. The host process reads the per-zone data area back from device L1.
2. `read_counters()` decodes each 32-bit config word (bit 31 valid, bits 7:0 bank, bits 16:8 `counter_sel`, bits 19:17 `l1_mux`), looks up the human-readable counter name (parsed at import from the same `hw_counters.h` — see [Counter inventory single source](#counter-inventory-single-source)), and pairs every event count with that zone's bank cycle count.
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

Bit-8-extended counters 256/264/272 expose `THREAD_INSTRUCTIONS_{0,1,2}` (one per per-thread instance), and 283 exposes `ANY_THREAD_STALL`.

### Counter inventory single source

The counter id↔name inventory is **defined once**, in metal's canonical `tt_metal/hw/inc/internal/tt-1xx/<arch>/hw_counters.h` — grouped `{PerfCounterType, id}` arrays per bank (`instrn_counters`, `fpu_counters`, `unpack_counters`, `pack_counters`, `l1_0..4_counters`). Both sides of the perf infra derive from it, so the list is never hand-maintained twice:

- **Device (`counters.h`)** `#include`s `hw_counters.h` (with the `PerfCounterType` enum from `perf_counters.hpp`) and builds `BUILTIN_COUNTER_CONFIG[]` from those arrays at compile time — a `constexpr` concatenation in the fixed bank order the readout expects (INSTRN, FPU, TDMA_UNPACK, TDMA_PACK, then L1 by ascending mux).
- **Host (`counters.py`)** parses the same `hw_counters.h` at import to recover the id→name tables used for decoding.

Adding or removing a counter in `hw_counters.h` therefore propagates to both automatically; the only pieces still mirrored by hand are the config-word bit layout (`PERF_CFG_*`) and the bank-id↔name mapping, which are this test infra's own L1 ABI rather than part of the HW inventory.

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

The layout is bounded by a `static_assert` to stay below `0x16AFF0` (the profiler region boundary). Each zone reserves `PERF_COUNTERS_ZONE_SIZE = (5 + 200) × 4 + 40 = 860` bytes, supporting up to `PERF_COUNTERS_MAX_ZONES = 8` zones per kernel.

The 200-word shared config is the authoritative runtime record of which counters are recorded for every zone (the host reads it back to decode). There is no per-zone configuration — every zone records the same set of counters but stores its own snapshot.

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

- **A pytest invocation compiles one build.** `--enable-perf-counters` selects WC, otherwise NC — a single invocation cannot produce both. The WC build records wall-clock (`ZONE_SCOPED`) *alongside* the counters, so it is self-contained; the NC build is run separately only when a counter-overhead-free timing baseline is wanted. Results merge off-line by `(test_variant, zone)`.
- **The window is `[all threads armed … all threads finished]`.** The `sync_point` barrier arms after every thread has entered and freezes after every thread has finished, so the window brackets the full cross-thread span. The window boundary is the `sync_point` release, which each thread then stamps with its own wall-clock read — those reads serialize on the single shared clock, so the per-thread zone starts differ by ~12–40 cyc (irreducible on BH; not a bug).
- **`PERF_COUNTERS_MAX_ZONES = 8` per kernel.** Adding a 9th distinct `MEASURE_PERF_COUNTERS("...")` name silently reuses zone 0. Reuse the same name across multiple call sites if you want them in the same bucket.
- **L1 mux mutual exclusion is handled inside `MEASURE_PERF_COUNTERS`.** The freeze path re-programs `PERF_CNT_MUX_CTRL` before each L1 slot read, so a single zone snapshot contains counters from multiple mux positions without per-zone configuration changes.
- **BRISC compile flag.** When `--enable-perf-counters` is set, BRISC is rebuilt with `-DPERF_COUNTERS_COMPILED`. Otherwise BRISC does not touch the counter HW at all — this keeps the NC build free of any counter-armed monitoring overhead.
- **Test isolation.** As with every LLK test, counter state at kernel entry is whatever the previous test left behind. The BRISC reset path clears the shared config and zone buffers, so each test starts from a known L1 state, but HW counter registers themselves may carry residual values until the first `MEASURE_PERF_COUNTERS` rising-edge clear.
- **NC/WC bit-identity is fragile.** The goal is that the WC counter code doesn't perturb the measured timing, which requires WC codegen to match NC outside the counter parts. `get_counter_base_addr` uses a `volatile` index cast specifically to stop GCC from emitting a `CSWTCH` jump table (it would shift GP-relative offsets and break that bit-identity), and `freeze_and_read` uses `#pragma GCC unroll 0`. Measured counters are sensitive to BRISC boot *timing* at the ~0.1 % level, so avoid reshaping the BRISC boot path (e.g. the config scan) even when it looks logically equivalent.
- **The BRISC boot arm is redundant but retained.** RTL (`tt_perf_cnt.sv`) confirms a rising-edge start both *clears* and starts the counters, so the per-zone `arm_all_counters` fully resets them from any prior state — the boot-time `arm_hardware()` measures a window nobody reads. It is kept only because removing it changes boot timing (see previous point). The essential BRISC work is `configure_hardware` (period/mode) + the `DBG_FEATURE_DISABLE` scrub.
- **L1 layout must stay below the profiler region.** `PERF_COUNTERS_LAYOUT_END` must not overlap the profiler's lowest L1 address (`llk_profiler::EPOCH_ADDR`). Two `static_assert`s enforce this — a literal one in the always-compiled section (BRISC has no `llk_profiler` namespace) and a symbolic one in the `LLK_PROFILER` section that tracks the profiler layout automatically.
- **Small-count ratio noise.** On tiny workloads (e.g. a single-tile variant, ~400 cyc) the derived L1 *ratio* metrics can swing wildly run-to-run because their denominators are only a few events, so one event of genuine cycle-level L1-arbitration variance is a large percentage. The `mean` timing metric stays stable, and on realistically-sized workloads (e.g. 64 tiles) the ratios are byte-identical run-to-run. Note the residual `L1_TO_L1` run-to-run jitter is addressed by the `sync_point` per-zone re-alignment (which the NC path also uses), not by the WC-only `DBG_FEATURE_DISABLE` scrub.
- **The single-inventory source couples the perf build to two metal headers.** `counters.h` `#include`s `perf_counters.hpp` (`PerfCounterType` enum, reached via `-I…/tools/profiler`) and the arch `hw_counters.h`; the host `counters.py` parses the same `hw_counters.h` at import. This removes the hand-duplicated inventory, at the cost that if those headers move or the enum/array shape changes, the LLK perf build and decoder must follow. The config-word bit layout (`PERF_CFG_*`) and bank-id↔name mapping are still mirrored between `counters.h` and `counters.py` — they are this infra's own L1 ABI, not part of `hw_counters.h`.
