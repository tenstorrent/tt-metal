# LLK Performance Counters Guide

## Quick Links
- Device-side counter HW driver: [tests/helpers/include/counters.h](../../tests/helpers/include/counters.h)
- Test-helper mock functions: [tests/helpers/include/perf.h](../../tests/helpers/include/perf.h)
- Profiler zone macros: [tests/helpers/include/profiler.h](../../tests/helpers/include/profiler.h)
- Cross-thread rendezvous: [tests/helpers/include/barrier.h](../../tests/helpers/include/barrier.h)
- Report schema (must be updated with any metric rename): [tests/python_tests/helpers/perf/schema.py](../../tests/python_tests/helpers/perf/schema.py)
- Host-side counter readback: [tests/python_tests/helpers/counters.py](../../tests/python_tests/helpers/counters.py)
- Host-side derived metrics: [tests/python_tests/helpers/metrics.py](../../tests/python_tests/helpers/metrics.py)
- Test driver: [tests/python_tests/helpers/perf/core.py](../../tests/python_tests/helpers/perf/core.py)
- Test sources: [tests/sources/](../../tests/sources/) (files ending in `_perf.cpp`)
- Pytest CLI registration: [tests/python_tests/conftest.py](../../tests/python_tests/conftest.py)
- Upstream tech report (metal-level): [tech_reports/PerfCounters/perf-counters.md](../../../../tech_reports/PerfCounters/perf-counters.md)

## Overview

This guide documents the LLK test-suite interface for collecting Tensix hardware performance counters. The LLK suite runs bare-metal kernels in `tests/sources/*_perf.cpp` directly on the TRISC cores. There is no firmware or NoC stack involved, so the counter-collection path is implemented entirely inside the test kernels: a C++ macro arms and freezes the hardware counters around a measured scope, writes the snapshot to a fixed L1 buffer, and the host process reads that buffer back from outside the kernel.

Tensix cores contain five hardware performance counter banks. Every bank exposes two values per measurement: total elapsed cycles (`OUT_L`) and an event count for the selected `counter_sel` (`OUT_H`). The counter set is fixed at compile time in `counters.h` and written to L1 by BRISC; the host no longer configures anything, it reads the shared config back to decode the snapshot. The device walks the slots at freeze and the driver pulls the results into a pandas DataFrame and optionally a CSV. The derived metrics (utilisation %, stall %, and per-unit efficiency ratios) are computed in Python on top of the raw counts.

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
| NC (no counters) | defined | undefined | `ZONE_SCOPED` (timing) + `MEASURE_PERF_COUNTERS` (barrier only) | Per-zone wall-clock cycles (`RISCV_DEBUG_REG_WALL_CLOCK_L`), reported differently per run type: `L1_TO_L1` is the unpack-start to pack-end cross-thread span, the isolates are the measured thread's own zone, and `L1_CONGESTION` yields two columns, `[UNPACK]` and `[PACK]` |
| WC (with counters) | defined | defined | `MEASURE_PERF_COUNTERS` **and** `ZONE_SCOPED` | Per-zone HW counter snapshot **and** wall-clock cycles |

`START_PERF_MEASURE(name)` expands to `MEASURE_PERF_COUNTERS(name)` + `ZONE_SCOPED(name)`. In the NC build, `MEASURE_PERF_COUNTERS` performs the same real three-thread rendezvous with an empty action (so it is not free, and it moves the NC baseline); on Quasar it expands to nothing and `ZONE_SCOPED` records the per-zone wall-clock timestamps. In the WC build **both** are live: the counter scope performs the rendezvous *and* arms/freezes the HW counters, while `ZONE_SCOPED` records the per-zone wall-clock timestamps without adding another rendezvous. A single WC run therefore yields both counter and wall-clock data per zone under the same name; the host driver keys everything by `(test_variant, zone)` and can merge NC and WC results (or use the WC wall-clock directly).

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

Each zone is registered once at its first encounter (the counter half is RAII-scoped and assigns a stable zone id by hashing the name), so placing `START_PERF_MEASURE` **outside** the loop is preferred: counter start is not a no-op and would dominate per-iteration cost if done on every tile.

### `PerfRunType` and the single-thread arm/freeze model

Each LLK perf test is associated with a `PerfRunType` (declared in `perf.h`): `L1_TO_L1` runs the full handshaked unpack → math → pack pipeline. `L1_CONGESTION` keeps unpack and pack real but decouples them (math is reduced to a dvalid drain and pack free-runs), so the two hammer L1 concurrently instead of forming a pipeline; `UNPACK_ISOLATE` / `MATH_ISOLATE` / `PACK_ISOLATE` exercise a single stage. The run type selects which threads do real work, whether they are handshaked or free-running, and whether the exit rendezvous exists. The other threads are not idle: they run the minimum dvalid and semaphore mocks in `perf.h` that the measured stage's hardware handshake requires.

**Pack arms the counters for every run type.** `llk_barrier::is_action_thread()` in `barrier.h` returns true only on pack. Freezing is not uniform: see the table below. Arming can be fixed to one thread because:

- The perf counters are **global hardware** driven by shared debug registers (`PERF_CNT_ALL` and the per-bank `*2` command registers), so any RISC can arm/freeze them. The identity of the issuing thread does not change what is counted.
- The entry rendezvous waits for every thread before arming, so the window opens after all of them have finished the previous zone.

Pack is chosen because a sweep found arming there halves the total `L1_CONGESTION` error against arming on TRISC0. Fixing it also matters for its own sake: letting the arming thread vary is how the two builds ended up releasing from different threads.

**Freezing is per run type**, decided by `exit_barrier_for()` in `counters.h`, which defaults to `wants_exit_barrier()` unless `LLK_PERF_EXIT_BARRIER` overrides it:

| Run type | Exit rendezvous | Freezing thread |
|---|---|---|
| `L1_TO_L1`, `L1_CONGESTION` | yes | pack, once all three have arrived |
| `UNPACK_ISOLATE` | yes | pack, once all three have arrived |
| `MATH_ISOLATE`, `PACK_ISOLATE` | no | the measured thread, just after its own `ZONE_END` |

So for `MATH_ISOLATE` the freeze is done by math, with no barrier at all. Note that `START_PERF_MEASURE` declares the counter scope before the profiler zone, so on every run type the counters are armed before `ZONE_START` and frozen after `ZONE_END`: `OUT_L` always exceeds the wall-clock span, by the arrival spread where the exit barrier is kept (largest for `UNPACK_ISOLATE`) and by the peers' epilogue where it is not. Without the exit barrier the other two threads leave the scope while the counters are still armed, so their `tensix_sync`, `KERNEL` `ZONE_END` and completion-mailbox write land inside the measured window. `LLK_PERF_EXIT_BARRIER` overrides the table (default `-1` = per run type) but has no harness plumbing, so it must be passed by hand; a `static_assert` rejects forcing it off for a run type where no thread would freeze.

### The `MEASURE_PERF_COUNTERS` macro

Expands to a `perf_counter_scoped<PERF_RUN_TYPE>` RAII object; the run type is a template parameter because it selects the exit shape above. Its constructor and destructor execute the following sequence (only on the WC build):

1. **Constructor (zone entry).** Calls `llk_barrier::rendezvous(llk_barrier::is_action_thread(), arm_all_counters)`. All three threads rendezvous; the **action thread (pack)** then writes the rising-edge start bit to `PERF_CNT_ALL` (FPU + INSTRN), `PERF_CNT_TDMA_UNPACK2`, `PERF_CNT_L1_2`, and `PERF_CNT_TDMA_PACK2`, which clears all banks and starts the count, and releases the others.

2. **Body.** All three threads run the work inside the scope. Counters tick continuously on the shared backend.

3. **Destructor (zone exit).** For the run types that keep the exit rendezvous, calls `llk_barrier::rendezvous(llk_barrier::is_action_thread(), freeze_and_read_all_counters)`; otherwise the measured thread freezes directly. Note `PROFILER_SYNC()` (`tensix_sync`) is a hand-written statement in each kernel and the isolate and congestion paths can `return` past it, so a thread's backend is not guaranteed drained when the stop bit is written; `fence_compiler()` around the rendezvous is a compiler barrier only. The **action thread (pack)** writes the rising-edge stop bit to the same four registers, then walks the shared 200-word config buffer at `0x169000` and reads every valid slot: for each it programs the bank's mode register with the `counter_sel`, reads `OUT_H` (the event count), and stores it in the per-zone data area (`OUT_L` is read once from the INSTRN_THREAD bank and copied into all five per-bank cycle words, so those five words always hold the same value). It then sets the zone's `SYNC_ZONE_COMPLETE` flag and releases the others.

Each zone gets its own data block in L1 (see [L1 Layout](#l1-layout-and-zone-buffers)) so multiple measurement scopes in the same kernel produce independent snapshots. The device supports `PERF_COUNTERS_MAX_ZONES = 8` zones, but the host names only two: `perf.py` hard-codes zone 0 as `INIT` and zone 1 as `TILE_LOOP`, so a third counter zone appears in the CSV as the literal `ZONE_2` and never joins the wall-clock rows. Zone names in use are `INIT`, `TILE_LOOP`, and `UNINIT` in `fast_tilize_bh` / `fast_untilize` only; `UNINIT` uses bare `ZONE_SCOPED`, so it is timing-only with no rendezvous; identical names share a zone.

#### The `llk_barrier::rendezvous` barrier

The barrier is `llk_barrier::rendezvous` in `barrier.h`, on the `PACK_DONE` **hardware semaphore**, and both builds compile the identical one. It replaced three separate rendezvous, including `llk_profiler::sync_point`, which was an actor-release protocol on an L1 epoch word. `sync_point` was removed because that release gave the actor a head start worth about one cycle per tile on a strict producer/consumer loop, and because the no-counter build could not reach the semaphore version and silently fell back to it, so the two builds measured the same zone with different instruments.

Every thread announces by incrementing; the action thread waits for all of them, runs its action, then drains the count back to zero, and that return to zero is the release. Two consequences worth being explicit about. The action thread spins **twice**, once before the action and once draining after it, so on entry that drain runs inside the counter window. And the barrier is not free: ablation showed the exit rendezvous merely existing was worth +11 of the +12 cycles by which the counter build differed from the no-counter build on `MATH_ISOLATE`. What the shared barrier buys is that both builds pay the same cost at zone entry, not that the cost is zero.

The semaphore is used rather than L1 because its release is symmetric, detection is a short `pc_buf` poll, and it puts no traffic on the L1 being measured. Quasar has no spare semaphore and keeps the L1 form in `profiler.h`.

### Configure-once from BRISC

Before any TRISC kernel runs, BRISC executes `configure_and_arm_from_brisc()` once (called from `brisc.cpp` when the WC build flag is set). This:

- Writes the per-architecture `BUILTIN_COUNTER_CONFIG` (110 slots on WH, 101 on BH, since only one L1 mux group is emitted) into the shared L1 config buffer at `0x169000`. That array is built at compile time from the canonical metal inventory, see [Counter inventory single source](#counter-inventory-single-source).
- Clears every per-zone data area and sync word.
- Clears `DBG_FEATURE_DISABLE` to `0`, see [DBG_FEATURE_DISABLE scrub](#dbg_feature_disable-scrub) below.
- Programs each bank's reference-period and mode registers, sets `PERF_CNT_MUX_CTRL` for L1, and does an initial global arm (later overridden by the first `MEASURE_PERF_COUNTERS` zone).

After BRISC releases the TRISCs, the shared config is read-only for the rest of the run.

##### `DBG_FEATURE_DISABLE` scrub

`DBG_FEATURE_DISABLE` is a 16-bit debug/chicken-bit register whose bits toggle low-level behaviors, notably randomized L1 arbitration (bit 3; the name is from the RTL and is not a symbol in this tree), L1 atomic serialization, and L1 read-enable override. It resets to `0` (all normal), but HW register state **leaks between tests** run back-to-back on an un-reset device, so a prior test that set one of these bits would silently perturb, and make nondeterministic, the L1 counters (16 per run, since one mux group is captured). BRISC writes `0` here to guarantee a clean baseline regardless of leaked state; the blanket write (rather than clearing one bit) is deliberate because any of the bits, not just LFSR, would skew the measurement. Verified: with a leaked `0x8` present, the L1 metrics jitter 40–98 % run-to-run without this scrub and are byte-identical with it. Note this scrub is WC-only (it lives in the counter path); the NC path has no equivalent.

### Reading results from host

After the kernel completes:

1. The host process reads the per-zone data area back from device L1.
2. `read_counters()` decodes each 32-bit config word (bit 31 valid, bits 7:0 bank, bits 16:8 `counter_sel`, bits 19:17 `l1_mux`), looks up the human-readable counter name (parsed at import from the same `hw_counters.h`, see [Counter inventory single source](#counter-inventory-single-source)), and pairs every event count with that zone's bank cycle count.
3. `read_counters()` returns an in-memory long-format DataFrame with columns `zone`, `bank`, `counter_name`, `counter_id`, `cycles`, `count`, `l1_mux`. `compute_metrics()` produces its own separate rows; it does not add columns to that frame. Host-side only zone 0 and zone 1 are named (hard-coded to `INIT` and `TILE_LOOP`); further zones stay `ZONE_n` and never join the wall-clock rows.

Because both wall-clock cycles (NC build, `ZONE_SCOPED` start/end timestamps from `RISCV_DEBUG_REG_WALL_CLOCK_L`) and HW counter cycles (WC build, `OUT_L`) are tagged with the same zone name, the test driver merges them by `(test_variant, zone)`.

## How to Run

The LLK test suite uses a two-phase pytest flow: a compile-producer phase that builds every variant in parallel and a compile-consumer phase that runs them on hardware.

```bash
./setup_testing_env.sh        # installs the SFPI toolchain; run it, do not source it (it calls exit)
cd tt_metal/tt-llk/tests     # LLK_HOME is defaulted by conftest.py; you do not need to set it
export CHIP_ARCH=blackhole   # or wormhole; counters are not compiled on quasar

# Phase 1: build all variants (no HW access)
pytest --compile-producer --enable-perf-counters -n 8 -x ./python_tests/perf_eltwise_binary.py

# Phase 2: run on HW
pytest --compile-consumer --enable-perf-counters -x ./python_tests/perf_eltwise_binary.py
```

Wipe the artefact root (`/tmp/tt-llk-build`, or `$RUNNER_TEMP/tt-llk-build`) when switching between the two builds: the variant hash and the build markers ignore the counter flags, so the ELFs are otherwise reused.

To capture a different L1 mux group, `export LLK_PERF_L1_MUX_GROUP=<0-4>` before **both** phases. It is an environment variable rather than a CLI flag and is compiled into `brisc.elf`, so each group needs its own producer run; the readout checks the group found in L1 against the one requested and fails the run if they disagree, so a stale `brisc.elf` cannot return a self-consistently mislabelled dataset.


The `--enable-perf-counters` flag triggers two things:

1. Test sources are compiled with `-DPERF_COUNTERS_COMPILED` (the WC build). BRISC is compiled with the same flag so it runs `configure_and_arm_from_brisc()` once at startup.
2. The Python driver calls `read_counters()` after every run, writes the derived percentage metrics into the main CSV, and writes raw counts only to `*.counters.csv` under `--dump-perf-counters`.

Without the flag the suite still runs the same sources but builds the NC variant, and only `ZONE_SCOPED` wall-clock data is collected.

### CLI flags

| Flag | Implies `--enable-perf-counters` | Effect |
|------|----------------------------------|--------|
| `--enable-perf-counters` | no | Build the WC variant and collect raw counters per zone |
| `--dump-perf-counters` | yes | Export raw counter values to a separate `<test>.counters.csv` alongside the main results CSV |

`--dump-perf-counters` implicitly enables counter collection; you don't need to specify `--enable-perf-counters` separately.

### Output

For each test variant, the WC build emits:

- One row per zone in `perf_data/<test>/<test>.csv`, with wide `<RUN_TYPE>_<stat>(<metric>)` columns. `<test>.post.csv` is the same data with `TILE_LOOP` wall-clock divided by `loop_factor x tile_cnt`; `INIT` rows and every counter/`OUT_L` column stay absolute, so multiply before comparing the two cycle numbers.
- A `*.counters.csv` file if `--dump-perf-counters` was passed. There is no console dump and no min/median/max aggregation.

The NC build emits per-zone wall-clock cycle counts in the same results DataFrame so a single run with both builds (different pytest invocations) can be merged off-line to compare wall-clock cycles against counter-derived cycle counts.

## Architecture Summary

| | Wormhole | Blackhole |
|---|---|---|
| INSTRN_THREAD slots in inventory | 59 | 59 |
| FPU slots | 3 | 3 |
| TDMA_UNPACK slots | 18 | 18 |
| TDMA_PACK slots | 14 | 5 |
| L1 mux positions (Tensix) | 2 | 6 (one group per build, `LLK_PERF_L1_MUX_GROUP` 0 to 5) |
| L1 slots in inventory | 32 (16 × 2 mux) | 84 (16 × 5 mux + 4 on position 5) |
| Total slots in `BUILTIN_COUNTER_CONFIG` (one L1 mux group) | 110 | 101 |
| Total config words in L1 | 200 (rest are zero-padded) | 200 |

**Wormhole** has `PACK_COUNT = 4` (per-engine packer busy signals are live in RTL), so `TDMA_PACK` exposes counters 11–14 for dest-read availability, 15–18 for per-engine plus aggregate packer busy, and 267–272 for per-engine dest-read grants plus `MATH_NOT_STALLED_DEST_WR_PORT` and `AVAILABLE_MATH`. The L1 mux is 1-bit wide: position 0 covers NoC Ring 0 plus L1 arbitration, position 1 covers NoC Ring 1 plus TDMA-extended signals.

**Blackhole** has `PACK_COUNT = 1`; per-engine packer busy and dest-read signals for engines 1–3 are tied to constants in RTL and are omitted from the inventory. Only counters 11, 18, 267, 271, 272 remain on the `TDMA_PACK` bank. BH compensates with more L1 mux positions (4 extra) which expose additional NoC rings and miscellaneous L1 ports.

**INSTRN_THREAD bank.** Counters 0–8 and 12–23 are per-thread instruction-type availability (CFG/SYNC/THCON/MOVE/FPU/UNPACK/PACK, 3 threads each, 21 slots; the XSEARCH sels 9–11 are tied off in RTL and are not in the inventory). Counters 24–26 are per-thread total stall cycles. The stall-reason layout differs:

- WH: the four shared stall reasons (SRCA/B clear/valid) are replicated per thread in HW but only the first slot of each is enumerated, at sels 27/30/33/36; per-thread stall reasons then occupy counters 39–65.
- BH: shared stall reasons occupy single slots (27–30), per-thread stall reasons occupy 31–57.

Bit-8-extended counters 256/264/272 expose `THREAD_INSTRUCTIONS_{0,1,2}` (one per per-thread instance), and 283 exposes `ANY_THREAD_STALL`.

### Counter inventory single source

The counter id↔name inventory is **defined once**, in metal's canonical `tt_metal/hw/inc/internal/tt-1xx/<arch>/hw_counters.h`, as grouped `{PerfCounterType, id}` arrays per bank (`instrn_counters`, `fpu_counters`, `unpack_counters`, `pack_counters`, `l1_0..5_counters`). Both sides of the perf infra derive from it, so the list is never hand-maintained twice:

- **Device (`counters.h`)** `#include`s `hw_counters.h` (with the `PerfCounterType` enum from `perf_counters.hpp`) and builds `BUILTIN_COUNTER_CONFIG[]` from those arrays at compile time, a `constexpr` concatenation in the fixed bank order the readout expects (INSTRN, FPU, TDMA_UNPACK, TDMA_PACK, then the single selected L1 mux group).

Only **one** L1 mux group is emitted per build, chosen by `LLK_PERF_L1_MUX_GROUP` (default 0). There are only eight physical L1 counters and `PERF_CNT_MUX_CTRL` routes a group of eight client interfaces into them *while they count*, not when they are read, so a run observes exactly one group. Sweep the flag across runs to cover the others; a metric documented below as requiring the mux-1 slot is not obtainable from a default run. The group is a compile-time constant baked into `brisc.elf`, so a sweep must recompile the producer. The readout checks the group decoded from L1 against the requested one and fails if they disagree.
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

The layout is bounded by two `static_assert`s to stay below `0x16AFF0` (the profiler region boundary). Each zone reserves `PERF_COUNTERS_ZONE_SIZE = (5 + 200) × 4 + 40 = 860` bytes, supporting up to `PERF_COUNTERS_MAX_ZONES = 8` zones per kernel.

The 200-word shared config is the authoritative runtime record of which counters are recorded for every zone (the host reads it back to decode). There is no per-zone configuration: every zone records the same set of counters but stores its own snapshot.

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
| `PERF_CNT_MUX_CTRL` | 0x218 | L1 mux selector: bit 4 on Wormhole (`L1_MUX_MASK = 0x1 << 4`), bits 6:4 on Blackhole (`0x7 << 4`) |

### Mode register (`PERF_CNT_*1`)

| Bits | Field | Description |
|------|-------|-------------|
| 7:0 | mode | 0 = continuous with cycle tracking; 1 = stop after `PERF_CNT_*0` cycles; 2 = continuous without cycle tracking |
| 16:8 | counter_sel | Selects which counter event is routed to `OUT_H` |
| 31:17 | reserved | unused |

The macro path always uses mode 0. Mode 1 is unused in the LLK test suite. The `counter_sel` field is rewritten on each slot read so a single bank can multiplex multiple counters into one measurement window.

### Start/Stop register (`PERF_CNT_*2`)

Rising-edge triggered. Bit 0 = start (0→1 also clears the counter), bit 1 = stop. The per-zone path writes `1` to arm and `2` to freeze, so each write is itself a rising edge on the opposite bit and the stop write is what re-creates the next 0→1 edge. Only BRISC's boot `arm_hardware()` writes `1` then `0`.

### L1 mux (`PERF_CNT_MUX_CTRL`)

Each L1 mux group exposes 8 client interfaces x 2 counters (request sels 0–7 and grant sels 256–263), so 16 `counter_sel` values per group, giving the 32 (WH, 2 groups) and 84 (BH, 6 groups) inventory totals above. The mux field selects the group: bit 4 on Wormhole, bits 6:4 on Blackhole (6 of 8 encodings populated):

| Mux | WH meaning | BH meaning |
|-----|------------|------------|
| 0 | unpacker 0, packer port 1 (+ECC), TDMA bundles 0/1, NoC Ring 0 | unpacker 0, port 1 (unpacker 1 + ECC), TDMA bundles 0/1, NoC Ring 0 |
| 1 | TDMA packer 2, ext unpackers 1–3, NoC Ring 1 | packer interface 0 (port 8), unpacker 1 extended interfaces 1-3, NoC Ring 1 |
| 2 | not present | unpacker 1 extended interfaces 4-7, NoC Ring 0 secondary channels |
| 3 | not present | NoC Ring 1 secondary channels, ext packers 2–5 |
| 4 | not present | ext packers 6-7, packer interface 1 (with the tag-search accelerator), unpacker 0 extended interfaces 1-5 |
| 5 | not present | unpacker 0 extended interfaces 6-7 (only slots 0 and 1 are wired; slots 2-7 read 0) |

The Blackhole column is taken from the A0 tapeout RTL (ws-tensix `BH_A0_RC6`, `tt_tensix.sv`) and was confirmed on silicon by reading every selector under real workloads; positions 6 and 7 have no decode case and fall back to position 0.

The mux routes interfaces into the counters while they count and is written once by BRISC before arming, so the freeze path cannot re-aim it. A zone snapshot therefore contains exactly one mux position: the group that was selected while the counters ran. Sweep it by exporting `LLK_PERF_L1_MUX_GROUP` before the producer phase (it is an environment variable, not a CLI flag, and is baked in at compile time, so each value needs its own `--compile-producer`). Sweep across runs to cover the other groups.

## Derived Metrics Reference

The harness computes every metric of the shared engine, [tools/tracy/perf_metrics_common.py](../../../../tools/tracy/perf_metrics_common.py), once per zone and run from the raw counters; `helpers/metrics.py` only adapts the counter frame to the engine's view. The catalogue with every key, label and formula is [tech_reports/PerfCounters/perf-counters.md](../../../../tech_reports/PerfCounters/perf-counters.md); the same keys appear here as `<RUN_TYPE>_<stat>(<key>)` columns and `perf/schema.py::METRIC_BASES` mirrors them, so a metric added to the engine shows up in the report without harness changes.

A metric whose counters this architecture does not expose, or whose counter group was not captured in the run, is empty in the CSV, never 0. Counters that exist on Wormhole only (per-engine packer busy) or Blackhole only (L1 banks 2-5) make their metrics empty on the other chip.

---

## Notes and Caveats

- **Four tests sit on a bistable L1 operating point.** `matmul`, `math_matmul`, `pack_dest_bank` and
  `pack_untilize` each land on one of two discrete levels, chosen deterministically per binary and re-rolled
  by any change to code layout. Worst observed spread is about 31 percent on `pack_dest_bank`. This is build
  sensitivity, not counter cost: the same two levels appear when one no-counter build is compared against
  another that differs only in layout. If you see a large outlier on those four, check which level both
  sides landed on before treating it as a regression. Tracked in #51901, #51902 and #51904.

- **Never guard `PERF_RUN_TYPE` with `#ifndef`.** It arrives as a `constexpr` in the generated `build.h`, so a preprocessor guard cannot see it, always fires, and silently compiles every run type as the fallback. Identical values and identical `TEXT_SIZE` across run types is the symptom (PR #51918).
- **`L1_CONGESTION` is not free-running everywhere yet.** `eltwise_binary_sfpu`, `eltwise_unary_sfpu`, `eltwise_unary_typecast`, `sfpu_binop_scalar` and `sfpu_ternary` still run its pack path on the math handshake, so they measure the handshake rather than L1 contention.
- **`PACK_DONE` is reserved by the barrier.** It is safe only because the count returns to zero each time, so a measured kernel must not use `semaphore::PACK_DONE` for its own handshake.
- **The host asserts zones do not overlap across threads.** No thread may open `TILE_LOOP` before every thread has closed `INIT`. A failure almost always means a kernel used `ZONE_SCOPED` instead of `START_PERF_MEASURE`, which is what supplies the entry rendezvous. Skipped on Quasar.
- **Both builds write the same report path.** `perf_data/<module>/<module>.csv` is written by whichever invocation ran last, so move the first report aside before running the second build; there is no cross-build merge in code.
- **`--logging-level DEBUG` or `TRACE` recompiles the measured kernel** with `-DDEBUG_PRINT_ENABLED`, which perturbs the numbers. Do not use it for a measurement run.
- **`no zone returned counter data` means the test was never measured.** The counter and metric columns will be absent; a with-counters versus no-counters comparison for that test is meaningless, not zero.
- **Instrumenting a kernel moves its no-counter baseline.** The rendezvous is real in both builds, so converting a kernel from `ZONE_SCOPED` to `START_PERF_MEASURE` shifts its timings; numbers from before and after are not comparable.
- **`PROFILER_SYNC()` is per-kernel and not universal.** `fast_tilize_bh_test.cpp` omits it entirely and no `UNINIT` zone has one, so those windows close without draining the backend.

- **A pytest invocation compiles one build.** `--enable-perf-counters` selects WC, otherwise NC, so a single invocation cannot produce both. The WC build records wall-clock (`ZONE_SCOPED`) *alongside* the counters, so it is self-contained; the NC build is run separately only when a counter-overhead-free timing baseline is wanted. Results merge off-line by `(test_variant, zone)`.
- **The window is `[all threads armed … all threads finished]`.** The rendezvous arms after every thread has entered, and freezes after every thread has finished for the run types that keep the exit barrier. Each thread stamps the release with its own wall-clock read, and those reads serialize on the single shared clock, so the per-thread zone starts differ by ~12–40 cyc (irreducible; not a bug).
- **`PERF_COUNTERS_MAX_ZONES = 8` per kernel.** Adding a 9th distinct `MEASURE_PERF_COUNTERS("...")` name silently reuses zone 0. Reuse the same name across multiple call sites if you want them in the same bucket.
- **One L1 mux group per run.** `PERF_CNT_MUX_CTRL` selects the group while the counters count, not when they are read, so the freeze path cannot re-aim it and a run observes exactly one group. Select it with `LLK_PERF_L1_MUX_GROUP` and sweep it across runs.
- **BRISC compile flag.** When `--enable-perf-counters` is set, BRISC is rebuilt with `-DPERF_COUNTERS_COMPILED`. Otherwise BRISC does not touch the counter HW at all, which keeps the NC build free of any counter-armed monitoring overhead.
- **Test isolation.** As with every LLK test, counter state at kernel entry is whatever the previous test left behind. The BRISC reset path clears the shared config and zone buffers, so each test starts from a known L1 state, but HW counter registers themselves may carry residual values until the first `MEASURE_PERF_COUNTERS` rising-edge clear.
- **NC/WC bit-identity is fragile.** The goal is that the WC counter code doesn't perturb the measured timing, which requires WC codegen to match NC outside the counter parts. `get_counter_base_addr` uses a `volatile` index cast specifically to stop GCC from emitting a `CSWTCH` jump table (it would shift GP-relative offsets and break that bit-identity), and `freeze_and_read_all_counters` uses `#pragma GCC unroll 0`. Measured counters are sensitive to BRISC boot *timing* at the ~0.1 % level, so avoid reshaping the BRISC boot path (e.g. the config scan) even when it looks logically equivalent.
- **The BRISC boot arm is redundant but retained.** The RTL (see `tech_reports/PerfCounters/perf-counters.md`) confirms a rising-edge start both *clears* and starts the counters, so the per-zone `arm_all_counters` fully resets them from any prior state, so the boot-time `arm_hardware()` measures a window nobody reads. It is kept only because removing it changes boot timing (see previous point). The essential BRISC work is `configure_hardware` (period/mode) + the `DBG_FEATURE_DISABLE` scrub.
- **L1 layout must stay below the profiler region.** `PERF_COUNTERS_LAYOUT_END` must not overlap the profiler's lowest L1 address (`llk_profiler::EPOCH_ADDR`). Two `static_assert`s enforce this: a literal one in the always-compiled section (BRISC has no `llk_profiler` namespace) and a symbolic one in the `LLK_PROFILER` section that tracks the profiler layout automatically.
- **Minimum window size.** Size every measured window above ~1k cycles using the test's `LOOP_FACTOR`; PR #51912 raised the suite's factors for exactly this reason. Below that, a few cycles of instrument floor read as a large percentage, and the timing `mean` is affected as well as the derived ratios. Note the report divides `TILE_LOOP` wall-clock by `loop_factor x tile_cnt` but leaves `INIT` and every counter column absolute.

- **The single-inventory source couples the perf build to two metal headers.** `counters.h` `#include`s `perf_counters.hpp` (`PerfCounterType` enum, reached via `-I…/tools/profiler`) and the arch `hw_counters.h`; the host `counters.py` parses the same `hw_counters.h` at import. This removes the hand-duplicated inventory, at the cost that if those headers move or the enum/array shape changes, the LLK perf build and decoder must follow. The config-word bit layout (`PERF_CFG_*`) and bank-id↔name mapping are still mirrored between `counters.h` and `counters.py`. They are this infra's own L1 ABI, not part of `hw_counters.h`.
