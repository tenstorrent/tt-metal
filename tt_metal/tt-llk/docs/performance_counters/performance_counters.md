# LLK Performance Counters Guide

## Quick Links
- Device-side counter HW driver: [tests/helpers/include/counters.h](../../tests/helpers/include/counters.h)
- Test-helper mock functions: [tests/helpers/include/perf.h](../../tests/helpers/include/perf.h)
- Profiler zone macros: [tests/helpers/include/profiler.h](../../tests/helpers/include/profiler.h)
- Cross-thread rendezvous: [tests/helpers/include/barrier.h](../../tests/helpers/include/barrier.h)
- Quasar SFPU stub (fourth thread of the rendezvous): [tests/helpers/include/sfpu_stub.h](../../tests/helpers/include/sfpu_stub.h)
- Hardware reference (inventory, registers, limitations, metric catalogue): [hardware_reference.md](hardware_reference.md)
- Shared counter definitions (names, select tables, registers, primitives): [tools/include/perf_counters/](../../tools/include/perf_counters/)
- Shared host package (header parsers, metric engine): [tools/python/tt_llk_perf/](../../tools/python/tt_llk_perf/)
- Report schema (metric columns derive from the shared engine): [tests/python_tests/helpers/perf/schema.py](../../tests/python_tests/helpers/perf/schema.py)
- Host-side counter readback: [tests/python_tests/helpers/counters.py](../../tests/python_tests/helpers/counters.py)
- Host-side derived metrics adapter: [tests/python_tests/helpers/metrics.py](../../tests/python_tests/helpers/metrics.py)
- Test driver: [tests/python_tests/helpers/perf/core.py](../../tests/python_tests/helpers/perf/core.py)
- Test sources: [tests/sources/](../../tests/sources/) (files ending in `_perf.cpp`; Quasar kernels are under [tests/sources/quasar/](../../tests/sources/quasar/))
- Pytest CLI registration: [tests/python_tests/conftest.py](../../tests/python_tests/conftest.py)
- Upstream tech report (metal-level): [tech_reports/PerfCounters/perf-counters.md](../../../../tech_reports/PerfCounters/perf-counters.md)

## Overview

This guide documents the LLK test-suite interface for collecting Tensix hardware performance counters. The LLK suite runs bare-metal kernels in `tests/sources/*_perf.cpp` directly on the TRISC cores. There is no firmware or NoC stack involved, so the counter-collection path is implemented entirely inside the test kernels: a C++ macro arms and freezes the hardware counters around a measured scope, writes the snapshot to a fixed L1 buffer, and the host process reads that buffer back from outside the kernel.

Tensix cores contain five hardware performance counter banks (four on Quasar, which has no L1 bank; every NEO has its own set). Every bank exposes two values per measurement: total elapsed cycles (`OUT_L`) and an event count for the selected `counter_sel` (`OUT_H`). The counter set is fixed at compile time in `counters.h` and written to L1 by BRISC (by the unpack TRISC on Quasar); the host no longer configures anything, it reads the shared config back to decode the snapshot. The device walks the slots at freeze and the driver pulls the results into a pandas DataFrame and optionally a CSV. The derived metrics (utilisation %, stall %, and per-unit efficiency ratios) are computed in Python on top of the raw counts.

| Bank | Description |
|------|-------------|
| INSTRN_THREAD | Per-thread instruction issue counts, availability, and stall reasons |
| FPU | FPU and SFPU active cycles |
| TDMA_UNPACK | Unpacker busy signals, math pipeline status, srcA/B write port and overwrite signals |
| TDMA_PACK | Packer busy, dest-read availability, math availability |
| L1 | NoC ring transactions and L1 port arbitration (mux-selected). Absent on Quasar, where bank slot 3 carries the one l1_client CSR selection instead |

## How It Works

### Two builds, one test source

Every test source under `tests/sources/*_perf.cpp` is compiled twice from the same C++ file. The build is selected by two preprocessor flags:

| Build | `LLK_PROFILER` | `PERF_COUNTERS_COMPILED` | Active half(s) of `START_PERF_MEASURE` | What it measures |
|-------|----------------|--------------------------|--------------------------------------|------------------|
| NC (no counters) | defined | undefined | `ZONE_SCOPED` (timing) + `MEASURE_PERF_COUNTERS` (barrier only) | Per-zone wall-clock cycles (`RISCV_DEBUG_REG_WALL_CLOCK_L`), reported differently per run type: `L1_TO_L1` is the unpack-start to pack-end cross-thread span, the isolates are the measured thread's own zone, and `L1_CONGESTION` yields two columns, `[UNPACK]` and `[PACK]` |
| WC (with counters) | defined | defined | `MEASURE_PERF_COUNTERS` **and** `ZONE_SCOPED` | Per-zone HW counter snapshot **and** wall-clock cycles |

`START_PERF_MEASURE(name)` expands to `MEASURE_PERF_COUNTERS(name)` + `ZONE_SCOPED(name)`. In the NC build, `MEASURE_PERF_COUNTERS` performs the same real rendezvous with an empty action (so it is not free, and it moves the NC baseline). The rendezvous has three threads on Wormhole and Blackhole and four on Quasar, where the SFPU TRISC takes part too: `sfpu_stub.h` opens the same `INIT` and `TILE_LOOP` zones, so a kernel with no SFPU half still completes it. In the WC build **both** are live: the counter scope performs the rendezvous *and* arms/freezes the HW counters, while `ZONE_SCOPED` records the per-zone wall-clock timestamps without adding another rendezvous. A single WC run therefore yields both counter and wall-clock data per zone under the same name; the host driver keys everything by `(test_variant, zone)` and can merge NC and WC results (or use the WC wall-clock directly).

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
| `SFPU_ISOLATE` (Quasar) | no | the SFPU TRISC (`LLK_TRISC_ISOLATE_SFPU`), just after its own `ZONE_END` |

So for `MATH_ISOLATE` the freeze is done by math, with no barrier at all. Note that `START_PERF_MEASURE` declares the counter scope before the profiler zone, so on every run type the counters are armed before `ZONE_START` and frozen after `ZONE_END`: `OUT_L` always exceeds the wall-clock span, by the arrival spread where the exit barrier is kept (largest for `UNPACK_ISOLATE`) and by the peers' epilogue where it is not. Without the exit barrier the other two threads leave the scope while the counters are still armed, so their `tensix_sync`, `KERNEL` `ZONE_END` and completion-mailbox write land inside the measured window. `LLK_PERF_EXIT_BARRIER` overrides the table (default `-1` = per run type) but has no harness plumbing, so it must be passed by hand; a `static_assert` rejects forcing it off for a run type where no thread would freeze.

### The `MEASURE_PERF_COUNTERS` macro

Expands to a `perf_counter_scoped<PERF_RUN_TYPE>` RAII object; the run type is a template parameter because it selects the exit shape above. Its constructor and destructor execute the following sequence (only on the WC build):

1. **Constructor (zone entry).** Calls `llk_barrier::rendezvous(llk_barrier::is_action_thread(), arm_all_counters)`. Every thread rendezvous; the **action thread (pack)** then writes the rising-edge start bit to `PERF_CNT_ALL` (FPU + INSTRN), `PERF_CNT_TDMA_UNPACK2`, `PERF_CNT_L1_2`, and `PERF_CNT_TDMA_PACK2`, which clears all banks and starts the count, and releases the others. On Quasar there is no L1 bank to pulse; if an l1_client selection is compiled in, its clear-on-read `CNT` register is read once at that point instead, so the window starts at zero.

2. **Body.** Every thread runs the work inside the scope. Counters tick continuously on the shared backend.

3. **Destructor (zone exit).** For the run types that keep the exit rendezvous, calls `llk_barrier::rendezvous(llk_barrier::is_action_thread(), freeze_and_read_all_counters)`; otherwise the measured thread freezes directly. Note `PROFILER_SYNC()` (`tensix_sync`) is a hand-written statement in each kernel and the isolate and congestion paths can `return` past it, so a thread's backend is not guaranteed drained when the stop bit is written; `fence_compiler()` around the rendezvous is a compiler barrier only. The **action thread (pack)** writes the rising-edge stop bit to the same four registers, then walks the shared 200-word config buffer at `0x169000` and reads every valid slot: for each it programs the bank's mode register with the `counter_sel`, reads `OUT_H` (the event count), and stores it in the per-zone data area (`OUT_L` is read once from the INSTRN_THREAD bank and copied into all five per-bank cycle words, so those five words always hold the same value). On Quasar slot 3 is the l1_client selection: its value is `l1_client_read()` of the clear-on-read `CNT` register and its cycles are the INSTRN bank's. It then sets the zone's `SYNC_ZONE_COMPLETE` flag and releases the others.

Each zone gets its own data block in L1 (see [L1 Layout](#l1-layout-and-zone-buffers)) so multiple measurement scopes in the same kernel produce independent snapshots. The device supports `PERF_COUNTERS_MAX_ZONES = 8` zones, but the host names only two: `perf.py` hard-codes zone 0 as `INIT` and zone 1 as `TILE_LOOP`, so a third counter zone appears in the CSV as the literal `ZONE_2` and never joins the wall-clock rows. Zone names in use are `INIT`, `TILE_LOOP`, and `UNINIT` in `fast_tilize_bh` / `fast_untilize` only; `UNINIT` uses bare `ZONE_SCOPED`, so it is timing-only with no rendezvous; identical names share a zone.

The counters are driven through the registers described in the [hardware reference](hardware_reference.md#hardware-register-reference). The macro path always uses mode 0. Mode 1 is unused in the LLK test suite. The `counter_sel` field is rewritten on each slot read so a single bank can multiplex multiple counters into one measurement window. The per-zone path writes `1` to arm and `2` to freeze, so each write is itself a rising edge on the opposite bit and the stop write is what re-creates the next 0→1 edge. Only BRISC's boot `arm_hardware()` writes `1` then `0`.

#### The `llk_barrier::rendezvous` barrier

The barrier is `llk_barrier::rendezvous` in `barrier.h`, on the `PACK_DONE` **hardware semaphore**, and both builds compile the identical one. It replaced three separate rendezvous, including `llk_profiler::sync_point`, which was an actor-release protocol on an L1 epoch word. `sync_point` was removed because that release gave the actor a head start worth about one cycle per tile on a strict producer/consumer loop, and because the no-counter build could not reach the semaphore version and silently fell back to it, so the two builds measured the same zone with different instruments.

Every thread announces by incrementing; the action thread waits for all of them, runs its action, then drains the count back to zero, and that return to zero is the release. Two consequences worth being explicit about. The action thread spins **twice**, once before the action and once draining after it, so on entry that drain runs inside the counter window. And the barrier is not free: ablation showed the exit rendezvous merely existing was worth +11 of the +12 cycles by which the counter build differed from the no-counter build on `MATH_ISOLATE`. What the shared barrier buys is that both builds pay the same cost at zone entry, not that the cost is zero.

The semaphore is used rather than L1 because its release is symmetric, detection is a short `pc_buf` poll, and it puts no traffic on the L1 being measured. Quasar uses the same implementation, only with a different pair of indices: it has 32 Tensix semaphores (`tt_tensix_pkg.sv` `SEM_COUNT`, reachable from a TRISC as four banks of eight through the PC buffer at words 32 to 63, each resetting to 0 with max 15) and every Quasar LLK op stays inside bank 0, so the barrier claims 8 and 9 in bank 1 where nothing can collide with live dataflow.

Quasar reached that through an L1 generation barrier first, and the measurements are worth keeping: with waiters polling an L1 word and invalidating their data cache on every poll, the emulator showed the unpack windows up to 5.4% longer and `L1_CONGESTION` up to 5.0% longer than the no-counter build. Spacing the polls with 32 nops brought that into the noise but left a bias on `L1_TO_L1` (median +0.48%, worst 120 cycles over 12 variants), because Quasar has a fourth thread: the SFPU stub opens both zones and then waits for the whole pipeline window. On semaphores the same 12 variants give a median of -0.05% and a worst case of 21 cycles on that run type, the nop knob is gone, and both architectures compile the same barrier body.

### Configure-once from BRISC

Before any TRISC kernel runs, BRISC executes `configure_and_arm_from_brisc()` once (called from `brisc.cpp` when the WC build flag is set). Quasar has no BRISC in this harness, so the unpack TRISC calls the same `configure_and_arm()` from `trisc.cpp`, after `device_setup()` and before it clears the other TRISCs out of soft reset; the registers are reached through the NEO local window (`bank_regs(Bank)` defaults to `LOCAL_REGS_WINDOW` there) instead of the RISC-V debug block. This:

- Writes the per-architecture `BUILTIN_COUNTER_CONFIG` (110 slots on WH, 101 on BH, since only one L1 mux group is emitted; 77 on Quasar, 78 with an l1_client selection) into the shared L1 config buffer at `0x169000`. That array is built at compile time from the shared select tables, see [Shared counter definitions](hardware_reference.md#where-the-definitions-live).
- Clears every per-zone data area and sync word.
- Clears `DBG_FEATURE_DISABLE` to `0`, see [DBG_FEATURE_DISABLE scrub](#dbg_feature_disable-scrub) below.
- Programs each bank's reference-period and mode registers, sets `PERF_CNT_MUX_CTRL` for L1, and does an initial global arm (later overridden by the first `MEASURE_PERF_COUNTERS` zone).

After BRISC releases the TRISCs, the shared config is read-only for the rest of the run.

##### `DBG_FEATURE_DISABLE` scrub

`DBG_FEATURE_DISABLE` is a 16-bit debug/chicken-bit register whose bits toggle low-level behaviors, notably randomized L1 arbitration (bit 3; the name is from the RTL and is not a symbol in this tree), L1 atomic serialization, and L1 read-enable override. It resets to `0` (all normal), but HW register state **leaks between tests** run back-to-back on an un-reset device, so a prior test that set one of these bits would silently perturb, and make nondeterministic, the L1 counters (16 per run, since one mux group is captured). BRISC writes `0` here to guarantee a clean baseline regardless of leaked state; the blanket write (rather than clearing one bit) is deliberate because any of the bits, not just LFSR, would skew the measurement. Verified: with a leaked `0x8` present, the L1 metrics jitter 40–98 % run-to-run without this scrub and are byte-identical with it. Note this scrub is WC-only (it lives in the counter path); the NC path has no equivalent.

### Reading results from host

After the kernel completes:

1. The host process reads the per-zone data area back from device L1.
2. `read_counters()` decodes each 32-bit config word (bit 31 valid, bits 7:0 bank, bits 16:8 `counter_sel`, bits 19:17 `l1_mux`), looks up the human-readable counter name (parsed at import from the same arch header by `tt_llk_perf.headers.bank_tables()`, see [Shared counter definitions](hardware_reference.md#where-the-definitions-live)), and pairs every event count with that zone's bank cycle count.
3. `read_counters()` returns an in-memory long-format DataFrame with columns `zone`, `bank`, `counter_name`, `counter_id`, `cycles`, `count`, `l1_mux`. `compute_metrics()` produces its own separate rows; it does not add columns to that frame. Host-side only zone 0 and zone 1 are named (hard-coded to `INIT` and `TILE_LOOP`); further zones stay `ZONE_n` and never join the wall-clock rows.

Because both wall-clock cycles (NC build, `ZONE_SCOPED` start/end timestamps from `RISCV_DEBUG_REG_WALL_CLOCK_L`) and HW counter cycles (WC build, `OUT_L`) are tagged with the same zone name, the test driver merges them by `(test_variant, zone)`.

## How to Run

The LLK test suite uses a two-phase pytest flow: a compile-producer phase that builds every variant in parallel and a compile-consumer phase that runs them on hardware.

```bash
./setup_testing_env.sh        # installs the SFPI toolchain; run it, do not source it (it calls exit)
cd tt_metal/tt-llk/tests     # LLK_HOME is defaulted by conftest.py; you do not need to set it
export CHIP_ARCH=blackhole   # or wormhole or quasar

# Phase 1: build all variants (no HW access)
pytest --compile-producer --enable-perf-counters -n 8 -x ./python_tests/perf_eltwise_binary.py

# Phase 2: run on HW
pytest --compile-consumer --enable-perf-counters -x ./python_tests/perf_eltwise_binary.py
```

Wipe the artefact root (`/tmp/tt-llk-build`, or `$RUNNER_TEMP/tt-llk-build`) when switching between the two builds: the variant hash and the build markers ignore the counter flags, so the ELFs are otherwise reused.

To capture a different L1 mux group, `export LLK_PERF_L1_MUX_GROUP=<0-5>` before **both** phases. It is an environment variable rather than a CLI flag and is compiled into `brisc.elf`, so each group needs its own producer run; the readout checks the group found in L1 against the one requested and fails the run if they disagree, so a stale `brisc.elf` cannot return a self-consistently mislabelled dataset.

Quasar has no L1 counter bank. Its bank slot 3 can carry one event of the L1 client CSR instead: `export LLK_PERF_L1_CLIENT_SEL=<subport*8+event>` before the producer phase (default off; it is compiled into the four TRISC ELFs, so each selection needs its own producer run). Sub-ports are 0-3 TRISC, 4 THCON, 5-24 unpacker reads, 25-36 packer writes; events 1-7 are named by `tt_llk_perf.metrics.quasar_l1_client_label()`, event 0 is unused and THCON events 1-3 are rejected at compile time (`llk::perf::l1_client_selection_is_valid`) because they alias the TRISC port. The counter is clear-on-read and has no reference counter of its own, so it is referenced to the INSTRN bank's cycles, and its metric column is named after the selection (`l1_client_<port>_<event>_pct`, or `_ratio` for the pending-request carry; see the tech report's "L1 client events (Quasar)").

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

## Hardware reference

The counter inventory per architecture, where the shared definitions live and how to add a counter, the debug registers, the L1 mux groups and the hardware limitations are in [hardware_reference.md](hardware_reference.md). On top of that the harness emits one L1 mux group per build, so `BUILTIN_COUNTER_CONFIG` holds 110 slots on Wormhole, 101 slots on Blackhole, 77 (78 with an l1_client selection) slots on Quasar, written into the 200 config words in L1 with the rest zero.

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

## Derived Metrics Reference

The harness computes every metric of the shared engine, [tools/python/tt_llk_perf/metrics.py](../../tools/python/tt_llk_perf/metrics.py), once per zone and run from the raw counters; `helpers/metrics.py` only adapts the counter frame to the engine's `CounterView`. The catalogue with every key, label and formula is [hardware_reference.md](hardware_reference.md#derived-metrics-reference); the same keys appear here as `<RUN_TYPE>_<stat>(<key>)` columns and `perf/schema.py::METRIC_BASES` mirrors them, so a metric added to the engine shows up in the report without harness changes.

A metric whose counters this architecture does not expose, or whose counter group was not captured in the run, is empty in the CSV, never 0. Counters that exist on Wormhole only (per-engine packer busy), Blackhole only (L1 banks 2-5) or Quasar only (thread 3, the INSTISSUE class, the thread-ORed stall reasons, unpacker 2) make their metrics empty on the other chips; every L1 and NoC metric is empty on Quasar, whose only L1 measurement is the dynamic `l1_client_*` column (`helpers/metrics.py` appends it through `compute_l1_client_metrics`).

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
- **The host asserts zones do not overlap across threads.** No thread may open `TILE_LOOP` before every thread has closed `INIT`. A failure almost always means a kernel used `ZONE_SCOPED` instead of `START_PERF_MEASURE`, which is what supplies the entry rendezvous. A hang (`TENSIX TIMED OUT`) in every build means a thread skipped a zone: every thread must open every zone in every configuration, so a compile-time condition on a thread's work belongs inside the zone, not around it (the Quasar unpack-to-dest variants of the pack and unpack tests hung this way until their math zones were opened unconditionally).
- **Both builds write the same report path.** `perf_data/<module>/<module>.csv` is written by whichever invocation ran last, so move the first report aside before running the second build; there is no cross-build merge in code.
- **`--logging-level DEBUG` or `TRACE` recompiles the measured kernel** with `-DDEBUG_PRINT_ENABLED`, which perturbs the numbers. Do not use it for a measurement run.
- **`no zone returned counter data` means the test was never measured.** The counter and metric columns will be absent; a with-counters versus no-counters comparison for that test is meaningless, not zero.
- **Instrumenting a kernel moves its no-counter baseline.** The rendezvous is real in both builds, so converting a kernel from `ZONE_SCOPED` to `START_PERF_MEASURE` shifts its timings; numbers from before and after are not comparable.
- **`PROFILER_SYNC()` is per-kernel and not universal.** `fast_tilize_bh_test.cpp` omits it entirely and no `UNINIT` zone has one, so those windows close without draining the backend.

- **A pytest invocation compiles one build.** `--enable-perf-counters` selects WC, otherwise NC, so a single invocation cannot produce both. The WC build records wall-clock (`ZONE_SCOPED`) *alongside* the counters, so it is self-contained; the NC build is run separately only when a counter-overhead-free timing baseline is wanted. Results merge off-line by `(test_variant, zone)`.
- **The window is `[all threads armed … all threads finished]`.** The rendezvous arms after every thread has entered, and freezes after every thread has finished for the run types that keep the exit barrier. Each thread stamps the release with its own wall-clock read, and those reads serialize on the single shared clock, so the per-thread zone starts differ by ~12–40 cyc (irreducible; not a bug).
- **`PERF_COUNTERS_MAX_ZONES = 8` per kernel.** Adding a 9th distinct `MEASURE_PERF_COUNTERS("...")` name silently reuses zone 0. Reuse the same name across multiple call sites if you want them in the same bucket.
- **One L1 mux group per run.** `PERF_CNT_MUX_CTRL` selects the group while the counters count, not when they are read, so the freeze path cannot re-aim it and a run observes exactly one group. Select it with `LLK_PERF_L1_MUX_GROUP` and sweep it across runs.
- **BRISC compile flag.** When `--enable-perf-counters` is set, BRISC is rebuilt with `-DPERF_COUNTERS_COMPILED` (plus `-DLLK_PERF_L1_MUX_GROUP=<n>`; on Quasar the four TRISCs get `-DPERF_COUNTERS_COMPILED -DLLK_PERF_L1_CLIENT_SEL=<n>` instead and the unpack TRISC does the setup). Otherwise BRISC does not touch the counter HW at all, which keeps the NC build free of any counter-armed monitoring overhead.
- **Test isolation.** As with every LLK test, counter state at kernel entry is whatever the previous test left behind. The BRISC reset path clears the shared config and zone buffers, so each test starts from a known L1 state, but HW counter registers themselves may carry residual values until the first `MEASURE_PERF_COUNTERS` rising-edge clear.
- **NC/WC bit-identity is fragile.** The goal is that the WC counter code doesn't perturb the measured timing, which requires WC codegen to match NC outside the counter parts. `get_bank_regs` uses a `volatile` index cast specifically to stop GCC from emitting a `CSWTCH` jump table (it would shift GP-relative offsets and break that bit-identity), and `freeze_and_read_all_counters` uses `#pragma GCC unroll 0`. Measured counters are sensitive to BRISC boot *timing* at the ~0.1 % level, so avoid reshaping the BRISC boot path (e.g. the config scan) even when it looks logically equivalent.
- **The BRISC boot arm is redundant but retained.** The RTL (see [hardware_reference.md](hardware_reference.md#hardware-register-reference)) confirms a rising-edge start both *clears* and starts the counters, so the per-zone `arm_all_counters` fully resets them from any prior state, so the boot-time `arm_hardware()` measures a window nobody reads. It is kept only because removing it changes boot timing (see previous point). The essential BRISC work is `configure_hardware` (period/mode) + the `DBG_FEATURE_DISABLE` scrub.
- **L1 layout must stay below the profiler region.** `PERF_COUNTERS_LAYOUT_END` must not overlap the profiler's lowest L1 address (`llk_profiler::EPOCH_ADDR`). Two `static_assert`s enforce this: a literal one in the always-compiled section (BRISC has no `llk_profiler` namespace) and a symbolic one in the `LLK_PROFILER` section that tracks the profiler layout automatically.
- **Minimum window size.** Size every measured window above ~1k cycles using the test's `LOOP_FACTOR`; PR #51912 raised the suite's factors for exactly this reason. Below that, a few cycles of instrument floor read as a large percentage, and the timing `mean` is affected as well as the derived ratios. Note the report divides `TILE_LOOP` wall-clock by `loop_factor x tile_cnt` but leaves `INIT` and every counter column absolute.

- **The inventory is shared with metal.** `counters.h` includes `tools/include/perf_counters/{inventory,registers,hw}.h` and `counters.py` parses the same headers through `tt_llk_perf.headers`. The Python parser matches one `{PerfCounterType::NAME, select}` entry per line and `<name>_counters [ATTRIBUTE] =` array names (the optional upper-case token is `quasar.h`'s `LLK_PERF_TABLE_SECTION`), so keep that literal form when editing the tables. The config-word bit layout (`PERF_CFG_*`) and the bank-id to name mapping are still mirrored between `counters.h` and `counters.py`; they are this infra's own L1 ABI, not part of the shared headers.
