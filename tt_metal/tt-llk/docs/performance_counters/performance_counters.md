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

Tensix cores contain five hardware performance counter banks (four on Quasar, which has no L1 bank; every NEO has its own set). Every bank exposes two values per measurement: total elapsed cycles (`OUT_L`) and an event count for the selected `counter_sel` (`OUT_H`). The counter set is fixed at compile time in `counters.h`; each readout writes it to L1 next to the counts, and the host reads it back to decode the snapshot. The device walks the slots at freeze and the driver pulls the results into a pandas DataFrame and optionally a CSV. The derived metrics (utilisation %, stall %, and per-unit efficiency ratios) are computed in Python on top of the raw counts.

| Bank | Description |
|------|-------------|
| INSTRN_THREAD | Per-thread instruction issue counts, availability, and stall reasons |
| FPU | FPU and SFPU active cycles |
| TDMA_UNPACK | Unpacker busy signals, math pipeline status, srcA/B write port and overwrite signals |
| TDMA_PACK | Packer busy, dest-read availability, math availability |
| L1 | NoC ring transactions and L1 port arbitration (mux-selected). Absent on Quasar, where bank slot 3 carries the one l1_client CSR selection instead |

## How It Works

### Two builds, one test source

Perf tests support two builds of the same C++ source; each pytest invocation selects one build. `--enable-perf-counters` compiles the participating TRISCs and BRISC where present with `-DPERF_COUNTERS_COMPILED` (the WC build); without it `counters.h` compiles no counter code (the NC build). With counter readback available (hardware or the RTL emulator), the WC host writes a launch token and the counted-zone mask to `COUNTED_ZONES_REG` (`counters.h`, a bit per zone id) before it starts the cores. On ttsim, the host writes no launch word and the device suppresses counter-register accesses:

| Build | `PERF_COUNTERS_COMPILED` | Counted zones | Active part(s) of `START_PERF_MEASURE` | What it measures |
|-------|--------------------------|---------------|--------------------------------------|------------------|
| NC (no counters) | undefined | none, no counter code | `ZONE_SCOPED` (timing) + `MEASURE_PERF_COUNTERS` (the zone's rendezvous alone) | Per-zone wall-clock cycles (`RISCV_DEBUG_REG_WALL_CLOCK_L`), reported differently per run type: `L1_TO_L1` is the unpack-start to pack-end cross-thread span, the isolates are the measured thread's own zone, and `L1_CONGESTION` yields two columns, `[UNPACK]` and `[PACK]` |
| WC (with counters) | defined | `TILE_LOOP` in measured launches and `INIT` in extra launches when counter readback is available; none on ttsim | `MEASURE_PERF_COUNTERS` **and** `ZONE_SCOPED` | Per-zone HW counter snapshot **and** wall-clock cycles; wall-clock only on ttsim |

A hardware or RTL-emulator counter sweep adds one INIT-only launch per measured launch: `2 * run_count` launches per run type. On ttsim, which has no counter readback, the INIT-only launches are skipped and each run type uses `run_count` launches.

`START_PERF_MEASURE(name)` expands to `MEASURE_PERF_COUNTERS(name)` + `ZONE_SCOPED(name)`. In the NC build, `MEASURE_PERF_COUNTERS` performs the same real rendezvous as WC, at entry and, for the run types that keep it, at exit, with no action and no counter code (so it is not free, and it moves the NC baseline). The rendezvous has three threads on Wormhole and Blackhole and four on Quasar, where the SFPU TRISC takes part too: `sfpu_stub.h` opens the same `INIT` and `TILE_LOOP` zones, so a kernel with no SFPU half still completes it. Where counter readback is available, the WC build keeps **both** live: the counter scope performs the rendezvous *and* arms/freezes the HW counters, while `ZONE_SCOPED` records the per-zone wall-clock timestamps without adding another rendezvous. A complete WC run with counter readback therefore yields both counter and wall-clock data per counted zone under the same name; the host driver keys everything by `(test_variant, zone)` and can merge NC and WC results (or use the WC wall-clock directly).

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

- The perf counters are **shared by the core's TRISCs** (within one NEO on Quasar) and driven by shared debug registers (`PERF_CNT_ALL` and the per-bank `*2` command registers), so any RISC can arm/freeze them. The identity of the issuing thread does not change what is counted.
- The entry rendezvous waits for every thread before arming, so the window opens after all of them have finished the previous zone.

Pack is chosen because a sweep found arming there halves the total `L1_CONGESTION` error against arming on TRISC0. Fixing it also matters for its own sake: letting the arming thread vary is how the two builds ended up releasing from different threads.

**Freezing is per run type**, decided by `exit_barrier_for()` in `counters.h`: span run types keep the exit rendezvous; every single-thread run type omits it.

| Run type | Exit rendezvous | Freezing thread |
|---|---|---|
| `L1_TO_L1`, `L1_CONGESTION` | yes | pack, once all threads have arrived (three on WH/BH, four on Quasar) |
| `UNPACK_ISOLATE` | no | unpack, just after its own `ZONE_END` |
| `MATH_ISOLATE`, `PACK_ISOLATE` | no | the measured thread, just after its own `ZONE_END` |
| `SFPU_ISOLATE` (Quasar) | no | the SFPU TRISC (`LLK_TRISC_ISOLATE_SFPU`), just after its own `ZONE_END` |

For `MATH_ISOLATE`, math freezes the counters without an exit barrier; the entry rendezvous still runs. `START_PERF_MEASURE` declares the counter scope before the profiler zone, so counters are armed before `ZONE_START` and frozen after `ZONE_END`. For a counted zone on hardware or the RTL emulator, before counter overflow, `OUT_L` therefore exceeds the measured thread's wall-clock span: it includes the arming/release and freeze overhead. Without an exit rendezvous, peers can leave the scope while the measured thread is still running, so their epilogues can contribute events inside its counter window.

### The `MEASURE_PERF_COUNTERS` macro

Expands to a `perf_counter_scoped<PERF_RUN_TYPE>` RAII object in both builds (in NC, it keeps the rendezvous but compiles no counter code); the run type is a template parameter because it selects the exit shape above. Its constructor and destructor execute the following sequence; the rendezvous runs in both builds, and the counter actions are WC-only. The register actions below apply to hardware and the RTL emulator; ttsim suppresses counter-register accesses and reports wall-clock profiling:

1. **Constructor (zone entry).** Every thread rendezvous; the **action thread (pack)** then runs `arm_all_counters` and releases the others. When the launch counts the zone (the launch word, below) it writes the rising-edge start bit to `PERF_CNT_ALL` (FPU + INSTRN), `PERF_CNT_TDMA_UNPACK2`, `PERF_CNT_L1_2`, and `PERF_CNT_TDMA_PACK2`, which clears all banks and starts the count; otherwise it writes 0, no command, so every launch runs the same instructions. For a single thread run type it first reads out an earlier zone the measured thread froze, when the launch counted one (INIT's own launch). On Quasar there is no L1 bank to pulse; if an l1_client selection is compiled in, its clear-on-read `CNT` register is read once at that point instead, so the window starts at zero.

2. **Body.** Every thread runs the work inside the scope. Counters tick continuously on the shared backend.

3. **Destructor (zone exit).** For the run types that keep the exit rendezvous, the action thread freezes the counters in that rendezvous and reads the zone if the launch counts it; otherwise the measured thread freezes directly (`freeze_zone`, which records the zone as pending) and reads it later, after `run_kernel` (`read_last_zone`) or in the next zone's entry rendezvous, so no readout runs inside a window. Note `PROFILER_SYNC()` (`tensix_sync`) is a hand-written statement in each kernel and the isolate and congestion paths can `return` past it, so a thread's backend is not guaranteed drained when the stop bit is written; `fence_compiler()` around the rendezvous is a compiler barrier only. The freeze writes the rising-edge stop bit to the same four registers. The readout (`read_all_counters`) stores each bank's `OUT_L` (reference cycles) in the zone's five cycle words, writes the built-in config into the shared 200-word config buffer at `0x169000` and, for every valid slot, programs the bank's mode register with the `counter_sel`, reads `OUT_H` (the event count) and stores it in the per-zone data area. On Quasar slot 3 is the l1_client selection: its value is `l1_client_read()` of the clear-on-read `CNT` register and its cycles are the INSTRN bank's. It then writes the zone's `SYNC_ZONE_COMPLETE` flag with the launch's token above it; the host reads only the zones the launch counted, only with that token, and fails the test when one has no such readout.

Each zone gets its own data block in L1 (see [L1 Layout](#l1-layout-and-zone-buffers)) so multiple measurement scopes in the same kernel produce independent snapshots. The device supports `PERF_COUNTERS_MAX_ZONES = 8` zones, but the host counts only two: zone 0 is `INIT` and zone 1 is `TILE_LOOP`, and the launch word names one of them per launch, so a third `START_PERF_MEASURE` zone keeps its rendezvous and its wall clock but is never counted. Zone names in use are `INIT`, `TILE_LOOP`, and `UNINIT` in `fast_tilize_bh` / `fast_untilize` only; `UNINIT` uses bare `ZONE_SCOPED`, so it is timing-only with no rendezvous; identical names share a zone.

The counters are driven through the registers described in the [hardware reference](hardware_reference.md#hardware-register-reference). The macro path always uses mode 0. Mode 1 is unused in the LLK test suite. The `counter_sel` field is rewritten on each slot read so a single bank can multiplex multiple counters into one measurement window. The per-zone path writes `1` to arm and `2` to freeze, so each write is itself a rising edge on the opposite bit and the stop write is what re-creates the next 0→1 edge.

#### The `llk_barrier::rendezvous` barrier

`llk_barrier::rendezvous` in `barrier.h` uses separate arrival and release semaphores. Every peer samples its release level, posts an arrival, and waits for the release level to change. Pack waits for all peers, drains the arrival semaphore, executes the action, and flips the release level. The arrival drain therefore happens **before** arming at zone entry. Both builds use the same rendezvous; NC supplies no counter action.

On Wormhole and Blackhole, `PACK_DONE` is the arrival semaphore and `UNPACK_OPERAND_SYNC` is the shared release semaphore. Peers observe the release level without consuming it. The header reserves these names against other LLK use. Quasar has 32 Tensix semaphores in four banks of eight, each with maximum value 15. The barrier uses bank 1: semaphore 8 is the arrival semaphore and 9–11 are separate release semaphores for unpack, math and SFPU; LLK operations use bank 0. On Quasar, the unpack TRISC clears these semaphores on hardware or the RTL emulator before releasing the others in a profiler launch. On Wormhole and Blackhole, BRISC calls `device_setup()` to initialize the two barrier semaphores before releasing the TRISCs.

Waiters poll the PC buffer rather than L1. The rendezvous still has a cost in both builds; removing counter work does not remove synchronization. The exit rendezvous exists only for the span run types listed above.

### Configure each counting launch from BRISC or the host

On hardware or the RTL emulator, before any TRISC kernel runs, BRISC calls `clear_debug_features()` and, in the WC build, `configure_from_brisc()` (both from `brisc.cpp`). Quasar has no BRISC in this harness, so the unpack TRISC calls `clear_debug_features()` from `trisc.cpp`, after `device_setup()` and before it clears the other TRISCs out of soft reset; the host programs Quasar's bank registers through NEO0's register window before a counting launch (`counters.py`). This:

- Clears `DBG_FEATURE_DISABLE` to `0` in every launch of both builds, see [DBG_FEATURE_DISABLE scrub](#dbg_feature_disable-scrub) below.
- In a WC launch that counts a zone, programs each bank's reference-period and mode registers. Wormhole and Blackhole also set `PERF_CNT_MUX_CTRL` for L1; Quasar has no L1 counter bank, so its `PERF_CNT_MUX_CTRL` register (offset `0xF4`) is present but unused by this path. Registers only: BRISC reading an L1 config right before the release moved the TRISCs' timing.

On ttsim these register accesses are suppressed. Nothing is armed during configuration: each counted zone's arm starts the counters. The readout writes the per-architecture `BUILTIN_COUNTER_CONFIG` (110 slots on WH; 101 on BH for L1 mux groups 0–4 or 89 for group 5, since only one group is emitted; 77 on Quasar, 78 with an l1_client selection) into the shared L1 config buffer at `0x169000` next to the counts, and stamps the zone's sync word with the launch's token; the host reads only the zones the launch counted and only with that token. That array is built at compile time from the shared select tables, see [Shared counter definitions](hardware_reference.md#where-the-definitions-live).

##### `DBG_FEATURE_DISABLE` scrub

`DBG_FEATURE_DISABLE` is a 16-bit debug/chicken-bit register whose bits toggle low-level behaviors, notably randomized L1 arbitration (bit 3; the name is from the RTL and is not a symbol in this tree), L1 atomic serialization, and L1 read-enable override. It resets to `0` (all normal), but HW register state **leaks between tests** run back-to-back on an un-reset device, so a prior test that set one of these bits would silently perturb, and make nondeterministic, the L1 counters (16 per run, since one mux group is captured). BRISC writes `0` here to guarantee a clean baseline regardless of leaked state; the blanket write (rather than clearing one bit) is deliberate because any of the bits, not just LFSR, would skew the measurement. Verified: with a leaked `0x8` present, the L1 metrics jitter 40–98 % run-to-run without this scrub and are byte-identical with it. Both builds run this scrub in every launch, as metal's firmware does before each kernel: it is not counter work.

### Reading results from host

After the kernel completes:

1. The host process reads the per-zone data area back from device L1.
2. `read_counters()` decodes each 32-bit config word (bit 31 valid, bits 7:0 bank, bits 16:8 `counter_sel`, bits 19:17 `l1_mux`), looks up the human-readable counter name (parsed at import from the same arch header by `tt_llk_perf.headers.bank_tables()`, see [Shared counter definitions](hardware_reference.md#where-the-definitions-live)), and pairs every event count with that zone's bank cycle count.
3. `read_counters()` returns an in-memory long-format DataFrame with columns `zone`, `bank`, `counter_name`, `counter_id`, `cycles`, `count`, `l1_mux`. `compute_metrics()` produces its own separate rows; it does not add columns to that frame. Host-side only zone 0 and zone 1 are named (hard-coded to `INIT` and `TILE_LOOP`); further zones stay `ZONE_n` and never join the wall-clock rows.

Because both wall-clock cycles (NC build, `ZONE_SCOPED` start/end timestamps from `RISCV_DEBUG_REG_WALL_CLOCK_L`) and HW counter cycles (WC build, `OUT_L`) are tagged with the same zone name, the test driver merges them by `(test_variant, zone)`.

## How to Run

The LLK test suite uses a two-phase pytest flow: a compile-producer phase that builds every variant in parallel and a compile-consumer phase that runs them on hardware.

```bash
cd tt_metal/tt-llk/tests     # LLK_HOME is defaulted by the pytest plugin
./setup_testing_env.sh        # installs the SFPI toolchain; run it, do not source it (it calls exit)
export CHIP_ARCH=blackhole   # or wormhole or quasar

# Phase 1: build all variants (no HW access)
pytest --compile-producer --enable-perf-counters -n 8 -x ./python_tests/perf_eltwise_binary.py

# Phase 2: run on HW
pytest --compile-consumer --enable-perf-counters -x ./python_tests/perf_eltwise_binary.py
```

The WC build keeps its artefacts apart from NC's (its variant hashes and shared directory include the L1 selection), so switching between the builds needs no wipe, but each needs its own producer run.

To capture a different L1 mux group, `export LLK_PERF_L1_MUX_GROUP=<0-5>` before **both** phases. It is an environment variable rather than a CLI flag and is compiled into `brisc.elf`, so each group needs its own producer run; the readout checks the group found in L1 against the one requested and fails the run if they disagree, so a stale `brisc.elf` cannot return a self-consistently mislabelled dataset.

Quasar has no L1 counter bank. Its bank slot 3 can carry one event of the L1 client CSR instead: `export LLK_PERF_L1_CLIENT_SEL=<subport*8+event>` before the producer phase (default off; it is compiled into the four TRISC ELFs, so each selection needs its own producer run). Sub-ports are 0-3 TRISC, 4 THCON, 5-24 unpacker reads, 25-36 packer writes; events 1-7 are named by `tt_llk_perf.metrics.quasar_l1_client_label()`, event 0 is unused and THCON events 1-3 are rejected at compile time (`llk::perf::l1_client_selection_is_valid`) because they alias the TRISC port. The counter is clear-on-read and has no reference counter of its own, so it is referenced to the INSTRN bank's cycles, and its metric column is named after the selection (`l1_client_<port>_<event>_pct`, or `_ratio` for the pending-request carry; see the tech report's "L1 client events (Quasar)").

The `--enable-perf-counters` flag triggers three things:

1. Test sources (and BRISC where present) are compiled with `-DPERF_COUNTERS_COMPILED` (the WC build) and the L1 selection (`-DLLK_PERF_L1_MUX_GROUP=<n>`, or `-DLLK_PERF_L1_CLIENT_SEL=<n>` on Quasar).
2. On hardware or the RTL emulator, the driver writes the launch token and counted zones before each launch. BRISC (the host on Quasar) configures the counters, the zones arm them and the TRISCs read each counted zone out. Measured launches count `TILE_LOOP`; `INIT` uses one extra launch per run type and run, keeping its readout out of the measured launch. On ttsim, no launch word is written and no INIT-only launch runs.
3. Where counter readback is available, the Python driver calls `read_counters()` after each counted launch, writes derived percentage metrics into the main CSV, and writes raw counts to `*.counters.csv` under `--dump-perf-counters`. On ttsim, it skips counter readback and reports profiler timings only.

Without the flag the suite builds the NC variant, which compiles no counter code, and only `ZONE_SCOPED` wall-clock data is collected.

### CLI flags

| Flag | Implies `--enable-perf-counters` | Effect |
|------|----------------------------------|--------|
| `--enable-perf-counters` | no | Build the WC variant and collect raw counters per zone |
| `--dump-perf-counters` | yes | Export raw counter values to a separate `<test>.counters.csv` alongside the main results CSV |

`--dump-perf-counters` implicitly enables counter collection; you don't need to specify `--enable-perf-counters` separately.

### Output

For each test variant, the WC build emits:

- One row per zone in `perf_data/runs/<tag>/<test>/<test>.csv`, with wide `<RUN_TYPE>_<stat>(<metric>)` columns. `<test>.post.csv` is the same data with `TILE_LOOP` wall-clock divided by `loop_factor x tile_cnt`; `INIT` rows and every counter/`OUT_L` column stay absolute, so multiply before comparing the two cycle numbers.
- A `*.counters.csv` file when counter readback is available and `--dump-perf-counters` was passed. On ttsim, there is no counter CSV.

The NC build emits per-zone wall-clock cycle counts in the same results DataFrame so a single run with both builds (different pytest invocations) can be merged off-line to compare wall-clock cycles against counter-derived cycle counts.

## Hardware reference

The counter inventory per architecture, where the shared definitions live and how to add a counter, the debug registers, the L1 mux groups and the hardware limitations are in [hardware_reference.md](hardware_reference.md). On top of that the harness emits one L1 mux group per build, so `BUILTIN_COUNTER_CONFIG` holds 110 slots on Wormhole, 101 slots on Blackhole for L1 mux groups 0–4 or 89 for group 5, 77 (78 with an l1_client selection) slots on Quasar, written into the 200 config words in L1 with the rest zero.

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
         │ Quasar: the launch word (counters.h)      │
         +────────────────────────────────────────────+
```

The layout is bounded by two `static_assert`s to stay below `0x16AFF0` (the profiler region boundary). Each zone reserves `PERF_COUNTERS_ZONE_SIZE = (5 + 200) × 4 + 40 = 860` bytes, supporting up to `PERF_COUNTERS_MAX_ZONES = 8` zones per kernel.

The 200-word shared config, written by every readout, is the authoritative record of which counters are recorded for every zone (the host reads it back to decode). There is no per-zone configuration: every zone records the same set of counters but stores its own snapshot.

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
- **`PACK_DONE` and `UNPACK_OPERAND_SYNC` are reserved by the barrier on Wormhole and Blackhole.** A measured kernel must not use either semaphore for its own handshake; the barrier owns their arrival-count and release-level state.
- **The host asserts zones do not overlap across threads.** No thread may open `TILE_LOOP` before every thread has closed `INIT`. A failure almost always means a kernel used `ZONE_SCOPED` instead of `START_PERF_MEASURE`, which is what supplies the entry rendezvous. A hang (`TENSIX TIMED OUT`) in every build means a thread skipped a zone: every thread must open every zone in every configuration, so a compile-time condition on a thread's work belongs inside the zone, not around it (the Quasar unpack-to-dest variants of the pack and unpack tests hung this way until their math zones were opened unconditionally).
- **Reports are grouped by invocation.** Results live under `perf_data/runs/<tag>/`, with `perf_data/latest` pointing at the latest run. Use a distinct `PERF_RUN_TAG` for each build when retaining WC and NC reports for comparison.
- **`--logging-level DEBUG` or `TRACE` recompiles the measured kernel** with `-DDEBUG_PRINT_ENABLED`, which perturbs the numbers. Do not use it for a measurement run.
- **`no zone returned counter data` means the test was never measured.** The counter and metric columns will be absent; a with-counters versus no-counters comparison for that test is meaningless, not zero.
- **Instrumenting a kernel moves its no-counter baseline.** The rendezvous is real in both builds, so converting a kernel from `ZONE_SCOPED` to `START_PERF_MEASURE` shifts its timings; numbers from before and after are not comparable.
- **`PROFILER_SYNC()` is per-kernel and not universal.** `fast_tilize_bh_test.cpp` omits it entirely and no `UNINIT` zone has one, so those windows close without draining the backend.

- **A pytest invocation runs one build.** `--enable-perf-counters` selects WC, otherwise NC, so a single invocation cannot produce both. The WC build records wall-clock (`ZONE_SCOPED`) *alongside* the counters, so it is self-contained; the NC build is run separately only when a counter-overhead-free timing baseline is wanted. Results merge off-line by `(test_variant, zone)`.
- **The window is `[all threads armed … all threads finished]`.** The rendezvous arms after every thread has entered, and freezes after every thread has finished for the run types that keep the exit barrier. Each thread stamps the release with its own wall-clock read, and those reads serialize on the single shared clock, so the per-thread zone starts differ by ~12–40 cyc (irreducible; not a bug).
- **`PERF_COUNTERS_MAX_ZONES = 8` per kernel.** Adding a 9th distinct `MEASURE_PERF_COUNTERS("...")` name silently reuses zone 0. Reuse the same name across multiple call sites if you want them in the same bucket.
- **One L1 mux group per run.** `PERF_CNT_MUX_CTRL` selects the group while the counters count, not when they are read, so the freeze path cannot re-aim it and a run observes exactly one group. Select it with `LLK_PERF_L1_MUX_GROUP` and sweep it across runs.
- **Counters off compiles no counter code.** Only WC gets `-DPERF_COUNTERS_COMPILED` and the L1 selection; NC keeps each zone's rendezvous and the `DBG_FEATURE_DISABLE` scrub, and its host writes no launch word. Every counter access sits under `#if defined(PERF_COUNTERS_COMPILED)`, and without it the launch hooks `brisc.cpp` and `trisc.cpp` call are empty. The two builds are therefore different programs: the counter code moves every later address in WC, and Wormhole timing follows code placement, so a WC minus NC delta includes the layout effect any code change has, not only counter overhead.
- **Test isolation.** As with every LLK test, counter state at kernel entry is whatever the previous test left behind. Nothing clears the zone buffers: the host reads only the zones a launch counted, and only when the zone's sync word carries that launch's token. HW counter registers may carry residual values until the first `MEASURE_PERF_COUNTERS` rising-edge clear.
- **WC's configure before the release writes registers only.** When BRISC read the 200 config words from L1 there, the L1 arbiter state it left moved 859 of 1508 Wormhole TILE_LOOP values against a launch without them, while register writes or a delay of the same length moved none.
- **L1 layout must stay below the profiler region.** `PERF_COUNTERS_LAYOUT_END` must not overlap the profiler's lowest L1 address (`llk_profiler::EPOCH_ADDR`). Two `static_assert`s enforce this: a literal one in the always-compiled section (BRISC has no `llk_profiler` namespace) and a symbolic one in the `LLK_PROFILER` section that tracks the profiler layout automatically.
- **Minimum window size.** Size every measured window above ~1k cycles using the test's `LOOP_FACTOR`; PR #51912 raised the suite's factors for exactly this reason. Below that, a few cycles of instrument floor read as a large percentage, and the timing `mean` is affected as well as the derived ratios. Note the report divides `TILE_LOOP` wall-clock by `loop_factor x tile_cnt` but leaves `INIT` and every counter column absolute.

- **The inventory is shared with metal.** `counters.h` includes `tools/include/perf_counters/{inventory,registers,hw}.h` and `counters.py` parses the same headers through `tt_llk_perf.headers`. The Python parser matches one `{PerfCounterType::NAME, select}` entry per line and `<name>_counters [ATTRIBUTE] =` array names (the optional upper-case token is `quasar.h`'s `LLK_PERF_TABLE_SECTION`), so keep that literal form when editing the tables. The config-word bit layout (`PERF_CFG_*`) and the bank-id to name mapping are still mirrored between `counters.h` and `counters.py`; they are this infra's own L1 ABI, not part of the shared headers.
