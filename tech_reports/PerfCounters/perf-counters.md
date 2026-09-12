# Hardware Performance Counters

## Quick Links
- User guide: [docs/source/ttnn/ttnn/profiling_ttnn_operations.rst](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst)
- Counter names, per-arch select tables, register map (shared with the tt-llk harness): [tt_metal/tt-llk/tools/include/perf_counters/](../../tt_metal/tt-llk/tools/include/perf_counters/)
- Profiler firmware side (record format, group bits, readout): [tt_metal/tools/profiler/perf_counters.hpp](../../tt_metal/tools/profiler/perf_counters.hpp)
- Header parsers and metric formulas (shared with the tt-llk harness): [tt_metal/tt-llk/tools/python/tt_llk_perf/](../../tt_metal/tt-llk/tools/python/tt_llk_perf/)
- Tracy adapter: [tools/tracy/perf_counter_analysis.py](../../tools/tracy/perf_counter_analysis.py)
- CSV writer: [tools/tracy/process_ops_logs.py](../../tools/tracy/process_ops_logs.py)

## Overview

Each Tensix core contains hardware performance counters organized into five banks (FPU, TDMA_UNPACK, TDMA_PACK, INSTRN_THREAD, L1). These counters measure cycle-level events: how many cycles the FPU was active, how many cycles a thread was stalled, how many cycles each L1 memory port had traffic, and so on. Quasar has the first four banks in every NEO and no L1 bank; one l1_client event CSR per NEO stands in for it.

The counters are built from a reusable RTL module (`tt_perf_cnt`) that provides three values per event: **req_cnt** (cycles the event signal was high), **grant_cnt** (cycles the grant/ready signal was high), and **ref_cnt** (total elapsed cycles). From these raw values, the profiler computes derived metrics like utilization (`req_cnt / ref_cnt`), backpressure (`(req_cnt - grant_cnt) / req_cnt`), and cross-bank ratios that combine counters from different banks.

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

With `--perf-counter-multipass` a request is split into passes (at most three groups and one L1 bank per pass) and `all` expands to the architecture's full group set.

Available counter groups for `--profiler-capture-perf-counters`: `fpu`, `pack`, `unpack`, `l1_0`, `l1_1`, `instrn`, `all`. Blackhole also supports `l1_2`, `l1_3`, `l1_4`, `l1_5`; `all` expands to the running architecture's full set. Quasar has no L1 groups: `all` there is `fpu,pack,unpack,instrn` and a request naming an `l1_*` group is rejected. The multipass scheduler still splits those four groups 3 + 1; the environment variable path below takes all four in one run.

Two limits force a request like `all` into several capture passes: the BRISC firmware image only fits the readout code for 3 counter groups, and the L1 banks share one count-time mux, so at most one L1 bank can count per run. `python -m tracy` schedules the passes automatically. A request that fits one pass runs once, exactly as before; a request that does not stops with the printed pass plan unless `--perf-counter-multipass` is given, in which case the workload is replayed once per pass and the per-pass device logs are merged. See the [user guide](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst) for details.

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

The env-var path selects one pass directly, so keep it to at most 3 groups: the BRISC firmware image only fits the readout code for 3, and a larger mask overflows its `.text` section (measured on Blackhole). Example single-pass capture:

```bash
export TT_METAL_PROFILE_PERF_COUNTERS=11   # FPU | PACK | L1 bank 0
```

**L1 bank mutual exclusion:** all L1 banks share the same hardware mux (selected via `MUX_CTRL`), so only one L1 bank may be enabled per run; the env-var path throws if more than one L1 bit is set. For anything that needs several passes, use `python -m tracy --perf-counter-multipass` (above), which schedules the passes and merges the results.

**Quasar mask:** the L1 bits do not exist there. `tt_metal/jit_build/build.cpp` remaps the front end's `all` mask 47 to 39 (`FPU | PACK | UNPACK | INSTRN`) and fails with `TT_FATAL` on any other mask with a bit outside those four; `perf_counters.hpp` has a matching `#error`. The four groups fit the DM0 firmware in one run.

**`TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL`** (Quasar only, default -1 = off) routes the one l1_client event counter of every NEO for the run. The value is `subport * 8 + event`: 37 subports (0-3 TRISC, 4 THCON, 5-24 unpacker read interfaces, 25-36 packer write interfaces) by 8 events, 296 selections of which 256 are valid. Event 0 reads 0 in the RTL and the THCON subport's events 1 to 3 duplicate the TRISC port's SBank 0, so `build.cpp` rejects those (same rule as `llk::perf::l1_client_selection_is_valid` in `quasar.h`, which the firmware `static_assert`s). The CSR has no reference counter; its record carries the wall-clock span between arm and freeze, and the host names the resulting column after the selection (see [L1 client events (Quasar)](#l1-client-events-quasar)).

```bash
export TT_METAL_PROFILE_PERF_COUNTERS=39        # FPU | PACK | UNPACK | INSTRN, the Quasar "all"
export TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL=41 # subport 5 (unpacker 0, interface 0), event 1: L1_CLIENT_UNPACK0_IF0_SBANK0_SBANK_POP
```

### Architecture Summary

The per-architecture inventory is the set of tables in `tt_metal/tt-llk/tools/include/perf_counters/blackhole.h`, `wormhole.h` and `quasar.h`; the derived-metric catalogue below is shared, and a metric whose counters exist on only one architecture reports N/A on the others.

**Wormhole** has `PACK_COUNT=4` (4 packer engines), active `o_math_instrnbuf_rden`, and all TDMA counters live. The L1 mux is 1-bit (2 positions: ports 0-7 and 8-15).

**Blackhole** has fewer raw TDMA counters because `PACK_COUNT=1` ties the per-engine busy and dest-read signals for engines 1-3 to constants. Only RTL-live signals are read from hardware. Any counter whose RTL signal is hardwired to a constant has been omitted from the arch tables, and any aliased grant counter is consolidated to one canonical entry. `Math-to-Pack Handoff Efficiency` falls back to the bank's reference cycles as denominator when `PACKER_BUSY` is 0 for a given workload (e.g. pure-SFPU ops that don't drive the packer); `Packer Efficiency` reports N/A there. TDMA_UNPACK grant banks 4-6 (sels 260-262) have identical RTL wiring on WH and BH (verified: srcB port, srcA overwrite, srcA port). Blackhole has more L1 mux positions (6 vs 2 for Tensix, `L1_MUX_POSITIONS` in the arch headers).

**INSTRN_THREAD bank.** `perf_cnt_instrn_thread` is built from a Verilog generate array in `tt_instruction_thread.sv` and has architecture-specific counter_sel mappings. Req-side: sels 0-23 are per-thread instruction-type availability (CFG/SYNC/THCON/MOVE/FPU/UNPACK/PACK, 3 threads each), sels 24-26 are per-thread total stall cycles, and sels 27+ are stall reasons. On WH the shared stall conditions (SRCA/B clear/valid) are replicated across 3 slots each (sels 27-38); on BH they occupy 1 slot each (sels 27-30). Per-thread stall reasons are thread-major: WH sels 39-65 (9 types x 3 threads), BH sels 31-57. Grant-side: the RTL wires grant as `{8{ibuffer_rden[th]}}` per instance and `{9{inst_stall_thread[th]}}` per per-thread stall-reason instance, so the 24 possible issue-count sels collapse to 3 distinct per-thread values and the per-thread stall-reason grants reproduce `THREAD_STALLS_{th}`. We expose only the distinct grants: `THREAD_INSTRUCTIONS_{0,1,2}` at sels 256/264/272 (one per instance) and `ANY_THREAD_STALL` at sel 283. The tables are in the arch headers `blackhole.h`, `wormhole.h` and `quasar.h`; `inventory.h` picks one from `ARCH_*` and `perf_counters.hpp` is arch-agnostic (`table_for` returns an empty table for the L1 mux positions Wormhole does not decode, and for the L1 bank on Quasar).

**Quasar** keeps four of the banks in every NEO (`quasar.h`: FPU 3, TDMA_UNPACK 18, TDMA_PACK 5, INSTRN_THREAD 51 entries) and DM0 reads all four NEOs. FPU exposes selects 0 and 1 plus the grant of 1 (`MATH_COUNTER`); the grant of select 0 is tied to 0. TDMA_UNPACK and TDMA_PACK index one shared readout of 21 slices through their two register sets: slices 0 to 10 are the unpack instance (18 live selections including the grants), slices 11 to 18 the pack instance (request slices 12 to 17 are tied to 0, so only `PACKER0_DEST_READ_REQ`, `PACKER_BUSY` and the grants at 11, 15 and 16 are read), slices 19 and 20 are constant 0 and selects 21 to 31 read 0. INSTRN_THREAD has 51 slices: selects 0 to 31 are instruction class times thread (`class*4 + thread` over CFG, SYNC, THCON, XSEARCH, INSTISSUE, MATH, UNPACK, PACK; XSEARCH is tied to 0 and left out of the table), 32 to 35 the per-thread stall counts, 36 to 50 fifteen backend stall reasons OR-reduced across the four threads; the grant of every class select is that thread's issue count (`THREAD_INSTRUCTIONS_t`). There is no L1 counter bank (`table_for(Bank::L1)` is empty, `L1_MUX_POSITIONS` is 0). Instead every NEO has one clear-on-read l1_client CSR behind a 37 subport by 8 event mux, routed per run with `TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL` (above).

---

## Derived Metrics Reference

Every derived metric is computed by one shared module, [tt_metal/tt-llk/tools/python/tt_llk_perf/metrics.py](../../tt_metal/tt-llk/tools/python/tt_llk_perf/metrics.py). The Tracy tool computes it per operation and core and aggregates to Min/Median/Max/Avg across cores; the tt-llk test harness computes it per zone and run and aggregates to mean/std across runs. The tables below are the complete set. The module is the source of truth, and a unit test (`tests/ttnn/tracy/test_perf_metrics_common.py`) fails if this file stops listing a metric the module computes.

Metrics come in two families, told apart by the key suffix:

- `*_pct`: percentages bounded to 0-100 by construction: the numerator counts a subset of the cycles or events the denominator counts (a 1-bit counter over its bank's `ref_cnt`, an RTL-proven subset such as an arbiter accept over its request, or a share of a sum). Where a row says "clamped", the value is additionally clamped to 0..100 and the row states why.
- `*_ratio`: unbounded raw ratios that can exceed 1.0 by design, because the numerator and denominator come from different measurement domains or because overlapping events are summed. Reported with a `(ratio)` unit and never clamped; the excess over 1.0 is the signal.

A metric whose counters do not exist on the running architecture reports N/A (blank), never 0: the Wormhole-only per-engine packer metrics are N/A on Blackhole, the Blackhole-only extended L1 groups are N/A on Wormhole, every L1 and NoC row is N/A on Quasar, and the Quasar-only rows (thread 3, the INSTISSUE class, the OR-reduced stall reasons, unpacker 2) are N/A on Wormhole and Blackhole. Cross-bank metrics are likewise N/A when one of their counter groups was not captured in the run.

In the formulas, "fpu / instrn / pack / l1 cycles" is that bank's reference-cycle count (`ref_cnt`, the elapsed cycles between counter start and stop), and `1 - x / y` denotes the complement of a counter that counts not-stalled or granted cycles.

### Compute

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| FPU Util (%) | `fpu_utilization_pct` | `FPU_COUNTER / fpu cycles` | Fraction of cycles the FPU executed an instruction. |
| MATH Util (%) | `compute_utilization_pct` | `MATH_COUNTER / fpu cycles` | FPU or SFPU active (the counter is the OR of both). |
| SFPU Util (%) | `sfpu_utilization_pct` | `SFPU_COUNTER / fpu cycles` | Fraction of cycles the SFPU was active. |
| FPU Execution Efficiency (ratio) | `fpu_exec_eff_ratio` | `FPU_COUNTER / MATH_INSTRN_AVAILABLE_1` | UNBOUNDED ratio: FPU busy cycles (dequeues from every thread) per cycle thread 1 had a math instruction ready; near or above 1 = compute-efficient, low = stall-bound. |
| Math Pipeline Utilization (%) | `math_pipeline_util_pct` | `MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE` | Available math instructions that actually started. |

### Math pipeline stalls

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Data Hazard Stall Rate (%) | `data_hazard_stall_pct` | `1 - MATH_NOT_D2S_STALLED / MATH_INSTRN_AVAILABLE` | MOVD2A data-hazard stall rate (counter counts not-stalled cycles). |
| Math Scoreboard Stall Rate (%) | `math_scoreboard_stall_pct` | `1 - MATH_NOT_SCOREBOARD_STALLED / MATH_INSTRN_AVAILABLE` | Scoreboard stalls. N/A when the pack group was not captured; clamped to 0..100 because the two counters come from separately started groups. |

### Unpacker

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Unpacker-to-Math Data Flow (ratio) | `unpack_to_math_flow_ratio` | `mean of the srcA and srcB flows` | UNBOUNDED ratio; combined unpacker-to-math data flow. |
| Unpacker-to-Math Data Flow (srcA) (ratio) | `unpack_to_math_flow0_ratio` | `SRCA_WRITE_REQ / UNPACK0_BUSY_THREAD0` | UNBOUNDED ratio: srcA write requests per unpacker-0 busy cycle. THCON and other-thread writes also count, so it can exceed 1. |
| Unpacker-to-Math Data Flow (srcB) (ratio) | `unpack_to_math_flow1_ratio` | `SRCB_WRITE_REQ / UNPACK1_BUSY_THREAD0` | UNBOUNDED ratio: srcB write requests per unpacker-1 busy cycle. |
| SrcA Write Actual Efficiency (%) | `srca_write_eff_pct` | `SRCA_WRITE_NOT_BLOCKED_PORT / SRCA_WRITE_REQ` | srcA write requests that the write port accepted. |
| SrcB Write Actual Efficiency (%) | `srcb_write_eff_pct` | `SRCB_WRITE_NOT_BLOCKED_OVR / SRCB_WRITE_REQ` | srcB write requests not blocked by overwrite protection. |
| SrcA Write Overwrite Blocked Rate (%) | `srca_write_ovr_blocked_pct` | `1 - SRCA_WRITE_NOT_BLOCKED_OVR / SRCA_WRITE_REQ` | srcA writes blocked by overwrite protection. |
| SrcB Write Port Blocked Rate (%) | `srcb_write_port_blocked_pct` | `1 - SRCB_WRITE_NOT_BLOCKED_PORT / SRCB_WRITE_REQ` | srcB writes blocked on the write port. |
| Unpacker0 T1 Share (%) | `unpack0_thread1_share_pct` | `UNPACK0_BUSY_THREAD1 / (thread0 + thread1 busy)` | Unpacker-0 busy cycles driven by the math thread. |
| Unpacker1 T1 Share (%) | `unpack1_thread1_share_pct` | `UNPACK1_BUSY_THREAD1 / (thread0 + thread1 busy)` | Unpacker-1 busy cycles driven by the math thread. |
| SrcA Write Even-TID Share (%) | `srca_write_even_tid_share_pct` | `SRCA_WRITE_TID_EVEN / (even + odd writes)` | srcA writes from even thread ids (the counter tests thread-id bit 0, lane 0). |
| SrcB Write Even-TID Share (%) | `srcb_write_even_tid_share_pct` | `SRCB_WRITE_TID_EVEN / (even + odd writes)` | srcB writes from even thread ids (same split). |

### Packer

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Packer Utilization (%) | `pack_utilization_pct` | `PACKER_BUSY / pack cycles` | Fraction of cycles any packer engine was busy. |
| Packer Efficiency (%) | `pack_dest_eff_pct` | `PACKER0_DEST_READ_REQ / PACKER_BUSY` | Packer busy cycles that issued a dest read request. A dest read request implies a non-empty packer request FIFO, which is the busy condition, so this is a true fraction. N/A when the packer is idle. |
| Pack Dest Grant Efficiency (%) | `pack_dest_grant_eff_pct` | `DEST_READ_GRANTED_0 / PACKER0_DEST_READ_REQ` | Dest read requests that were granted. |
| Packer Engine 0 Util (%) | `packer0_util_pct` | `PACKER_BUSY_0 / pack cycles` | Per-engine packer 0. Wormhole only; N/A on Blackhole. |
| Packer Engine 1 Util (%) | `packer1_util_pct` | `PACKER_BUSY_1 / pack cycles` | Per-engine packer 1. Wormhole only; N/A on Blackhole. |
| Packer Engine 2 Util (%) | `packer2_util_pct` | `PACKER_BUSY_2 / pack cycles` | Per-engine packer 2. Wormhole only; N/A on Blackhole. |
| Packer Load Imbalance (%) | `packer_load_imbalance_pct` | `(max - min) / max over the four packer engine counters` | Wormhole only: N/A unless all four engine counters were captured; an idle engine counts as 100% imbalance. |

### Pipeline handoff

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Math-to-Pack Handoff Efficiency (ratio) | `math_to_pack_handoff_ratio` | `MATH_NOT_SCOREBOARD_STALLED / PACKER_BUSY (pack cycles when the packer is idle)` | UNBOUNDED ratio; above 1 the packer is the handoff bottleneck. |
| Compute-to-Unpack Ratio (ratio) | `compute_to_unpack_ratio` | `MATH_COUNTER / (UNPACK0_BUSY_THREAD0 + UNPACK1_BUSY_THREAD0)` | UNBOUNDED ratio; above 1 = compute-bound, below 1 = unpack-bound. |

### Thread stalls and waits

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Thread 0 Stall Rate (%) | `unpack_thread_stall_pct` | `THREAD_STALLS_0 / instrn cycles` | Thread 0 (unpack) stall rate. |
| Thread 1 Stall Rate (%) | `math_thread_stall_pct` | `THREAD_STALLS_1 / instrn cycles` | Thread 1 (math) stall rate. |
| Thread 2 Stall Rate (%) | `pack_thread_stall_pct` | `THREAD_STALLS_2 / instrn cycles` | Thread 2 (pack) stall rate. |
| SrcA Valid Wait (%) | `math_wait_srca_pct` | `WAITING_FOR_SRCA_VALID / instrn cycles` | Math waiting for srcA to become valid. |
| SrcB Valid Wait (%) | `math_wait_srcb_pct` | `WAITING_FOR_SRCB_VALID / instrn cycles` | Math waiting for srcB to become valid. |
| SrcA Clear Wait (%) | `srca_clear_wait_pct` | `WAITING_FOR_SRCA_CLEAR / instrn cycles` | Unpack waiting for srcA to clear. |
| SrcB Clear Wait (%) | `srcb_clear_wait_pct` | `WAITING_FOR_SRCB_CLEAR / instrn cycles` | Unpack waiting for srcB to clear. |
| Math Idle Wait T1 (%) | `math_idle_wait_t1_pct` | `WAITING_FOR_MATH_IDLE_1 / instrn cycles` | Thread 1 waiting for its own math unit to go idle. |
| Pack Idle Wait T2 (%) | `pack_idle_wait_t2_pct` | `WAITING_FOR_PACK_IDLE_2 / instrn cycles` | Thread 2 waiting for the packer to go idle. |
| Unpack Idle Wait T0 (%) | `unpack_idle_wait_t0_pct` | `WAITING_FOR_UNPACK_IDLE_0 / instrn cycles` | Thread 0 waiting for the unpacker to go idle. |
| Math Waiting on Unpack (T1) (%) | `math_wait_unpack_pct` | `WAITING_FOR_UNPACK_IDLE_1 / instrn cycles` | Math thread blocked on the unpacker. |
| Pack Waiting on Math (T2) (%) | `pack_wait_math_pct` | `WAITING_FOR_MATH_IDLE_2 / instrn cycles` | Pack thread blocked on math. |
| Unpack Waiting on Pack (T0) (%) | `unpack_wait_pack_pct` | `WAITING_FOR_PACK_IDLE_0 / instrn cycles` | Unpack thread blocked on the packer. |
| SFPU Idle Wait T1 (%) | `math_wait_sfpu_pct` | `WAITING_FOR_SFPU_IDLE_1 / instrn cycles` | Math thread waiting for the SFPU. |
| CFG Idle Wait T0 (%) | `cfg_idle_wait_t0_pct` | `WAITING_FOR_CFG_IDLE_0 / instrn cycles` | Thread 0 waiting for the config unit (trisc_cfg). |
| THCON Idle Wait T0 (%) | `thcon_idle_wait_t0_pct` | `WAITING_FOR_THCON_IDLE_0 / instrn cycles` | Thread 0 waiting for THCON. |
| MOVE Idle Wait T0 (%) | `move_idle_wait_t0_pct` | `WAITING_FOR_MOVE_IDLE_0 / instrn cycles` | Thread 0 waiting for MOVE. |
| Semaphore Zero Wait T1 (%) | `math_sem_wait_pct` | `WAITING_FOR_NONZERO_SEM_1 / instrn cycles` | Thread 1 waiting on a non-zero semaphore. |
| Semaphore Zero Wait T2 (%) | `pack_sem_wait_pct` | `WAITING_FOR_NONZERO_SEM_2 / instrn cycles` | Thread 2 waiting on a non-zero semaphore. |
| Semaphore Zero Wait T0 (%) | `sem_zero_wait_t0_pct` | `WAITING_FOR_NONZERO_SEM_0 / instrn cycles` | Thread 0 waiting on a non-zero semaphore. |
| Semaphore Full Wait T0 (%) | `sem_full_wait_t0_pct` | `WAITING_FOR_NONFULL_SEM_0 / instrn cycles` | Thread 0 waiting on a non-full semaphore. |
| Semaphore Full Wait T1 (%) | `sem_full_wait_t1_pct` | `WAITING_FOR_NONFULL_SEM_1 / instrn cycles` | Thread 1 waiting on a non-full semaphore. |
| Semaphore Full Wait T2 (%) | `sem_full_wait_t2_pct` | `WAITING_FOR_NONFULL_SEM_2 / instrn cycles` | Thread 2 waiting on a non-full semaphore. |
| Stall Overlap T0 (ratio) | `stall_overlap_t0_ratio` | `sum of the nine WAITING_FOR_*_0 counters / instrn cycles` | UNBOUNDED ratio; above 1 means several waits overlap in the same cycle. |
| Stall Overlap T1 (ratio) | `stall_overlap_t1_ratio` | `sum of the nine WAITING_FOR_*_1 counters / instrn cycles` | Same for thread 1. |
| Stall Overlap T2 (ratio) | `stall_overlap_t2_ratio` | `sum of the nine WAITING_FOR_*_2 counters / instrn cycles` | Same for thread 2. |
| Any-Thread Stall Rate (%) | `any_thread_stall_pct` | `ANY_THREAD_STALL / instrn cycles` | Cycles where any thread was stalled; one pipeline-level indicator. |

### Instruction issue and availability

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| CFG Instrn Avail Rate T0 (%) | `cfg_instrn_avail_t0_pct` | `CFG_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a CFG instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| SYNC Instrn Avail Rate T0 (%) | `sync_instrn_avail_t0_pct` | `SYNC_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a SYNC instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| THCON Instrn Avail Rate T0 (%) | `thcon_instrn_avail_t0_pct` | `THCON_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a THCON instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| MOVE Instrn Avail Rate T0 (%) | `move_instrn_avail_t0_pct` | `MOVE_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a MOVE instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| MATH Instrn Avail Rate T1 (%) | `math_instrn_avail_t1_pct` | `MATH_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is a MATH instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). |
| UNPACK Instrn Avail Rate T0 (%) | `unpack_instrn_avail_t0_pct` | `UNPACK_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a UNPACK instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| PACK Instrn Avail Rate T2 (%) | `pack_instrn_avail_t2_pct` | `PACK_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is a PACK instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). |
| T0 Instrn Issue Rate (%) | `thread0_ipc_pct` | `THREAD_INSTRUCTIONS_0 / instrn cycles` | Thread 0 issue rate; single-issue, so at most 100%. |
| T1 Instrn Issue Rate (%) | `thread1_ipc_pct` | `THREAD_INSTRUCTIONS_1 / instrn cycles` | Thread 1 issue rate. |
| T2 Instrn Issue Rate (%) | `thread2_ipc_pct` | `THREAD_INSTRUCTIONS_2 / instrn cycles` | Thread 2 issue rate. |

### L1 client ports and NoC

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| L1 Unpacker Port Util (%) | `l1_unpacker_util_pct` | `L1_0_UNPACKER_0 / l1 cycles` | Unpacker-0 L1 port utilization. |
| L1 Port 1 Util (%) | `l1_port1_util_pct` | `port 1 requests / l1 cycles` | L1_0 port 1 requests are L1_0_UNPACKER_1_ECC_PACK1 on Wormhole (pack1+ECC) and L1_0_UNPACKER_1_ECC on Blackhole (unpacker1+ECC). |
| L1 Packer Port 8 Util (%) | `l1_packer_port8_util_pct` | `L1_1_TDMA_PACKER_2 (Wormhole) or L1_1_PACKER_IF_0 (Blackhole) / l1 cycles` | L1_1 port 8, the packer's L1 write interface (interface 0 on Blackhole). |
| L1 TDMA Bundle Util (%) | `l1_tdma_bundle_util_pct` | `mean over the two L1_0_TDMA_BUNDLE ports / l1 cycles` | RISC and TRISC TDMA bundle traffic. |
| L1 Unpacker1 Ext Util (%) | `l1_unpacker1_ext_util_pct` | `mean over unpacker 1's extended read ports (L1_1 ports 9-11; plus L1_2 ports 16-19 on Blackhole)` | On Blackhole these ports also carry the packer L1-to-L1 read. |
| L1 Unpacker0 Ext Util (%) | `l1_unpacker0_ext_util_pct` | `mean over unpacker 0's extended read ports (L1_4 ports 35-39, L1_5 ports 40-41)` | Blackhole only; N/A on Wormhole. |
| L1 Packer Interfaces Util (%) | `l1_ext_pack_util_pct` | `mean over L1_3_EXT_PACKER_2-5, L1_4_EXT_PACKER_6-7 and L1_4_PACKER_IF_1_TAG_SEARCH` | Blackhole only; N/A on Wormhole. Port 34 is packer interface 1, shared with the tag-search accelerator that tt-metal never enables. |
| L1 Mean Client Util (%) | `l1_mean_client_util_pct` | `mean busy/ref over every present L1 client port` | One number for overall L1 client pressure. |
| L1 Port 2 Util (%) | `l1_port2_util_pct` | `L1_0_TDMA_BUNDLE_0_RISC / l1 cycles` | Port 2 carries TDMA bundle 0 (mover, packer read, THCON) together with BRISC, TRISC0 and NCRISC. |
| NOC Ring 0 Util (%) | `noc_ring0_util_pct` | `mean over the ring-0 ports (4 on Wormhole, 8 on Blackhole) / l1 cycles` | NoC ring 0 utilization. |
| NOC Ring 1 Util (%) | `noc_ring1_util_pct` | `mean over the ring-1 ports (4 on Wormhole, 8 on Blackhole) / l1 cycles` | NoC ring 1 utilization. |
| NOC Ring 0 Outgoing Util (%) | `noc_ring0_out_util_pct` | `mean over L1_0_NOC_RING0_OUTGOING_0/1 / l1 cycles` | Primary ring-0 outgoing channels only. |
| NOC Ring 0 Incoming Util (%) | `noc_ring0_in_util_pct` | `mean over L1_0_NOC_RING0_INCOMING_0/1 / l1 cycles` | Primary ring-0 incoming channels only. |
| NOC Ring 1 Outgoing Util (%) | `noc_ring1_out_util_pct` | `mean over L1_1_NOC_RING1_OUTGOING_0/1 / l1 cycles` | Primary ring-1 outgoing channels only. |
| NOC Ring 1 Incoming Util (%) | `noc_ring1_in_util_pct` | `mean over L1_1_NOC_RING1_INCOMING_0/1 / l1 cycles` | Primary ring-1 incoming channels only. |
| NOC Ring 0 Grant Efficiency (%) | `noc_ring0_grant_eff_pct` | `sum of grants / sum of requests over the ring-0 ports with both counters captured` | Ring-0 requests that were granted. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| Unpacker L1 Efficiency (ratio) | `unpacker_l1_eff_ratio` | `L1_0_UNPACKER_0_GRANT / UNPACK0_BUSY_THREAD0` | UNBOUNDED ratio (cross-domain); above 1 = ample L1 bandwidth for the unpacker. |
| Packer L1 Efficiency (ratio) | `packer_l1_eff_ratio` | `L1_0_PORT1_GRANT / PACKER_BUSY` | UNBOUNDED ratio; Wormhole only, where port 1 carries pack1 traffic. N/A on Blackhole (port 1 is an unpacker there). |
| L1 Unpacker Backpressure (%) | `l1_unpacker_backpressure_pct` | `1 - L1_0_UNPACKER_0_GRANT / L1_0_UNPACKER_0` | Unpacker-0 requests not accepted by the L1 arbiter. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| L1 Port 1 Backpressure (%) | `l1_port1_backpressure_pct` | `1 - port 1 grants / port 1 requests` | Port-1 requests not accepted by the L1 arbiter; the request and grant counters are the per-architecture pair named under L1 Port 1 Util. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| NOC Ring 0 Outgoing Backpressure (%) | `noc_ring0_out_backpressure_pct` | `1 - grants / requests over the primary ring-0 outgoing pair` | Outgoing ring-0 requests not accepted by the L1 arbiter. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| NOC Ring 0 Incoming Backpressure (%) | `noc_ring0_in_backpressure_pct` | `1 - grants / requests over the primary ring-0 incoming pair` | Incoming ring-0 requests not accepted by the L1 arbiter. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| NOC Ring 1 Outgoing Backpressure (%) | `noc_ring1_out_backpressure_pct` | `1 - grants / requests over the primary ring-1 outgoing pair` | Outgoing ring-1 requests not accepted by the L1 arbiter. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| NOC Ring 1 Incoming Backpressure (%) | `noc_ring1_in_backpressure_pct` | `1 - grants / requests over the primary ring-1 incoming pair` | Incoming ring-1 requests not accepted by the L1 arbiter. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| NOC Ring 1 Grant Efficiency (%) | `noc_ring1_grant_eff_pct` | `sum of grants / sum of requests over the ring-1 ports with both counters captured` | Ring-1 requests that were granted. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| L1 Unpacker1 Ext Backpressure (%) | `l1_unpacker1_ext_backpressure_pct` | `1 - grants / requests over unpacker 1's extended read ports` | Contention on unpacker 1's extended interfaces. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| L1 Unpacker0 Ext Backpressure (%) | `l1_unpacker0_ext_backpressure_pct` | `1 - grants / requests over unpacker 0's extended read ports` | Blackhole only; N/A on Wormhole. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| L1 Packer Interfaces Backpressure (%) | `l1_ext_pack_backpressure_pct` | `1 - grants / requests over the packer interface ports (26-34)` | Blackhole only; N/A on Wormhole. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |

### L1 and NoC composites

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| L1 Total Bandwidth Util (%) | `l1_total_bw_pct` | `sum of the primary L1_0 port counters / (8 x l1 cycles)` | Aggregate L1 bank-0 bandwidth utilization. |
| L1 Read vs Write Ratio (%) | `l1_read_write_ratio_pct` | `reads / (reads + writes), reads = unpacker0 + ring0 out, writes = ring0 in; port 1 counts as a read on Blackhole (unpacker 1) and a write on Wormhole (pack1)` | Above 50% = read-dominated. |
| NOC Ring 0 Asymmetry (%) | `noc_ring0_asymmetry_pct` | `ring0 outgoing / (outgoing + incoming)` | Direction balance of ring-0 traffic. |
| TDMA vs NOC L1 Share (%) | `tdma_vs_noc_l1_share_pct` | `TDMA bundle / (bundle + ring0 traffic)` | Firmware traffic as a share of L1 bank-0 activity. |
| L1 Contention Index (%) | `l1_contention_index_pct` | `mean of (1 - grant/request) over the five primary request/grant pairs` | One number for L1 bank-0 contention. The grant counter is the L1 arbiter accept for the port, so grant <= request by construction; clamped to 0..100 only as a guard. |
| NOC vs Compute Balance (%) | `noc_vs_compute_balance_pct` | `ring0 traffic / (ring0 traffic + FPU_COUNTER)` | Above 50% = NoC-bound, below = compute-bound. |

### Per-class availability, unpacker busy and issue-ready ratios

The (class, thread) availability pairs, per-unpacker busy splits and issue-ready ratios that the sections above do not already carry. They compute wherever the counter exists: on Wormhole and Blackhole for threads 0-2 and unpackers 0-1, on Quasar for all four threads.

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| CFG Instrn Avail Rate T1 (%) | `cfg_instrn_avail_t1_pct` | `CFG_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is a CFG instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). |
| CFG Instrn Avail Rate T2 (%) | `cfg_instrn_avail_t2_pct` | `CFG_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is a CFG instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). |
| SYNC Instrn Avail Rate T1 (%) | `sync_instrn_avail_t1_pct` | `SYNC_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is a SYNC instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). |
| SYNC Instrn Avail Rate T2 (%) | `sync_instrn_avail_t2_pct` | `SYNC_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is a SYNC instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). |
| THCON Instrn Avail Rate T1 (%) | `thcon_instrn_avail_t1_pct` | `THCON_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is a THCON instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). |
| THCON Instrn Avail Rate T2 (%) | `thcon_instrn_avail_t2_pct` | `THCON_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is a THCON instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). |
| MATH Instrn Avail Rate T0 (%) | `math_instrn_avail_t0_pct` | `MATH_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a MATH instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| MATH Instrn Avail Rate T2 (%) | `math_instrn_avail_t2_pct` | `MATH_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is a MATH instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). |
| UNPACK Instrn Avail Rate T1 (%) | `unpack_instrn_avail_t1_pct` | `UNPACK_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is a UNPACK instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). |
| UNPACK Instrn Avail Rate T2 (%) | `unpack_instrn_avail_t2_pct` | `UNPACK_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is a UNPACK instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). |
| PACK Instrn Avail Rate T0 (%) | `pack_instrn_avail_t0_pct` | `PACK_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is a PACK instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). |
| PACK Instrn Avail Rate T1 (%) | `pack_instrn_avail_t1_pct` | `PACK_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is a PACK instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). |
| Unpacker0 Busy T0 Util (%) | `unpack0_busy_t0_pct` | `UNPACK0_BUSY_THREAD0 / unpack cycles` | Unpacker 0 busy on behalf of thread 0. |
| Unpacker1 Busy T0 Util (%) | `unpack1_busy_t0_pct` | `UNPACK1_BUSY_THREAD0 / unpack cycles` | Unpacker 1 busy on behalf of thread 0. |
| Unpacker0 Busy T1 Util (%) | `unpack0_busy_t1_pct` | `UNPACK0_BUSY_THREAD1 / unpack cycles` | Unpacker 0 busy on behalf of thread 1. |
| Unpacker1 Busy T1 Util (%) | `unpack1_busy_t1_pct` | `UNPACK1_BUSY_THREAD1 / unpack cycles` | Unpacker 1 busy on behalf of thread 1. N/A on Quasar, which does not wire UNPACK1_BUSY_THREAD1. |
| Math Src Data Ready Rate (%) | `math_src_data_ready_pct` | `MATH_SRC_DATA_READY / unpack cycles` | Cycles a math ALU instruction was valid with both source registers ready. |
| FPU SFPU Overlap (%) | `fpu_sfpu_overlap_pct` | `max(0, FPU_COUNTER + SFPU_COUNTER - MATH_COUNTER) / fpu cycles` | Cycles the FPU and SFPU were both active. |
| T0 Instrn Per Issue-Ready Cycle (ratio) | `thread0_instrn_per_ready_cycle_ratio` | `THREAD_INSTRUCTIONS_0 / max(1, instrn cycles - THREAD_STALLS_0)` | UNBOUNDED ratio: instructions issued per cycle thread 0 was not stalled. |
| T1 Instrn Per Issue-Ready Cycle (ratio) | `thread1_instrn_per_ready_cycle_ratio` | `THREAD_INSTRUCTIONS_1 / max(1, instrn cycles - THREAD_STALLS_1)` | UNBOUNDED ratio: instructions issued per cycle thread 1 was not stalled. |
| T2 Instrn Per Issue-Ready Cycle (ratio) | `thread2_instrn_per_ready_cycle_ratio` | `THREAD_INSTRUCTIONS_2 / max(1, instrn cycles - THREAD_STALLS_2)` | UNBOUNDED ratio: instructions issued per cycle thread 2 was not stalled. |

### Quasar only

Counters only Quasar's NEOs expose: a fourth thread, the INSTISSUE instruction class, fifteen stall reasons OR-reduced across the threads and a third unpacker. N/A on Wormhole and Blackhole.

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Thread 3 Stall Rate (%) | `thread3_stall_pct` | `THREAD_STALLS_3 / instrn cycles` | Fraction of cycles thread 3 was stalled. Quasar only (fourth thread). |
| T3 Instrn Issue Rate (%) | `thread3_ipc_pct` | `THREAD_INSTRUCTIONS_3 / instrn cycles` | Instructions issued per cycle on thread 3, as a percentage. Quasar only. |
| CFG Instrn Avail Rate T3 (%) | `cfg_instrn_avail_t3_pct` | `CFG_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is a CFG instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| SYNC Instrn Avail Rate T3 (%) | `sync_instrn_avail_t3_pct` | `SYNC_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is a SYNC instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| THCON Instrn Avail Rate T3 (%) | `thcon_instrn_avail_t3_pct` | `THCON_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is a THCON instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T0 (%) | `instissue_instrn_avail_t0_pct` | `INSTISSUE_INSTRN_AVAILABLE_0 / instrn cycles` | Cycles the head instruction on thread 0 is an INSTISSUE instruction not blocked by its unit (a blocked head counts in Thread 0 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T1 (%) | `instissue_instrn_avail_t1_pct` | `INSTISSUE_INSTRN_AVAILABLE_1 / instrn cycles` | Cycles the head instruction on thread 1 is an INSTISSUE instruction not blocked by its unit (a blocked head counts in Thread 1 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T2 (%) | `instissue_instrn_avail_t2_pct` | `INSTISSUE_INSTRN_AVAILABLE_2 / instrn cycles` | Cycles the head instruction on thread 2 is an INSTISSUE instruction not blocked by its unit (a blocked head counts in Thread 2 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T3 (%) | `instissue_instrn_avail_t3_pct` | `INSTISSUE_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is an INSTISSUE instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| MATH Instrn Avail Rate T3 (%) | `math_instrn_avail_t3_pct` | `MATH_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is a MATH instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| UNPACK Instrn Avail Rate T3 (%) | `unpack_instrn_avail_t3_pct` | `UNPACK_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is a UNPACK instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| PACK Instrn Avail Rate T3 (%) | `pack_instrn_avail_t3_pct` | `PACK_INSTRN_AVAILABLE_3 / instrn cycles` | Cycles the head instruction on thread 3 is a PACK instruction not blocked by its unit (a blocked head counts in Thread 3 Stall Rate). Quasar only (four threads; the MATH class also counts instissue instructions). |
| Tile Counter Stall Pack Rate (%) | `tile_counter_stall_pack_pct` | `TILE_COUNTER_STALL_PACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Tile Counter Stall Unpack Rate (%) | `tile_counter_stall_unpack_pct` | `TILE_COUNTER_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Srcs Stall Pack Rate (%) | `srcs_stall_pack_pct` | `SRCS_STALL_PACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Srcs Stall SFPU Rate (%) | `srcs_stall_sfpu_pct` | `SRCS_STALL_SFPU / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Srcs Stall Unpack Rate (%) | `srcs_stall_unpack_pct` | `SRCS_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall Pack Rate (%) | `dest_stall_pack_pct` | `DEST_STALL_PACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall SFPU Rate (%) | `dest_stall_sfpu_pct` | `DEST_STALL_SFPU / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall Math Rate (%) | `dest_stall_math_pct` | `DEST_STALL_MATH / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall Unpack Rate (%) | `dest_stall_unpack_pct` | `DEST_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SFPU Data Hazard Stall Rate (%) | `sfpu_data_hazard_stall_pct` | `SFPU_DATA_HAZARD_STALL / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads; the RTL folds dest data-valid stalls into it, so it overlaps the Dest Stall rows. Quasar only. |
| FPU Data Hazard Stall Rate (%) | `fpu_data_hazard_stall_pct` | `FPU_DATA_HAZARD_STALL / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads; the RTL folds dest data-valid stalls into it, so it overlaps the Dest Stall rows. Quasar only. |
| SrcB Stall Unpack Rate (%) | `srcb_stall_unpack_pct` | `SRCB_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SrcA Stall Unpack Rate (%) | `srca_stall_unpack_pct` | `SRCA_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Src Valid Stall Math Rate (%) | `dvalid_stall_math_pct` | `DVALID_STALL_MATH / instrn cycles` | Cycles math waited for srcA or srcB to become valid, OR-reduced across the four threads; SrcA Stall Math Rate is the srcA part and SrcB Stall Math Rate the remainder. Quasar only. |
| SrcA Stall Math Rate (%) | `srca_stall_math_pct` | `SRCA_STALL_MATH / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SrcB Stall Math Rate (%) | `srcb_stall_math_pct` | `(DVALID_STALL_MATH - SRCA_STALL_MATH) / instrn cycles` | The srcB part of the src-valid stall (the RTL counts srcA and the total so srcB can be recovered). Quasar only. |
| Tile Counter Stall Pack Share (%) | `tile_counter_stall_pack_share_pct` | `TILE_COUNTER_STALL_PACK / sum of the captured stall reasons` | Share of tile counter stall pack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Tile Counter Stall Unpack Share (%) | `tile_counter_stall_unpack_share_pct` | `TILE_COUNTER_STALL_UNPACK / sum of the captured stall reasons` | Share of tile counter stall unpack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Srcs Stall Pack Share (%) | `srcs_stall_pack_share_pct` | `SRCS_STALL_PACK / sum of the captured stall reasons` | Share of srcs stall pack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Srcs Stall SFPU Share (%) | `srcs_stall_sfpu_share_pct` | `SRCS_STALL_SFPU / sum of the captured stall reasons` | Share of srcs stall sfpu among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Srcs Stall Unpack Share (%) | `srcs_stall_unpack_share_pct` | `SRCS_STALL_UNPACK / sum of the captured stall reasons` | Share of srcs stall unpack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall Pack Share (%) | `dest_stall_pack_share_pct` | `DEST_STALL_PACK / sum of the captured stall reasons` | Share of dest stall pack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall SFPU Share (%) | `dest_stall_sfpu_share_pct` | `DEST_STALL_SFPU / sum of the captured stall reasons` | Share of dest stall sfpu among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall Math Share (%) | `dest_stall_math_share_pct` | `DEST_STALL_MATH / sum of the captured stall reasons` | Share of dest stall math among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall Unpack Share (%) | `dest_stall_unpack_share_pct` | `DEST_STALL_UNPACK / sum of the captured stall reasons` | Share of dest stall unpack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| SFPU Data Hazard Stall Share (%) | `sfpu_data_hazard_stall_share_pct` | `SFPU_DATA_HAZARD_STALL / sum of the captured stall reasons` | Share of sfpu data hazard stall among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| FPU Data Hazard Stall Share (%) | `fpu_data_hazard_stall_share_pct` | `FPU_DATA_HAZARD_STALL / sum of the captured stall reasons` | Share of fpu data hazard stall among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| SrcB Stall Unpack Share (%) | `srcb_stall_unpack_share_pct` | `SRCB_STALL_UNPACK / sum of the captured stall reasons` | Share of srcb stall unpack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| SrcA Stall Unpack Share (%) | `srca_stall_unpack_share_pct` | `SRCA_STALL_UNPACK / sum of the captured stall reasons` | Share of srca stall unpack among the OR-reduced stall reasons (src-valid counted once, as its srcA and srcB parts); N/A with fewer than two reasons captured. Quasar only. |
| SrcA Stall Math Share (%) | `srca_stall_math_share_pct` | `SRCA_STALL_MATH / sum of the captured stall reasons` | Share of srca stall math among the OR-reduced stall reasons (the src-valid total is replaced by its srcA and srcB parts so nothing is counted twice); N/A with fewer than two reasons captured. Quasar only. |
| SrcB Stall Math Share (%) | `srcb_stall_math_share_pct` | `(DVALID_STALL_MATH - SRCA_STALL_MATH) / sum of the captured stall reasons` | Share of the srcB part of the src-valid stall; N/A with fewer than two reasons captured. Quasar only. |
| Unpacker2 Busy T0 Util (%) | `unpack2_busy_t0_pct` | `UNPACK2_BUSY_THREAD0 / unpack cycles` | Unpacker 2 busy on behalf of thread 0. Quasar only. |
| T3 Instrn Per Issue-Ready Cycle (ratio) | `thread3_instrn_per_ready_cycle_ratio` | `THREAD_INSTRUCTIONS_3 / max(1, instrn cycles - THREAD_STALLS_3)` | UNBOUNDED ratio: instructions issued per cycle thread 3 was not stalled. Quasar only. |

### L1 client events (Quasar)

Quasar has no L1 counter bank. Each NEO has one l1_client event counter behind a 37 subport by 8 event mux, so a run measures one selection per NEO, chosen with `TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL` (metal) or `LLK_PERF_L1_CLIENT_SEL` (tt-llk harness). Its metric is not a fixed catalogue row: `tt_llk_perf.metrics.quasar_l1_client_label(sel)` names the counter `L1_CLIENT_<PORT>_<EVENT>`, `compute_l1_client_metrics()` divides the count by the capture's cycles, and the column appears after the fixed metric columns as `<name lower-cased>_pct` (label `<name> Rate`), or `_ratio` (label `<name> Mean Outstanding`) for the pending-request carry. The Tracy CSV and the harness report both add these columns dynamically.

`<PORT>` is `TRISC<n>` (subports 0-3), `THCON` (4), `UNPACK<u>_IF<i>_LANE<l>` (5-24: three unpackers, two interfaces each except unpacker 2 with one, four lanes) or `PACK<p>_IF<i>_LANE<l>` (25-36: packer 0 interfaces 0-1, packer 1 interface 0). Events 1 to 3 count per SBank of the whole port, not per lane, so their names carry `SBANK<n>` instead of `LANE<n>` (`L1_CLIENT_UNPACK0_IF0_SBANK0_SBANK_POP` for selection 41). Event 0 is unused and the THCON subport's events 1 to 3 duplicate the TRISC port's SBank 0 counters; those 40 selections are rejected at build time, leaving 256 valid ones.

| Event | Name | Value reported |
|---|---|---|
| 1 | `SBANK_POP` | per-cycle indicator; count / cycles is a rate |
| 2 | `ISSUE_STALL_CARRY` | fires once per four lane events; count / cycles is the mean per-lane fraction of stalled issue cycles |
| 3 | `ISSUE_WORK_CARRY` | same carry; mean per-lane fraction of issuing cycles |
| 4 | `FLEX_STALL_CARRY` | same carry, flex path |
| 5 | `FLEX_WORK_CARRY` | same carry, flex path |
| 6 | `PENDING_REQS_CARRY` | fires once per 64 outstanding-request cycles (128 on packer 0's two interfaces); count x divisor / cycles is the mean number of outstanding requests, reported as a ratio |
| 7 | `ORDER_FIFO_ACTIVE` | per-cycle indicator; count / cycles is a rate |

The reference cycles differ between the two consumers: the metal profiler stamps the wall-clock span between arm and freeze (the CSR has no reference counter), the harness uses the INSTRN bank's cycle count of the zone.

## Hardware Register Reference

Each counter bank `<X>` (`FPU`, `TDMA_PACK`, `TDMA_UNPACK`, `L1`, `INSTRN_THREAD`) is programmed via three RISC-V debug registers. The addresses and bit constants are in `tt_metal/tt-llk/tools/include/perf_counters/registers.h` (`bank_regs(Bank)` returns the five registers of a bank; the arch `tensix.h` macros are cross-checked by `static_assert`), and the write sequences used by `start_perf_counter()` / `stop_perf_counter()` / `read_perf_counters()` are the primitives in `hw.h`.

### Control registers (`RISCV_DEBUG_REG_PERF_CNT_<X>0..2`)

| Register | Field | Description |
|---|---|---|
| `RISCV_DEBUG_REG_PERF_CNT_<X>0` | all | Reference period in cycles. |
| `RISCV_DEBUG_REG_PERF_CNT_<X>1` | Bits [7:0] | Mode: `0` = continuous, `1` = count until refclk cycles hit, `2` = continuous (no refclk maintenance). |
| `RISCV_DEBUG_REG_PERF_CNT_<X>1` | Bits [8+:N], N = 1 (FPU), 3 (L1), 5 (TDMA), 6 (INSTRN) | Counter select within the bank. |
| `RISCV_DEBUG_REG_PERF_CNT_<X>1` | Bit [16] | Output format: `0` = req count on `_OUT_H_<X>`, `1` = grant count. |
| `RISCV_DEBUG_REG_PERF_CNT_<X>2` | Bit [0] | Start (rising edge only; 0→1 transition also clears the counters). |
| `RISCV_DEBUG_REG_PERF_CNT_<X>2` | Bit [1] | Stop (rising edge only). |

### Data registers

| Register | Value |
|---|---|
| `RISCV_DEBUG_REG_PERF_CNT_OUT_L_<X>` | `ref_cnt` (elapsed cycles between start and stop). |
| `RISCV_DEBUG_REG_PERF_CNT_OUT_H_<X>` | `req_cnt` if control bit [16]=0, otherwise `grant_cnt`. |

Because the software must toggle bit [16] and re-read to get both `req` and `grant`, each counter is read twice (the grant entries in the tables carry selects of 256 and up, which is bit [16] after the shift). Every mode-register write is followed by a readback poll (`llk::perf::select`) so the hardware has committed the new selection before the output registers are sampled; `volatile` reads alone do not provide MMIO ordering guarantees on RISC-V. The tt-llk harness bounds the poll at 1024 reads; the BRISC firmware polls without a bound because it sits a few bytes under its size limit.

### Quasar windows

The same `tt_perf_cnt` protocol applies, but the debug block is per NEO and has two addresses. `registers.h` keeps the per-bank offsets once (`detail::BANK_OFFSETS`) and adds them to a window base: `bank_regs(bank, window)`, `perf_cnt_all(window)`, `dbg_feature_disable(window)` and `l1_client_regs(window)`, with `LOCAL_REGS_WINDOW` as the default.

| Window | Base | Who uses it |
|---|---|---|
| Local | 0x00800000 (`LOCAL_REGS_WINDOW`, the `LOCAL_REGS_BASE` macro of `tt_t6_trisc_map.h`) | a TRISC reaching its own NEO (the tt-llk harness) |
| NoC | 0x01800000 + neo x 0x10000 (`neo_window(neo)`, `NUM_NEOS` = 4) | DM0 reaching every NEO (the metal profiler) |

| Bank or register | Offsets in the window | Notes |
|---|---|---|
| INSTRN_THREAD | control 0x0, 0x4, 0x8; OUT_L 0x98, OUT_H 0x9C | started and stopped by `PERF_CNT_ALL` too |
| TDMA_UNPACK | control 0xC, 0x10, 0x14; OUT_L 0xA0, OUT_H 0xA4 | own start/stop only |
| FPU | control 0x18, 0x1C, 0x20; OUT_L 0xB0, OUT_H 0xB4 | started and stopped by `PERF_CNT_ALL` too |
| TDMA_PACK | control 0x8C, 0x90, 0x94; OUT_L 0xA8, OUT_H 0xAC | own start/stop only |
| L1 | none | no L1 bank; `bank_regs(Bank::L1)` is all zero and `table_for(Bank::L1)` is empty |
| `PERF_CNT_ALL` | 0x24 | global start/stop for INSTRN_THREAD and FPU |
| `DBG_FEATURE_DISABLE` | 0x40 | cleared by the harness before arming |
| `PERF_CNT_MUX_CTRL` | 0xF4 | present but unused (no L1 bank) |
| l1_client `CTRL` | 0xA0AC | bit 0 enable, bits [9:4] subport, bits [14:12] event (`l1_client_ctrl_word(sel)` in `hw.h`) |
| l1_client `CNT` | 0xA0B0 | clear-on-read event count |

The offsets are `static_assert`ed against `NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_*_REG_ADDR` and the l1_client macros of `tensix_neo_reg.h` whenever that header is visible. The l1_client sequence in `hw.h` is `l1_client_start(regs, sel)` (write `CTRL`, read `CNT` once to clear it), `l1_client_read(regs)` and `l1_client_stop(regs)` (write `CTRL` 0).

---

## Hardware Limitations

### Not Accessible from Software

| Feature | Architecture | Reason | Workaround |
|---------|-------------|--------|------------|
| `stall_cnt` (bits [127:96]) | BH | `out_fmt` is 1-bit, no mode to route bits [127:96] | Software derives stall as `req - grant` |
| DDR5 RISC L1 counters | BH | Counters instantiated but no debug register interface | None (requires an RTL change) |

### Counter Set

Verified against the `wormhole_rtl` and `blackhole_rtl` branches and, for `quasar.h`, the Quasar A0 tapeout RTL. Every counter in the `blackhole.h` / `wormhole.h` / `quasar.h` tables is driven by a real RTL signal. Signals that are hardwired to a constant, or whose grant/req line is an alias of another counter already exposed, are omitted from the tables entirely. No post-hoc filtering is applied; every emitted counter is reported as-is.

Some counters are live wires that no tt-metal op has exercised so far. In a sweep of every selector over 22 Blackhole ops these read 0 throughout: the TRISC1 unpack path (`UNPACK0/1_BUSY_THREAD1`, `SRCA/SRCB_WRITE_TID_ODD`), the MOVE class, `THCON_INSTRN_AVAILABLE_1`, `UNPACK_INSTRN_AVAILABLE_1/2`, `PACK_INSTRN_AVAILABLE_0/1`, `WAITING_FOR_SRCA/SRCB_CLEAR` and the per-thread waits a thread never performs (for example `WAITING_FOR_SFPU_IDLE_0`). They stay in the tables because other kernels can drive them (the LLK perf suite does hit `WAITING_FOR_SRCA_CLEAR`); metrics built on them read 0%, not N/A. On Quasar the emulator sweep over 23 ops left 24 entries at zero: everything on thread 3, the THCON class, `CFG_INSTRN_AVAILABLE_1`, `UNPACK_INSTRN_AVAILABLE_1/2`, `PACK_INSTRN_AVAILABLE_0/1`, `SRCS_STALL_PACK/SFPU/UNPACK`, `UNPACK2_BUSY_THREAD0`, `UNPACK0_BUSY_THREAD1` and `SRCA/SRCB_WRITE_TID_ODD`.
