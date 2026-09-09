# Hardware Performance Counters

## Quick Links
- User guide: [docs/source/ttnn/ttnn/profiling_ttnn_operations.rst](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst)
- Firmware implementation: [tt_metal/tools/profiler/perf_counters.hpp](../../tt_metal/tools/profiler/perf_counters.hpp)
- Python analysis: [tools/tracy/perf_counter_analysis.py](../../tools/tracy/perf_counter_analysis.py)
- Metric formulas (shared with the tt-llk harness): [tools/tracy/perf_metrics_common.py](../../tools/tracy/perf_metrics_common.py)
- CSV writer: [tools/tracy/process_ops_logs.py](../../tools/tracy/process_ops_logs.py)

## Overview

Each Tensix core contains hardware performance counters organized into five banks (FPU, TDMA_UNPACK, TDMA_PACK, INSTRN_THREAD, L1). These counters measure cycle-level events: how many cycles the FPU was active, how many cycles a thread was stalled, how many cycles each L1 memory port had traffic, and so on.

The counters are built from a reusable RTL module (`tt_perf_cnt`) that provides three values per event: **req_cnt** (cycles the event signal was high), **grant_cnt** (cycles the grant/ready signal was high), and **ref_cnt** (total elapsed cycles). From these raw values, the profiler computes derived metrics like utilization (`req_cnt / ref_cnt`), backpressure (`(req_cnt - grant_cnt) / req_cnt`), and cross-bank ratios that combine counters from different banks.

## How It Works

1. **Kernel starts**: TRISC1 calls `start_perf_counter()` which writes the start bit to all enabled counter banks. All counters begin accumulating from zero.

2. **Kernel runs**: While the kernel executes, each counter increments every cycle its input signal is high. All counters within a bank run simultaneously — there is no multiplexing during measurement.

3. **Kernel ends**: TRISC1 calls `stop_perf_counter()` which freezes all counters. The counter values remain latched in the debug registers.

4. **BRISC reads counters**: After all TRISCs complete (`wait_ncrisc_trisc()`), BRISC calls `read_perf_counters()` which loops over enabled counter groups, reading each counter by cycling through `counter_sel` values. Each counter value is packed into a 64-bit profiler marker and written to BRISC's profiler buffer. Before each group (starting from the 2nd), BRISC calls `perf_counter_flush()` to push the buffer to DRAM, ensuring each group starts with a fresh buffer. TRISCs cannot do this because they have no NOC access.

5. **Host reads**: After the kernel completes, the host reads the profiler data from DRAM and decodes each marker into a counter type, value, and reference count.

6. **Python processes**: `perf_counter_analysis.py` decodes the markers and computes derived metrics per operation and core through the shared formula module `perf_metrics_common.py` (also used by the tt-llk test harness, so both report the same numbers from the same counters). Results are written to CSV and printed to console.

### How to Run

```bash
# Capture every counter group; needs several passes, so opt in to the workload replay
python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all \
    -m "pytest your_test.py -x -v"
```

With `--perf-counter-multipass` a request is split into passes (at most three groups and one L1 bank per pass) and `all` expands to the architecture's full group set.

Available counter groups for `--profiler-capture-perf-counters`: `fpu`, `pack`, `unpack`, `l1_0`, `l1_1`, `instrn`, `all`. Blackhole also supports `l1_2`, `l1_3`, `l1_4`, `l1_5`; `all` expands to the running architecture's full set.

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

### Architecture Summary

The per-architecture inventory is the set of `hw_counters.h` tables (`tt_metal/hw/inc/internal/tt-1xx/<arch>/`); the derived-metric catalogue below is shared, and a metric whose counters exist on only one architecture reports N/A on the other.

**Wormhole** has `PACK_COUNT=4` (4 packer engines), active `o_math_instrnbuf_rden`, and all TDMA counters live. The L1 mux is 1-bit (2 positions: ports 0-7 and 8-15).

**Blackhole** has fewer raw TDMA counters because `PACK_COUNT=1` ties the per-engine busy and dest-read signals for engines 1-3 to constants. Only RTL-live signals are read from hardware. Any counter whose RTL signal is hardwired to a constant has been omitted from the `hw_counters.h` arrays, and any aliased grant counter is consolidated to one canonical entry. `Math-to-Pack Handoff Efficiency` falls back to the bank's reference cycles as denominator when `PACKER_BUSY` is 0 for a given workload (e.g. pure-SFPU ops that don't drive the packer); `Packer Efficiency` reports N/A there. TDMA_UNPACK grant banks 4-6 (sels 260-262) have identical RTL wiring on WH and BH (verified: srcB port, srcA overwrite, srcA port). Blackhole has more L1 mux positions (5 vs 2 for Tensix, 4 vs 1 for Ethernet).

**INSTRN_THREAD bank** — `perf_cnt_instrn_thread` is built from a Verilog generate array in `tt_instruction_thread.sv` and has architecture-specific counter_sel mappings. Req-side: sels 0-23 are per-thread instruction-type availability (CFG/SYNC/THCON/MOVE/FPU/UNPACK/PACK, 3 threads each), sels 24-26 are per-thread total stall cycles, and sels 27+ are stall reasons. On WH the shared stall conditions (SRCA/B clear/valid) are replicated across 3 slots each (sels 27-38); on BH they occupy 1 slot each (sels 27-30). Per-thread stall reasons are thread-major: WH sels 39-65 (9 types × 3 threads), BH sels 31-57. Grant-side: the RTL wires grant as `{8{ibuffer_rden[th]}}` per instance and `{9{inst_stall_thread[th]}}` per per-thread stall-reason instance, so the 24 possible issue-count sels collapse to 3 distinct per-thread values and the per-thread stall-reason grants reproduce `THREAD_STALLS_{th}`. We expose only the distinct grants: `THREAD_INSTRUCTIONS_{0,1,2}` at sels 256/264/272 (one per instance) and `ANY_THREAD_STALL` at sel 283. The counter arrays are in arch-specific `hw_counters.h` files; `perf_counters.hpp` is arch-agnostic (WH defines empty L1_2/3/4 arrays).

---

## Derived Metrics Reference

Every derived metric is computed by one shared module, [tools/tracy/perf_metrics_common.py](../../tools/tracy/perf_metrics_common.py). The Tracy tool computes it per operation and core and aggregates to Min/Median/Max/Avg across cores; the tt-llk test harness computes it per zone and run and aggregates to mean/std across runs. The tables below are the complete set. The module is the source of truth, and a unit test (`tests/ttnn/tracy/test_perf_metrics_common.py`) fails if this file stops listing a metric the module computes.

Metrics come in two families, told apart by the key suffix:

- `*_pct`: percentages bounded to 0-100 by construction: the numerator counts a subset of the cycles or events the denominator counts (a 1-bit counter over its bank's `ref_cnt`, an RTL-proven subset such as an arbiter accept over its request, or a share of a sum). Where a row says "clamped", the value is additionally clamped to 0..100 and the row states why.
- `*_ratio`: unbounded raw ratios that can exceed 1.0 by design, because the numerator and denominator come from different measurement domains or because overlapping events are summed. Reported with a `(ratio)` unit and never clamped; the excess over 1.0 is the signal.

A metric whose counters do not exist on the running architecture reports N/A (blank), never 0: the Wormhole-only per-engine packer metrics are N/A on Blackhole, and the Blackhole-only extended L1 groups are N/A on Wormhole. Cross-bank metrics are likewise N/A when one of their counter groups was not captured in the run.

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
| CFG Instrn Avail Rate T1 (%) | `cfg_instrn_avail_t1_pct` | `CFG_INSTRN_AVAILABLE_1 / instrn cycles` | CFG instructions pending on thread 1. |
| CFG Instrn Avail Rate T2 (%) | `cfg_instrn_avail_t2_pct` | `CFG_INSTRN_AVAILABLE_2 / instrn cycles` | CFG instructions pending on thread 2. |
| SYNC Instrn Avail Rate T1 (%) | `sync_instrn_avail_t1_pct` | `SYNC_INSTRN_AVAILABLE_1 / instrn cycles` | SYNC instructions pending on thread 1. |
| SYNC Instrn Avail Rate T2 (%) | `sync_instrn_avail_t2_pct` | `SYNC_INSTRN_AVAILABLE_2 / instrn cycles` | SYNC instructions pending on thread 2. |
| THCON Instrn Avail Rate T1 (%) | `thcon_instrn_avail_t1_pct` | `THCON_INSTRN_AVAILABLE_1 / instrn cycles` | THCON instructions pending on thread 1. |
| THCON Instrn Avail Rate T2 (%) | `thcon_instrn_avail_t2_pct` | `THCON_INSTRN_AVAILABLE_2 / instrn cycles` | THCON instructions pending on thread 2. |
| MATH Instrn Avail Rate T0 (%) | `math_instrn_avail_t0_pct` | `MATH_INSTRN_AVAILABLE_0 / instrn cycles` | MATH instructions pending on thread 0. |
| MATH Instrn Avail Rate T2 (%) | `math_instrn_avail_t2_pct` | `MATH_INSTRN_AVAILABLE_2 / instrn cycles` | MATH instructions pending on thread 2. |
| UNPACK Instrn Avail Rate T1 (%) | `unpack_instrn_avail_t1_pct` | `UNPACK_INSTRN_AVAILABLE_1 / instrn cycles` | UNPACK instructions pending on thread 1. |
| UNPACK Instrn Avail Rate T2 (%) | `unpack_instrn_avail_t2_pct` | `UNPACK_INSTRN_AVAILABLE_2 / instrn cycles` | UNPACK instructions pending on thread 2. |
| PACK Instrn Avail Rate T0 (%) | `pack_instrn_avail_t0_pct` | `PACK_INSTRN_AVAILABLE_0 / instrn cycles` | PACK instructions pending on thread 0. |
| PACK Instrn Avail Rate T1 (%) | `pack_instrn_avail_t1_pct` | `PACK_INSTRN_AVAILABLE_1 / instrn cycles` | PACK instructions pending on thread 1. |
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
| CFG Instrn Avail Rate T3 (%) | `cfg_instrn_avail_t3_pct` | `CFG_INSTRN_AVAILABLE_3 / instrn cycles` | CFG instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| SYNC Instrn Avail Rate T3 (%) | `sync_instrn_avail_t3_pct` | `SYNC_INSTRN_AVAILABLE_3 / instrn cycles` | SYNC instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| THCON Instrn Avail Rate T3 (%) | `thcon_instrn_avail_t3_pct` | `THCON_INSTRN_AVAILABLE_3 / instrn cycles` | THCON instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T0 (%) | `instissue_instrn_avail_t0_pct` | `INSTISSUE_INSTRN_AVAILABLE_0 / instrn cycles` | INSTISSUE instructions pending on thread 0. Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T1 (%) | `instissue_instrn_avail_t1_pct` | `INSTISSUE_INSTRN_AVAILABLE_1 / instrn cycles` | INSTISSUE instructions pending on thread 1. Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T2 (%) | `instissue_instrn_avail_t2_pct` | `INSTISSUE_INSTRN_AVAILABLE_2 / instrn cycles` | INSTISSUE instructions pending on thread 2. Quasar only (four threads; the MATH class also counts instissue instructions). |
| INSTISSUE Instrn Avail Rate T3 (%) | `instissue_instrn_avail_t3_pct` | `INSTISSUE_INSTRN_AVAILABLE_3 / instrn cycles` | INSTISSUE instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| MATH Instrn Avail Rate T3 (%) | `math_instrn_avail_t3_pct` | `MATH_INSTRN_AVAILABLE_3 / instrn cycles` | MATH instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| UNPACK Instrn Avail Rate T3 (%) | `unpack_instrn_avail_t3_pct` | `UNPACK_INSTRN_AVAILABLE_3 / instrn cycles` | UNPACK instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| PACK Instrn Avail Rate T3 (%) | `pack_instrn_avail_t3_pct` | `PACK_INSTRN_AVAILABLE_3 / instrn cycles` | PACK instructions pending on thread 3. Quasar only (four threads; the MATH class also counts instissue instructions). |
| Tile Counter Stall Pack Rate (%) | `tile_counter_stall_pack_pct` | `TILE_COUNTER_STALL_PACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Tile Counter Stall Unpack Rate (%) | `tile_counter_stall_unpack_pct` | `TILE_COUNTER_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Srcs Stall Pack Rate (%) | `srcs_stall_pack_pct` | `SRCS_STALL_PACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Srcs Stall SFPU Rate (%) | `srcs_stall_sfpu_pct` | `SRCS_STALL_SFPU / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Srcs Stall Unpack Rate (%) | `srcs_stall_unpack_pct` | `SRCS_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall Pack Rate (%) | `dest_stall_pack_pct` | `DEST_STALL_PACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall SFPU Rate (%) | `dest_stall_sfpu_pct` | `DEST_STALL_SFPU / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall Math Rate (%) | `dest_stall_math_pct` | `DEST_STALL_MATH / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Dest Stall Unpack Rate (%) | `dest_stall_unpack_pct` | `DEST_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SFPU Data Hazard Stall Rate (%) | `sfpu_data_hazard_stall_pct` | `SFPU_DATA_HAZARD_STALL / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| FPU Data Hazard Stall Rate (%) | `fpu_data_hazard_stall_pct` | `FPU_DATA_HAZARD_STALL / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SrcB Stall Unpack Rate (%) | `srcb_stall_unpack_pct` | `SRCB_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SrcA Stall Unpack Rate (%) | `srca_stall_unpack_pct` | `SRCA_STALL_UNPACK / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| DValid Stall Math Rate (%) | `dvalid_stall_math_pct` | `DVALID_STALL_MATH / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| SrcA Stall Math Rate (%) | `srca_stall_math_pct` | `SRCA_STALL_MATH / instrn cycles` | Cycles the INSTRN unit reported this stall reason, OR-reduced across the four threads. Quasar only. |
| Tile Counter Stall Pack Share (%) | `tile_counter_stall_pack_share_pct` | `TILE_COUNTER_STALL_PACK / sum of the captured stall reasons` | Share of tile counter stall pack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Tile Counter Stall Unpack Share (%) | `tile_counter_stall_unpack_share_pct` | `TILE_COUNTER_STALL_UNPACK / sum of the captured stall reasons` | Share of tile counter stall unpack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Srcs Stall Pack Share (%) | `srcs_stall_pack_share_pct` | `SRCS_STALL_PACK / sum of the captured stall reasons` | Share of srcs stall pack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Srcs Stall SFPU Share (%) | `srcs_stall_sfpu_share_pct` | `SRCS_STALL_SFPU / sum of the captured stall reasons` | Share of srcs stall sfpu among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Srcs Stall Unpack Share (%) | `srcs_stall_unpack_share_pct` | `SRCS_STALL_UNPACK / sum of the captured stall reasons` | Share of srcs stall unpack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall Pack Share (%) | `dest_stall_pack_share_pct` | `DEST_STALL_PACK / sum of the captured stall reasons` | Share of dest stall pack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall SFPU Share (%) | `dest_stall_sfpu_share_pct` | `DEST_STALL_SFPU / sum of the captured stall reasons` | Share of dest stall sfpu among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall Math Share (%) | `dest_stall_math_share_pct` | `DEST_STALL_MATH / sum of the captured stall reasons` | Share of dest stall math among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Dest Stall Unpack Share (%) | `dest_stall_unpack_share_pct` | `DEST_STALL_UNPACK / sum of the captured stall reasons` | Share of dest stall unpack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| SFPU Data Hazard Stall Share (%) | `sfpu_data_hazard_stall_share_pct` | `SFPU_DATA_HAZARD_STALL / sum of the captured stall reasons` | Share of sfpu data hazard stall among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| FPU Data Hazard Stall Share (%) | `fpu_data_hazard_stall_share_pct` | `FPU_DATA_HAZARD_STALL / sum of the captured stall reasons` | Share of fpu data hazard stall among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| SrcB Stall Unpack Share (%) | `srcb_stall_unpack_share_pct` | `SRCB_STALL_UNPACK / sum of the captured stall reasons` | Share of srcb stall unpack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| SrcA Stall Unpack Share (%) | `srca_stall_unpack_share_pct` | `SRCA_STALL_UNPACK / sum of the captured stall reasons` | Share of srca stall unpack among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| DValid Stall Math Share (%) | `dvalid_stall_math_share_pct` | `DVALID_STALL_MATH / sum of the captured stall reasons` | Share of dvalid stall math among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| SrcA Stall Math Share (%) | `srca_stall_math_share_pct` | `SRCA_STALL_MATH / sum of the captured stall reasons` | Share of srca stall math among the 15 OR-reduced stall reasons; N/A with fewer than two reasons captured. Quasar only. |
| Unpacker2 Busy T0 Util (%) | `unpack2_busy_t0_pct` | `UNPACK2_BUSY_THREAD0 / unpack cycles` | Unpacker 2 busy on behalf of thread 0. Quasar only. |
| T3 Instrn Per Issue-Ready Cycle (ratio) | `thread3_instrn_per_ready_cycle_ratio` | `THREAD_INSTRUCTIONS_3 / max(1, instrn cycles - THREAD_STALLS_3)` | UNBOUNDED ratio: instructions issued per cycle thread 3 was not stalled. Quasar only. |


## Hardware Register Reference

Each counter bank `<X>` (`FPU`, `TDMA_PACK`, `TDMA_UNPACK`, `L1`, `INSTRN_THREAD`) is programmed via three RISC-V debug registers. The programming sequence in `start_perf_counter()` / `stop_perf_counter()` follows this map.

### Control registers (`RISCV_DEBUG_REG_PERF_CNT_<X>0..2`)

| Register | Field | Description |
|---|---|---|
| `RISCV_DEBUG_REG_PERF_CNT_<X>0` | — | Reference period in cycles. |
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

Because the software must toggle bit [16] and re-read to get both `req` and `grant`, each counter is read twice. The intermediate register writes are fenced by a readback poll in `read_single_group()` to ensure the hardware has committed the new mux selection before the output registers are sampled — `volatile` reads alone do not provide MMIO ordering guarantees on RISC-V.

---

## Hardware Limitations

### Not Accessible from Software

| Feature | Architecture | Reason | Workaround |
|---------|-------------|--------|------------|
| `stall_cnt` (bits [127:96]) | BH | `out_fmt` is 1-bit, no mode to route bits [127:96] | Software derives stall as `req - grant` |
| DDR5 RISC L1 counters | BH | Counters instantiated but no debug register interface | None — requires RTL change |

### Counter Set

Verified against the `wormhole_rtl` and `blackhole_rtl` branches. Every counter exposed via the `hw_counters.h` arrays is driven by a real RTL signal — signals that are hardwired to a constant, or whose grant/req line is an alias of another counter we already expose, are omitted from the arrays entirely. No post-hoc filtering is applied; every emitted counter is reported as-is.

Some counters are live wires that no tt-metal op has exercised so far. In a sweep of every selector over 22 Blackhole ops these read 0 throughout: the TRISC1 unpack path (`UNPACK0/1_BUSY_THREAD1`, `SRCA/SRCB_WRITE_TID_ODD`), the MOVE class, `THCON_INSTRN_AVAILABLE_1`, `UNPACK_INSTRN_AVAILABLE_1/2`, `PACK_INSTRN_AVAILABLE_0/1`, `WAITING_FOR_SRCA/SRCB_CLEAR` and the per-thread waits a thread never performs (for example `WAITING_FOR_SFPU_IDLE_0`). They stay in the tables because other kernels can drive them (the LLK perf suite does hit `WAITING_FOR_SRCA_CLEAR`); metrics built on them read 0%, not N/A.
