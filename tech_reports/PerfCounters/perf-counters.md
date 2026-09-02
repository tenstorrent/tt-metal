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

The `--perf-counter-multipass` option and the arch-wide expansion of `all` land with tt-metal PR #55166; until it merges, request at most three groups and one L1 bank per run.

Available counter groups for `--profiler-capture-perf-counters`: `fpu`, `pack`, `unpack`, `l1_0`, `l1_1`, `instrn`, `all`. Blackhole also supports `l1_2`, `l1_3`, `l1_4`; `all` expands to the running architecture's full set.

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
| `1 << 6` | 64 | L1 bank 2 (BH only: NOC Ring 2) |
| `1 << 7` | 128 | L1 bank 3 (BH only: NOC Ring 3) |
| `1 << 8` | 256 | L1 bank 4 (BH only: misc ports) |

The env-var path selects one pass directly, so keep it to at most 3 groups: the BRISC firmware image only fits the readout code for 3, and a larger mask overflows its `.text` section (measured on Blackhole). Example single-pass capture:

```bash
export TT_METAL_PROFILE_PERF_COUNTERS=11   # FPU | PACK | L1 bank 0
```

**L1 bank mutual exclusion:** all L1 banks share the same hardware mux (selected via `MUX_CTRL`), so only one L1 bank may be enabled per run; the env-var path throws if more than one L1 bit is set. For anything that needs several passes, use `python -m tracy --perf-counter-multipass` (above), which schedules the passes and merges the results.

### Architecture Summary

| | Wormhole | Blackhole |
|---|---|---|
| Tensix counters read | 135 | 154 |

The derived-metric catalogue (107 metrics, below) is shared between architectures; a metric whose counters exist on only one architecture reports N/A on the other.

**Wormhole** has `PACK_COUNT=4` (4 packer engines), active `o_math_instrnbuf_rden`, and all TDMA counters live. The L1 mux is 1-bit (2 positions: ports 0-7 and 8-15).

**Blackhole** has fewer raw TDMA counters because `PACK_COUNT=1` ties the per-engine busy and dest-read signals for engines 1-3 to constants. Only RTL-live signals are read from hardware — any counter whose RTL signal is hardwired to a constant has been omitted from the `hw_counters.h` arrays, and any aliased grant counter is consolidated to one canonical entry. `Math-to-Pack Handoff Efficiency` falls back to the bank's reference cycles as denominator when `PACKER_BUSY` is 0 for a given workload (e.g. pure-SFPU ops that don't drive the packer); `Packer Efficiency` reports N/A there. TDMA_UNPACK grant banks 4-6 (sels 260-262) have identical RTL wiring on WH and BH (verified: srcB port, srcA overwrite, srcA port). Blackhole has more L1 mux positions (5 vs 2 for Tensix, 4 vs 1 for Ethernet).

**INSTRN_THREAD bank** — `perf_cnt_instrn_thread` is built from a Verilog generate array in `tt_instruction_thread.sv` and has architecture-specific counter_sel mappings. Req-side: sels 0-23 are per-thread instruction-type availability (CFG/SYNC/THCON/MOVE/FPU/UNPACK/PACK, 3 threads each), sels 24-26 are per-thread total stall cycles, and sels 27+ are stall reasons. On WH the shared stall conditions (SRCA/B clear/valid) are replicated across 3 slots each (sels 27-38); on BH they occupy 1 slot each (sels 27-30). Per-thread stall reasons are thread-major: WH sels 39-65 (9 types × 3 threads), BH sels 31-57. Grant-side: the RTL wires grant as `{8{ibuffer_rden[th]}}` per instance and `{9{inst_stall_thread[th]}}` per per-thread stall-reason instance, so the 24 possible issue-count sels collapse to 3 distinct per-thread values and the per-thread stall-reason grants reproduce `THREAD_STALLS_{th}`. We expose only the distinct grants: `THREAD_INSTRUCTIONS_{0,1,2}` at sels 256/264/272 (one per instance) and `ANY_THREAD_STALL` at sel 283. The counter arrays are in arch-specific `hw_counters.h` files; `perf_counters.hpp` is arch-agnostic (WH defines empty L1_2/3/4 arrays).

---

## Derived Metrics Reference

Every derived metric is computed by one shared module, [tools/tracy/perf_metrics_common.py](../../tools/tracy/perf_metrics_common.py). The Tracy tool computes it per operation and core and aggregates to Min/Median/Max/Avg across cores; the tt-llk test harness computes it per zone and run and aggregates to mean/std across runs. The tables below are the complete set. The module is the source of truth, and a unit test (`tests/ttnn/tracy/test_perf_metrics_common.py`) fails if this file stops listing a metric the module computes.

Metrics come in two families, told apart by the key suffix:

- `*_pct` — bounded percentages (0-100%). The numerator is a subset of its denominator.
- `*_ratio` — unbounded raw ratios that can exceed 1.0 by design, because the numerator and denominator come from different measurement domains or because overlapping events are summed. Reported with a `(ratio)` unit and never clamped; the excess over 1.0 is the signal.

A metric whose counters do not exist on the running architecture reports N/A (blank), never 0: the Wormhole-only per-engine packer metrics are N/A on Blackhole, and the Blackhole-only extended L1 groups are N/A on Wormhole. Cross-bank metrics are likewise N/A when one of their counter groups was not captured in the run.

In the formulas, "fpu / instrn / pack / l1 cycles" is that bank's reference-cycle count (`ref_cnt`, the elapsed cycles between counter start and stop), and `1 - x / y` denotes the complement of a counter that counts not-stalled or granted cycles.

### Compute

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| FPU Util (%) | `fpu_utilization_pct` | `FPU_COUNTER / fpu cycles` | Fraction of cycles the FPU executed an instruction. |
| MATH Util (%) | `compute_utilization_pct` | `MATH_COUNTER / fpu cycles` | FPU or SFPU active (the counter is the OR of both). |
| SFPU Util (%) | `sfpu_utilization_pct` | `SFPU_COUNTER / fpu cycles` | Fraction of cycles the SFPU was active. |
| FPU Execution Efficiency (%) | `fpu_exec_eff_pct` | `FPU_COUNTER / FPU_INSTRN_AVAILABLE_1` | Of the cycles a math instruction was available, how many the FPU executed. |
| Math Pipeline Utilization (%) | `math_pipeline_util_pct` | `MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE` | Available math instructions that actually started. |

### Math pipeline stalls

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Fidelity Stall Rate (%) | `fidelity_stall_pct` | `MATH_FIDELITY_STALL / MATH_INSTRN_AVAILABLE` | Math-available cycles stalled by HiFi fidelity phases; 0 at LoFi. |
| Data Hazard Stall Rate (%) | `data_hazard_stall_pct` | `1 - DATA_HAZARD_STALLS_MOVD2A / MATH_INSTRN_AVAILABLE` | MOVD2A data-hazard stall rate (counter counts not-stalled cycles). |
| Math Dest Write Port Stall Rate (%) | `math_dest_wr_port_stall_pct` | `1 - MATH_NOT_STALLED_DEST_WR_PORT / MATH_INSTRN_AVAILABLE` | Dest write-port stalls. N/A when the pack group was not captured. |
| Math Scoreboard Stall Rate (%) | `math_scoreboard_stall_pct` | `1 - AVAILABLE_MATH / MATH_INSTRN_AVAILABLE` | Scoreboard stalls. N/A when the pack group was not captured. |

### Unpacker

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Unpacker0 Write Efficiency (%) | `unpack0_write_eff_pct` | `SRCA_WRITE_ACTUAL / UNPACK0_BUSY_THREAD0` | Unpacker-0 busy cycles that completed a srcA write. |
| Unpacker1 Write Efficiency (%) | `unpack1_write_eff_pct` | `SRCB_WRITE_ACTUAL / UNPACK1_BUSY_THREAD0` | Unpacker-1 busy cycles that completed a srcB write. |
| Unpacker Write Efficiency (%) | `unpack_write_eff_pct` | `mean of the two write efficiencies` | Combined unpacker write efficiency. |
| Unpacker-to-Math Data Flow (%) | `unpack_to_math_flow_pct` | `mean of the srcA and srcB flows` | Combined unpacker-to-math data flow. |
| Unpacker-to-Math Data Flow (srcA) (%) | `unpack_to_math_flow0_pct` | `SRCA_WRITE_AVAILABLE / UNPACK0_BUSY_THREAD0` | srcA buffer availability while unpacker 0 is busy. |
| Unpacker-to-Math Data Flow (srcB) (%) | `unpack_to_math_flow1_pct` | `SRCB_WRITE_AVAILABLE / UNPACK1_BUSY_THREAD0` | srcB buffer availability while unpacker 1 is busy. |
| SrcA Write Actual Efficiency (%) | `srca_write_eff_pct` | `SRCA_WRITE_ACTUAL / SRCA_WRITE_AVAILABLE` | Available srcA writes that completed. |
| SrcB Write Actual Efficiency (%) | `srcb_write_eff_pct` | `SRCB_WRITE_ACTUAL / SRCB_WRITE_AVAILABLE` | Available srcB writes that completed. |
| SrcA Write Port Blocked Rate (%) | `srca_write_port_blocked_pct` | `1 - SRCA_WRITE_ACTUAL / SRCA_WRITE_AVAILABLE` | srcA writes blocked on the write port. |
| SrcA Write Overwrite Blocked Rate (%) | `srca_write_ovr_blocked_pct` | `1 - SRCA_WRITE_NOT_BLOCKED_OVR / SRCA_WRITE_AVAILABLE` | srcA writes blocked by overwrite protection. |
| SrcB Write Overwrite Blocked Rate (%) | `srcb_write_ovr_blocked_pct` | `1 - SRCB_WRITE_ACTUAL / SRCB_WRITE_AVAILABLE` | srcB writes blocked by overwrite protection. |
| SrcB Write Port Blocked Rate (%) | `srcb_write_port_blocked_pct` | `1 - SRCB_WRITE_NOT_BLOCKED_PORT / SRCB_WRITE_AVAILABLE` | srcB writes blocked on the write port. |
| Unpacker0 T1 Share (%) | `unpack0_thread1_share_pct` | `UNPACK0_BUSY_THREAD1 / (thread0 + thread1 busy)` | Unpacker-0 busy cycles driven by the math thread. |
| Unpacker1 T1 Share (%) | `unpack1_thread1_share_pct` | `UNPACK1_BUSY_THREAD1 / (thread0 + thread1 busy)` | Unpacker-1 busy cycles driven by the math thread. |
| SrcA Write T0 Share (%) | `srca_write_thread0_share_pct` | `SRCA_WRITE_THREAD0 / (thread0 + thread1 writes)` | srcA writes issued from thread 0. |
| SrcB Write T0 Share (%) | `srcb_write_thread0_share_pct` | `SRCB_WRITE_THREAD0 / (thread0 + thread1 writes)` | srcB writes issued from thread 0. |

### Packer

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Packer Utilization (%) | `pack_utilization_pct` | `PACKER_BUSY / pack cycles` | Fraction of cycles any packer engine was busy. |
| Packer Efficiency (%) | `pack_dest_eff_pct` | `PACKER_DEST_READ_AVAILABLE / PACKER_BUSY` | Packer busy cycles with dest data available to read. N/A when the packer is idle. |
| Pack Dest Grant Efficiency (%) | `pack_dest_grant_eff_pct` | `DEST_READ_GRANTED_0 / PACKER_DEST_READ_AVAILABLE` | Dest read requests that were granted. |
| Dest Read Backpressure (%) | `dest_read_backpressure_pct` | `1 - DEST_READ_GRANTED_0 / PACKER_DEST_READ_AVAILABLE` | Dest reads waiting on the grant. |
| Packer Engine 0 Util (%) | `packer0_util_pct` | `PACKER_BUSY_0 / pack cycles` | Per-engine packer 0. Wormhole only; N/A on Blackhole. |
| Packer Engine 1 Util (%) | `packer1_util_pct` | `PACKER_BUSY_1 / pack cycles` | Per-engine packer 1. Wormhole only; N/A on Blackhole. |
| Packer Engine 2 Util (%) | `packer2_util_pct` | `PACKER_BUSY_2 / pack cycles` | Per-engine packer 2. Wormhole only; N/A on Blackhole. |
| Packer Engine 3 Util (%) | `packer3_util_pct` | `PACKER_BUSY / pack cycles` | Engine 3 shares the aggregate counter; present on both arches. |
| Packer Load Imbalance (%) | `packer_load_imbalance_pct` | `(max - min) / max over active packer engines` | N/A unless at least two engines were active. |

### Pipeline handoff

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| Math-to-Pack Handoff Efficiency (ratio) | `math_to_pack_handoff_ratio` | `AVAILABLE_MATH / PACKER_BUSY (pack cycles when the packer is idle)` | UNBOUNDED ratio; above 1 the packer is the handoff bottleneck. |
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
| MMIO Idle Wait T0 (%) | `mmio_idle_wait_t0_pct` | `WAITING_FOR_MMIO_IDLE_0 / instrn cycles` | Thread 0 waiting for MMIO. |
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
| CFG Instrn Avail Rate T0 (%) | `cfg_instrn_avail_t0_pct` | `CFG_INSTRN_AVAILABLE_0 / instrn cycles` | CFG instructions pending on thread 0. |
| SYNC Instrn Avail Rate T0 (%) | `sync_instrn_avail_t0_pct` | `SYNC_INSTRN_AVAILABLE_0 / instrn cycles` | SYNC instructions pending on thread 0. |
| THCON Instrn Avail Rate T0 (%) | `thcon_instrn_avail_t0_pct` | `THCON_INSTRN_AVAILABLE_0 / instrn cycles` | THCON instructions pending on thread 0. |
| MOVE Instrn Avail Rate T0 (%) | `move_instrn_avail_t0_pct` | `MOVE_INSTRN_AVAILABLE_0 / instrn cycles` | MOVE instructions pending on thread 0. |
| MATH Instrn Avail Rate T1 (%) | `math_instrn_avail_t1_pct` | `FPU_INSTRN_AVAILABLE_1 / instrn cycles` | Math instructions pending on thread 1. |
| UNPACK Instrn Avail Rate T0 (%) | `unpack_instrn_avail_t0_pct` | `UNPACK_INSTRN_AVAILABLE_0 / instrn cycles` | Unpack instructions pending on thread 0. |
| PACK Instrn Avail Rate T2 (%) | `pack_instrn_avail_t2_pct` | `PACK_INSTRN_AVAILABLE_2 / instrn cycles` | Pack instructions pending on thread 2. |
| T0 Instrn Issue Rate (%) | `thread0_ipc_pct` | `THREAD_INSTRUCTIONS_0 / instrn cycles` | Thread 0 issue rate; single-issue, so at most 100%. |
| T1 Instrn Issue Rate (%) | `thread1_ipc_pct` | `THREAD_INSTRUCTIONS_1 / instrn cycles` | Thread 1 issue rate. |
| T2 Instrn Issue Rate (%) | `thread2_ipc_pct` | `THREAD_INSTRUCTIONS_2 / instrn cycles` | Thread 2 issue rate. |

### L1 client ports and NoC

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| L1 Unpacker Port Util (%) | `l1_unpacker_util_pct` | `L1_0_UNPACKER_0 / l1 cycles` | Unpacker-0 L1 port utilization. |
| L1 Port 1 Util (%) | `l1_port1_util_pct` | `L1_0_UNPACKER_1_ECC_PACK1 / l1 cycles` | L1_0 port 1: pack1+ECC on Wormhole, unpacker1+ECC on Blackhole. |
| L1 TDMA Packer Port Util (%) | `l1_tdma_packer2_util_pct` | `L1_1_TDMA_PACKER_2 / l1 cycles` | L1_1 port 8: TDMA packer 2 on both arches (Blackhole used to misname it RISC core). |
| L1 TDMA Bundle Util (%) | `l1_tdma_bundle_util_pct` | `mean over the two L1_0_TDMA_BUNDLE ports / l1 cycles` | RISC and TRISC TDMA bundle traffic. |
| L1 Ext Unpacker Util (%) | `l1_ext_unpacker_util_pct` | `mean over the extended unpacker ports (L1_1 ext 1-3; plus L1_2 ext 4-7 on Blackhole)` | Extended unpacker interfaces. |
| L1 Ext Packer Util (%) | `l1_ext_pack_util_pct` | `mean over L1_3_EXT_PACKER_2-5 and L1_4_EXT_PACKER_6-7` | Blackhole only; N/A on Wormhole. |
| L1 Tag Search Util (%) | `l1_tag_search_util_pct` | `L1_4_TAG_SEARCH_PACKER_1 / l1 cycles` | Blackhole only; N/A on Wormhole. |
| L1 Mean Client Util (%) | `l1_mean_client_util_pct` | `mean busy/ref over every present L1 client port` | One number for overall L1 client pressure. |
| RISC Core L1 Util (%) | `risc_core_l1_util_pct` | `L1_0_TDMA_BUNDLE_0_RISC / l1 cycles` | The bundle-0 (RISC) port on its own. |
| NOC Ring 0 Util (%) | `noc_ring0_util_pct` | `mean over the ring-0 ports (4 on Wormhole, 8 on Blackhole) / l1 cycles` | NoC ring 0 utilization. |
| NOC Ring 1 Util (%) | `noc_ring1_util_pct` | `mean over the ring-1 ports (4 on Wormhole, 8 on Blackhole) / l1 cycles` | NoC ring 1 utilization. |
| NOC Ring 0 Outgoing Util (%) | `noc_ring0_out_util_pct` | `mean over L1_0_NOC_RING0_OUTGOING_0/1 / l1 cycles` | Primary ring-0 outgoing channels only. |
| NOC Ring 0 Incoming Util (%) | `noc_ring0_in_util_pct` | `mean over L1_0_NOC_RING0_INCOMING_0/1 / l1 cycles` | Primary ring-0 incoming channels only. |
| NOC Ring 1 Outgoing Util (%) | `noc_ring1_out_util_pct` | `mean over L1_1_NOC_RING1_OUTGOING_0/1 / l1 cycles` | Primary ring-1 outgoing channels only. |
| NOC Ring 1 Incoming Util (%) | `noc_ring1_in_util_pct` | `mean over L1_1_NOC_RING1_INCOMING_0/1 / l1 cycles` | Primary ring-1 incoming channels only. |
| NOC Ring 0 Grant Efficiency (%) | `noc_ring0_grant_eff_pct` | `sum of ring-0 grant counters / sum of ring-0 request counters` | Ring-0 requests that were granted. |
| Unpacker L1 Efficiency (ratio) | `unpacker_l1_eff_ratio` | `L1_0_UNPACKER_0_GRANT / UNPACK0_BUSY_THREAD0` | UNBOUNDED ratio (cross-domain); above 1 = ample L1 bandwidth for the unpacker. |
| Packer L1 Efficiency (ratio) | `packer_l1_eff_ratio` | `L1_0_PORT1_GRANT / PACKER_BUSY` | UNBOUNDED ratio; only meaningful on Wormhole, where port 1 carries pack1 traffic. |
| L1 Unpacker Backpressure (%) | `l1_unpacker_backpressure_pct` | `1 - L1_0_UNPACKER_0_GRANT / L1_0_UNPACKER_0` | Unpacker-0 requests waiting on the L1 arbiter. |
| L1 Port 1 Backpressure (%) | `l1_port1_backpressure_pct` | `1 - L1_0_PORT1_GRANT / L1_0_UNPACKER_1_ECC_PACK1` | Port-1 requests waiting on the L1 arbiter. |
| NOC Ring 0 Outgoing Backpressure (%) | `noc_ring0_out_backpressure_pct` | `1 - grants / requests over the primary ring-0 outgoing pair` | L1 not ready for outgoing ring-0 traffic. |
| NOC Ring 0 Incoming Backpressure (%) | `noc_ring0_in_backpressure_pct` | `1 - grants / requests over the primary ring-0 incoming pair` | L1 not ready for incoming ring-0 traffic. |
| NOC Ring 1 Outgoing Backpressure (%) | `noc_ring1_out_backpressure_pct` | `1 - grants / requests over the primary ring-1 outgoing pair` | L1 not ready for outgoing ring-1 traffic. |
| NOC Ring 1 Incoming Backpressure (%) | `noc_ring1_in_backpressure_pct` | `1 - grants / requests over the primary ring-1 incoming pair` | L1 not ready for incoming ring-1 traffic. |
| NOC Ring 1 Grant Efficiency (%) | `noc_ring1_grant_eff_pct` | `sum of ring-1 grant counters / sum of ring-1 request counters` | Ring-1 requests that were granted. |
| L1 Ext Unpacker Backpressure (%) | `l1_ext_unpacker_backpressure_pct` | `1 - grants / requests over the extended unpacker ports` | Extended unpacker contention. |
| L1 Ext Packer Backpressure (%) | `l1_ext_pack_backpressure_pct` | `1 - grants / requests over the extended packer ports` | Blackhole only; N/A on Wormhole. |
| L1 Tag Search Backpressure (%) | `l1_tag_search_backpressure_pct` | `1 - L1_4_TAG_SEARCH_PACKER_1_GRANT / L1_4_TAG_SEARCH_PACKER_1` | Blackhole only; N/A on Wormhole. |

### L1 and NoC composites

| Metric (Tracy CSV label) | Key (LLK CSV column) | Formula | Notes |
|---|---|---|---|
| L1 Total Bandwidth Util (%) | `l1_total_bw_pct` | `sum of the primary L1_0 port counters / (8 x l1 cycles)` | Aggregate L1 bank-0 bandwidth utilization. |
| L1 Read vs Write Ratio (%) | `l1_read_write_ratio_pct` | `reads / (reads + writes), reads = unpacker0 + ring0 out, writes = ring0 in; port 1 counts as a read on Blackhole (unpacker 1) and a write on Wormhole (pack1)` | Above 50% = read-dominated. |
| NOC Ring 0 Asymmetry (%) | `noc_ring0_asymmetry_pct` | `ring0 outgoing / (outgoing + incoming)` | Direction balance of ring-0 traffic. |
| TDMA vs NOC L1 Share (%) | `tdma_vs_noc_l1_share_pct` | `TDMA bundle / (bundle + ring0 traffic)` | Firmware traffic as a share of L1 bank-0 activity. |
| L1 Contention Index (%) | `l1_contention_index_pct` | `mean of (1 - grant/request) over the five primary request/grant pairs` | One number for L1 bank-0 contention. |
| NOC vs Compute Balance (%) | `noc_vs_compute_balance_pct` | `ring0 traffic / (ring0 traffic + FPU_COUNTER)` | Above 50% = NoC-bound, below = compute-bound. |

## Hardware Register Reference

Each counter bank `<X>` (`FPU`, `TDMA_PACK`, `TDMA_UNPACK`, `L1`, `INSTRN_THREAD`) is programmed via three RISC-V debug registers. The programming sequence in `start_perf_counter()` / `stop_perf_counter()` follows this map.

### Control registers (`RISCV_DEBUG_REG_PERF_CNT_<X>0..2`)

| Register | Field | Description |
|---|---|---|
| `RISCV_DEBUG_REG_PERF_CNT_<X>0` | — | Reference period in cycles. |
| `RISCV_DEBUG_REG_PERF_CNT_<X>1` | Bits [7:0] | Mode: `0` = continuous, `1` = count until refclk cycles hit, `2` = continuous (no refclk maintenance). |
| `RISCV_DEBUG_REG_PERF_CNT_<X>1` | Bits [12:8] | Bank select — selects which counter within the bank to read out. |
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

Some counters will still be 0 for a given workload — for example, `WAITING_FOR_SFPU_IDLE_{0,2}` never fires because only the math thread waits for SFPU, and `MATH_FIDELITY_STALL` fires only for HiFi math. These are workload-dependent zeros, not dead counters.
