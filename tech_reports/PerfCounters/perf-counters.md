# Hardware Performance Counters

## Quick Links
- User guide: [docs/source/ttnn/ttnn/profiling_ttnn_operations.rst](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst)
- Firmware implementation: [tt_metal/tools/profiler/perf_counters.hpp](../../tt_metal/tools/profiler/perf_counters.hpp)
- Python analysis: [tools/tracy/perf_counter_analysis.py](../../tools/tracy/perf_counter_analysis.py)
- Metric computation: [tools/tracy/process_ops_logs.py](../../tools/tracy/process_ops_logs.py)

## Overview

Each Tensix core contains hardware performance counters organized into five banks (FPU, TDMA_UNPACK, TDMA_PACK, INSTRN_THREAD, L1). These counters measure cycle-level events: how many cycles the FPU was active, how many cycles a thread was stalled, how many cycles each L1 memory port had traffic, and so on.

The counters are built from a reusable RTL module (`tt_perf_cnt`) that provides three values per event: **req_cnt** (cycles the event signal was high), **grant_cnt** (cycles the grant/ready signal was high), and **ref_cnt** (total elapsed cycles). From these raw values, the profiler computes derived metrics like utilization (`req_cnt / ref_cnt`), backpressure (`(req_cnt - grant_cnt) / req_cnt`), and cross-bank ratios that combine counters from different banks.

## How It Works

1. **Kernel starts**: TRISC1 calls `start_perf_counter()` which writes the start bit to all enabled counter banks. All counters begin accumulating from zero.

2. **Kernel runs**: While the kernel executes, each counter increments every cycle its input signal is high. All counters within a bank run simultaneously — there is no multiplexing during measurement.

3. **Kernel ends**: TRISC1 calls `stop_perf_counter()` which freezes all counters. The counter values remain latched in the debug registers.

4. **BRISC reads counters**: After all TRISCs complete (`wait_ncrisc_trisc()`), BRISC calls `read_perf_counters()` which loops over enabled counter groups, reading each counter by cycling through `counter_sel` values. Each counter value is packed into a 64-bit profiler marker and written to BRISC's profiler buffer. Before each group (starting from the 2nd), BRISC calls `perf_counter_flush()` to push the buffer to DRAM, ensuring each group starts with a fresh buffer. TRISCs cannot do this because they have no NOC access.

5. **Host reads**: After the kernel completes, the host reads the profiler data from DRAM and decodes each marker into a counter type, value, and reference count.

6. **Python processes**: `perf_counter_analysis.py` aggregates raw counter values across cores per operation and computes derived metrics (utilization percentages, backpressure rates, composite ratios). Results are written to CSV and printed to console.

### How to Run

```bash
# Capture all counter groups (fpu, pack, unpack, l1_0, instrn)
python -m tracy --profiler-capture-perf-counters=all \
    -m "pytest your_test.py -x -v"
```

Available counter groups for `--profiler-capture-perf-counters`: `fpu`, `pack`, `unpack`, `l1_0`, `l1_1`, `instrn`, `all`. Blackhole also supports `l1_2`, `l1_3`, `l1_4`. See the [user guide](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst) for details.

### Architecture Summary

| | Wormhole | Blackhole |
|---|---|---|
| Tensix counters read | 135 | 154 |
| Derived metrics | 60+ | 60+ |

**Wormhole** has `PACK_COUNT=4` (4 packer engines), active `o_math_instrnbuf_rden`, and all TDMA counters live. The L1 mux is 1-bit (2 positions: ports 0-7 and 8-15).

**Blackhole** has fewer active TDMA counters due to `PACK_COUNT=1` (single packer engine). Only RTL-live signals are read from hardware — any counter whose RTL signal is hardwired to a constant has been omitted from the `hw_counters.h` arrays, and any aliased grant counter is consolidated to one canonical entry. Three metrics (Packer Efficiency, Math Pipeline Utilization, Math-to-Pack Handoff) use BH-specific fallback formulas because their WH denominator (`PACKER_BUSY`) is always 0 on BH. TDMA_UNPACK grant banks 4-6 (sels 260-262) have identical RTL wiring on WH and BH (verified: srcB port, srcA overwrite, srcA port). Blackhole has more L1 mux positions (5 vs 2 for Tensix, 4 vs 1 for Ethernet).

**INSTRN_THREAD bank** — `perf_cnt_instrn_thread` is built from a Verilog generate array in `tt_instruction_thread.sv` and has architecture-specific counter_sel mappings. Req-side: sels 0-23 are per-thread instruction-type availability (CFG/SYNC/THCON/MOVE/FPU/UNPACK/PACK, 3 threads each), sels 24-26 are per-thread total stall cycles, and sels 27+ are stall reasons. On WH the shared stall conditions (SRCA/B clear/valid) are replicated across 3 slots each (sels 27-38); on BH they occupy 1 slot each (sels 27-30). Per-thread stall reasons are thread-major: WH sels 39-65 (9 types × 3 threads), BH sels 31-57. Grant-side: the RTL wires grant as `{8{ibuffer_rden[th]}}` per instance and `{9{inst_stall_thread[th]}}` per per-thread stall-reason instance, so the 24 possible issue-count sels collapse to 3 distinct per-thread values and the per-thread stall-reason grants reproduce `THREAD_STALLS_{th}`. We expose only the distinct grants: `THREAD_INSTRUCTIONS_{0,1,2}` at sels 256/264/272 (one per instance) and `ANY_THREAD_STALL` at sel 283. The counter arrays are in arch-specific `hw_counters.h` files; `perf_counters.hpp` is arch-agnostic (WH defines empty L1_2/3/4 arrays).

---

## Derived Metrics Reference

Each metric is listed with its formula, what high and low values mean, which architecture supports it, and what it's useful for.

---

### Compute Utilization

**1. SFPU Util**

Measures fraction of cycles the SFPU (special function processing unit) was executing a valid operation.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU |

```
SFPU Util = SFPU_COUNTER / ref_cnt * 100
```

- **High value (>20%)**: SFPU is actively computing. Expected for SFPU-heavy ops like sqrt, gelu, exp.
- **Low value (~0%)**: SFPU is idle. Expected for matmul (uses FPU path) or data movement ops.

**Use case:** Confirms whether an operation is using the SFPU pipeline. Compare with FPU Util to distinguish FPU vs SFPU workloads.

---

**2. FPU Util**

Measures fraction of cycles the FPU was executing a valid operation.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU |

```
FPU Util = FPU_COUNTER / ref_cnt * 100
```

- **High value (>20%)**: FPU is actively computing. Expected for matmul, eltwise multiply.
- **Low value (~0%)**: FPU is idle. Expected for SFPU ops or pure data movement.

**Use case:** Primary indicator of compute utilization for FPU-path operations.

---

**3. MATH Util**

Measures combined FPU + SFPU utilization (cycles where either was active).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU |

```
MATH Util = MATH_COUNTER / ref_cnt * 100
```

- **High value (>30%)**: Math hardware is well-utilized. Expected for compute-heavy ops like sqrt (74%), relu (36%).
- **Low value (~0%)**: No math activity. Expected for pure data movement (tilize, concat).

**Use case:** Single-number summary of total compute utilization regardless of FPU vs SFPU path.

---

### Pipeline Efficiency

**4. Packer Efficiency**

Measures how often the packer has valid destination data available when it's busy.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | PACK |

```
WH:  Packer Efficiency = PACKER_DEST_READ_AVAILABLE / PACKER_BUSY * 100
BH:  Packer Efficiency = DEST_READ_GRANTED_0 / PACKER_DEST_READ_AVAILABLE * 100
```

On Blackhole, `PACKER_BUSY` is always 0 empirically (the packer completes within its gated clock window and the OR of per-thread busy bits never stays high long enough for the ungated perf counter to sample). The BH fallback uses `DEST_READ_GRANTED_0` (counter_sel 267), the matched grant for `PACKER_DEST_READ_AVAILABLE` (counter_sel 11, req[0]). PACK_COUNT=1 on BH means grant[0] is the only live grant — grants 1-3 (sels 268-270) are RTL-tied to `1'b0` and not read. The ratio is the grant rate: 100% = every request granted, <100% = packer waited for dest data.

- **High value (100%)**: Packer always has data when busy (no stalls). This is the normal case on BH.
- **Low value (<80%)**: Packer is busy but waiting for destination register data from math stage.

**Use case:** Detects destination register stalls indicating the math stage is not keeping up with the packer.

---

**5. Math-to-Pack Handoff Efficiency**

Measures pipeline balance between math output and packer consumption. Capped at 100%.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | PACK |

```
WH:  Math-to-Pack Handoff = AVAILABLE_MATH / PACKER_BUSY * 100
BH:  Math-to-Pack Handoff = AVAILABLE_MATH / ref_cnt * 100
```

On Blackhole, `PACKER_BUSY` is always 0. The BH fallback measures what fraction of cycles the math pipeline was not stalled by the scoreboard — a direct measure of math availability to the pack stage.

- **WH high value (>100%)**: Math produces output faster than packer consumes.
- **BH high value (>25%)**: Math unit is available (not scoreboard-stalled) for a significant fraction of cycles.
- **Low value**: Math is frequently stalled, limiting pack throughput.

**Use case:** Identifies whether math is keeping up with the packer. Low values indicate math is the bottleneck.

---

**6. Unpacker-to-Math Data Flow**

Measures backpressure from math stage to unpackers.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
Unpacker-to-Math Data Flow = avg(SRCA_WRITE_AVAILABLE, SRCB_WRITE_AVAILABLE) /
                             avg(UNPACK0_BUSY_THREAD0, UNPACK1_BUSY_THREAD0) * 100
```

- **High value (>80%)**: Unpackers can write to source registers when busy. Good data flow.
- **Low value (<30%)**: Unpackers are busy but source register buffers are full. Math is not consuming data fast enough.

**Use case:** Detects math stage backpressure causing unpacker stalls. Compare with Unpacker Write Efficiency (WH only) to distinguish backpressure from other stall types.

---

### Thread Analysis

**7. Thread 0/1/2 Stall Rate**

Measures fraction of cycles each instruction thread was stalled.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
Thread N Stall Rate = THREAD_STALLS_N / ref_cnt * 100
```

Thread mapping: Thread 0 = unpack, Thread 1 = math, Thread 2 = pack.

- **High value (>30%)**: Thread is frequently stalled. For Thread 0 this usually means waiting for data (NOC, semaphore). For Thread 1, waiting for math hardware. For Thread 2, waiting for pack hardware.
- **Low value (<5%)**: Thread rarely stalls. Expected for compute-bound ops on the math thread.

**Use case:** First-order indicator of where time is being lost. The stall breakdown metrics (below) identify the specific stall reason.

---

**8. T0/T1/T2 Instrn Issue Rate**

Average instructions issued per cycle, per thread.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
TN Instrn Issue Rate = THREAD_INSTRUCTIONS_N / ref_cnt
```

Thread mapping: Thread 0 = unpack, Thread 1 = math, Thread 2 = pack.

- **High value (close to 1.0)**: Thread issues an instruction almost every cycle.
- **Low value (<0.1)**: Thread is stalled or idle most of the time.

**Use case:** Lets you see where the pipeline is spending instruction-issue bandwidth. Combined with the stall-rate metric (#7) it distinguishes "thread issued lots of instructions" from "thread sat idle".

---

### Pipeline Wait Metrics

**9. SrcA/SrcB Valid Wait**

Cycles waiting for source register data to become valid (unpacker hasn't filled it yet).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
SrcA Valid Wait = WAITING_FOR_SRCA_VALID / ref_cnt * 100
SrcB Valid Wait = WAITING_FOR_SRCB_VALID / ref_cnt * 100
```

- **High value (>5%)**: Math is waiting for unpacker to provide data. Data starvation.
- **Low value (~0%)**: Data is ready when math needs it.

**Use case:** Detects data starvation from the unpacker side.

---

**10. SrcA/SrcB Clear Wait**

Cycles waiting for source register to be cleared (math is still using the previous data).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
SrcA Clear Wait = WAITING_FOR_SRCA_CLEAR / ref_cnt * 100
SrcB Clear Wait = WAITING_FOR_SRCB_CLEAR / ref_cnt * 100
```

- **High value (>10%)**: Register pressure — math is holding srcA/B longer than unpack can wait. Common for silu (35%), relu (16%).
- **Low value (~0%)**: No register pressure. Math releases source registers quickly.

**Use case:** Identifies source register contention between unpack and math threads.

---

**11. Math/Pack/Unpack Idle Wait**

Cycles each thread waits for its primary hardware unit to become idle.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
Math Idle Wait T1 = WAITING_FOR_MATH_IDLE_1 / ref_cnt * 100
Pack Idle Wait T2 = WAITING_FOR_PACK_IDLE_2 / ref_cnt * 100
Unpack Idle Wait T0 = WAITING_FOR_UNPACK_IDLE_0 / ref_cnt * 100
```

- **High value (>10%)**: Hardware unit is busy when the thread needs it. Pipeline bottleneck.
- **Low value (~0%)**: Hardware is available when needed.

**Use case:** Pinpoints which hardware unit is the pipeline bottleneck.

---

### Semaphore Waits

**12. Semaphore Zero/Full Wait T0/T1/T2**

Cycles each thread waits for semaphore synchronization.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
Semaphore Zero Wait TN = WAITING_FOR_NONZERO_SEM_N / ref_cnt * 100
Semaphore Full Wait TN = WAITING_FOR_NONFULL_SEM_N / ref_cnt * 100
```

- **Semaphore Zero Wait high (>10%)**: Thread is waiting for a producer to signal (semaphore is 0). Common for tilize (7%) where unpack waits for data.
- **Semaphore Full Wait high (>5%)**: Thread is waiting for a consumer to drain (semaphore is at max). Indicates backpressure from downstream.
- **Both low (~0%)**: Good producer-consumer balance.

**Use case:** Identifies producer-consumer imbalances between threads or between host and device.

---

### TDMA Stall Metrics

**13. Data Hazard Stall Rate**

Cycles stalled by destination-to-source data hazards (MOVD2A/MOVD2B operations).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
Data Hazard Stall Rate = DATA_HAZARD_STALLS_MOVD2A / ref_cnt * 100
```

- **High value (>20%)**: Significant dest-to-src data movement stalls. Expected for matmul (29%).
- **Low value (~0%)**: No data hazard stalls. Expected for simple eltwise ops.

**Use case:** Identifies operations with heavy dest-to-src register movement overhead.

---


**14. SrcA Write Overwrite Blocked Rate**

Fraction of srcA DMA write attempts blocked by overwrite protection (data not yet consumed by math).

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
SrcA Write Blocked = (SRCA_WRITE_AVAILABLE - SRCA_WRITE_NOT_BLOCKED_OVR) /
                     SRCA_WRITE_AVAILABLE * 100
```

On WH, `SRCA_WRITE_NOT_BLOCKED_OVR` (counter_sel 261) directly measures srcA DMA writes not blocked by overwrite. On BH, counter_sel 260 is used (verified empirically).

- **High value (>30%)**: SrcA writes are frequently blocked. Data overwrite protection is active.
- **Low value (~0%)**: SrcA writes proceed without blocking.

**Use case:** Detects source register overwrite contention on the srcA path.

---

**15. SrcB Write Port Blocked Rate**

Fraction of srcB DMA write attempts blocked by port unavailability.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
SrcB Write Port Blocked = (SRCB_WRITE_AVAILABLE - SRCB_WRITE_NOT_BLOCKED_PORT) /
                          SRCB_WRITE_AVAILABLE * 100
```

On WH, `SRCB_WRITE_NOT_BLOCKED_PORT` (counter_sel 260) directly measures srcB DMA writes not blocked by the write port. On BH, counter_sel 262 is used (verified empirically).

- **High value (>50%)**: SrcB write port is heavily contended.
- **Low value (<20%)**: SrcB writes proceed with minimal blocking.

**Use case:** Detects srcB write port contention.

---

**16. Dest Read Backpressure**

Fraction of packer destination register reads that were blocked.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | PACK |

```
Dest Read Backpressure = (PACKER_DEST_READ_AVAILABLE - DEST_READ_GRANTED_0) /
                         PACKER_DEST_READ_AVAILABLE * 100
```

- **High value (>20%)**: Packer can't read destination register (math still writing).
- **Low value (~0%)**: No destination register contention.

**Use case:** Identifies math-to-pack register handoff bottleneck.

---

**17. Math Dest Write Port Stall Rate**

Fraction of math cycles stalled by destination register write port contention.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | PACK |

```
Math Dest Write Port Stall = (MATH_INSTRN_AVAILABLE - MATH_NOT_STALLED_DEST_WR_PORT) /
                             MATH_INSTRN_AVAILABLE * 100
```

- **High value (>10%)**: Math is stalled waiting for write port to destination register.
- **Low value (~0%)**: No write port stalls.

**Not available on Blackhole**: `MATH_NOT_STALLED_DEST_WR_PORT` is always 0 on BH (empirically verified), which would produce a bogus 100% stall rate. The metric is automatically hidden on BH. On WH, this counter is live and the metric works correctly.

**Use case:** Detects destination register write contention from the math side.

---

**18. Math Scoreboard Stall Rate**

Fraction of math cycles stalled by FPU data hazard scoreboard.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | PACK |

```
Math Scoreboard Stall = (MATH_INSTRN_AVAILABLE - AVAILABLE_MATH) /
                        MATH_INSTRN_AVAILABLE * 100
```

- **High value (>10%)**: FPU scoreboard is blocking math instructions (RAW/WAW hazards).
- **Low value (~0%)**: No scoreboard stalls.

**Use case:** Identifies FPU pipeline hazards that prevent math instructions from issuing.

---

### Instruction Availability Rates

**19. CFG/SYNC/THCON/MOVE/MATH/UNPACK/PACK Instrn Avail Rate**

Fraction of cycles each instruction type was available in its primary thread's instruction buffer.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
TYPE Instrn Avail Rate = TYPE_INSTRN_AVAILABLE_N / ref_cnt * 100
```

Where TYPE is CFG, SYNC, THCON, MOVE (on T0), MATH (on T1), UNPACK (on T0), PACK (on T2).

- **High value (>10%)**: This instruction type frequently sits in the buffer waiting to issue. The thread is spending significant time on this instruction type.
- **Low value (~0%)**: This instruction type is rarely pending. Either rarely used or issues immediately.

**Use case:** Identifies which instruction types dominate the scheduling pipeline. High THCON avail rate (matmul 13%) indicates thread control overhead.

---

### Stall Breakdown

> **Removed metrics:** The stall breakdown percentage metrics (`THCON/MOVE Idle Stall Pct T0`, `MMIO/SFPU Idle Stall Pct T1`) were removed. They divided `WAITING_FOR_X` by `THREAD_STALLS`, but `WAITING_FOR_X` counts cycles a hardware unit was busy — not cycles the thread was stalled by that unit. When thread stalls are low (e.g. concat, tilize), `WAITING_FOR_X >> THREAD_STALLS`, producing meaningless >100% values. Use the absolute idle wait metrics (`MMIO/SFPU/THCON/MOVE Idle Wait`) instead, which correctly measure % of total time each hardware unit was busy.

---

### Write Port Analysis

**20. SrcA Write Actual Efficiency**

Fraction of srcA write attempts that actually succeeded.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
SrcA Write Actual Efficiency = SRCA_WRITE_ACTUAL / SRCA_WRITE_AVAILABLE * 100
```

- **High value (100%)**: Every srcA write attempt succeeds. No write port blocking.
- **Low value (<80%)**: Significant fraction of srcA writes are blocked.

**Note:** SrcB Write Actual Efficiency works on both Wormhole and Blackhole.

**Use case:** Measures effective srcA write throughput. Low values indicate write port contention.

---

### Additional Idle Waits

**21. MMIO/SFPU/THCON/MOVE Idle Wait**

Fraction of total cycles each thread spent waiting for specific hardware units.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
MMIO Idle Wait T0 = WAITING_FOR_MMIO_IDLE_0 / ref_cnt * 100
SFPU Idle Wait T1 = WAITING_FOR_SFPU_IDLE_1 / ref_cnt * 100
THCON Idle Wait T0 = WAITING_FOR_THCON_IDLE_0 / ref_cnt * 100
MOVE Idle Wait T0 = WAITING_FOR_MOVE_IDLE_0 / ref_cnt * 100
```

- **High value (>5%)**: Significant time spent waiting for this unit. MOVE Idle Wait at 2.8% for tilize is expected.
- **Low value (~0%)**: Hardware unit is fast enough to never bottleneck. THCON and MMIO are typically ~0%.

**Use case:** Absolute (not relative) measure of time lost to each hardware unit. Unlike the stall breakdown metrics which show percentage of stalls, these show percentage of total time.

---

**22. RISC Core L1 Util**

RISC core L1 memory access utilization.

| | |
|---|---|
| **Architectures** | Blackhole only |
| **Counter group** | L1_1 |

```
RISC Core L1 Util = L1_1_RISC_CORE / ref_cnt * 100
```

- **High value (>10%)**: RISC core is actively accessing L1. Indicates firmware memory overhead.
- **Low value (~0%)**: Minimal RISC L1 traffic.

**Use case:** Measures firmware memory access overhead on BH. Requires L1_1 group enabled.

---

### L1 Memory Utilization

**23. L1 Unpacker/Packer Port Util**

Fraction of cycles each L1 port had a transaction attempt.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
L1 Unpacker Port Util = L1_0_UNPACKER_0 / ref_cnt * 100
L1 Packer Port Util = L1_0_PORT1 / ref_cnt * 100
```

- **High value (>20%)**: Port is heavily used. Matmul shows 15% on unpacker.
- **Low value (~0%)**: Port is idle (e.g. sqrt has 0% unpacker port util because data arrives via NOC).

**Use case:** Identifies which L1 ports are active for a given operation.

---

**24. L1 TDMA Bundle Util**

Average utilization of the two TDMA/RISC L1 ports.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
L1 TDMA Bundle Util = avg(L1_0_TDMA_BUNDLE_0_RISC, L1_0_TDMA_BUNDLE_1_TRISC) / ref_cnt * 100
```

- **High value (>10%)**: RISC/TDMA data movement is significant. Relu shows 11%.
- **Low value (~0%)**: Minimal RISC/TDMA L1 access.

**Use case:** Measures firmware and TDMA data movement overhead through L1.

---

**25. NOC Ring 0/1 Outgoing/Incoming Util**

Average utilization of NOC channels per ring.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 (Ring 0), L1_1 (Ring 1) |

```
NOC Ring 0 Outgoing Util = avg(L1_0_NOC_RING0_OUTGOING_0, _1) / ref_cnt * 100
NOC Ring 0 Incoming Util = avg(L1_0_NOC_RING0_INCOMING_0, _1) / ref_cnt * 100
```

- **High value (>15%)**: Significant NOC traffic. Matmul shows 15% outgoing + 16% incoming.
- **Low value (<3%)**: Minimal NOC traffic. Sqrt shows 3% outgoing, 0% incoming.

**Use case:** Measures NOC bandwidth utilization per ring. Compare outgoing vs incoming to understand data flow direction.

---

### L1 Backpressure

**26. NOC Ring 0/1 Outgoing/Incoming Backpressure**

Fraction of NOC transaction cycles where L1 was not ready.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 (Ring 0), L1_1 (Ring 1) |

```
NOC Ring 0 Outgoing BP = (req0 + req1 - grant0 - grant1) / (req0 + req1) * 100
```

- **High value (>15%)**: NOC outgoing traffic is being stalled by L1. Sqrt shows 15%.
- **Low value (<5%)**: NOC traffic flows with minimal L1 stalls. Matmul shows 2%.

**Use case:** Detects L1 port contention affecting NOC traffic. High outgoing BP means data can't leave the core fast enough.

---

**27. L1 Unpacker/Packer Port Backpressure**

L1 port contention for unpacker and packer.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
L1 Unpacker BP = (L1_0_UNPACKER_0 - L1_0_UNPACKER_0_GRANT) / L1_0_UNPACKER_0 * 100
L1 Packer Port BP = (L1_0_PORT1 - L1_0_PORT1_GRANT) / L1_0_PORT1 * 100
```

- **L1 Unpacker BP high (>80%)**: L1 is almost always busy when unpacker wants access. Values of 75-100% are common across all ops.
- **L1 Packer Port BP low (<5%)**: Packer port has low contention. Normal.

**Blackhole note:** On BH, the L1 unpacker grant counter can exceed the request counter on some cores (different signal semantics). When this happens, the backpressure metric is suppressed rather than showing meaningless values.

**Use case:** High unpacker BP is expected (unpacker competes with other ports). Investigate only if combined with high Thread 0 stall rate.

---

### L1 Composite Metrics

These metrics combine multiple counters to provide higher-level L1 insights.

**28. L1 Total Bandwidth Util**

Overall L1 memory bandwidth saturation.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
L1 Total BW Util = sum(all 8 port req counts) / (8 * ref_cnt) * 100
```

- **High value (>30%)**: L1 is heavily utilized. May become a bottleneck.
- **Medium value (10-20%)**: Moderate L1 usage. Matmul shows 12.6%.
- **Low value (<5%)**: L1 bandwidth is underutilized.

**Use case:** Single number showing how much of the theoretical L1 bandwidth is being used. If this is high and performance is low, L1 is the bottleneck.

---

**29. L1 Read vs Write Ratio**

Balance between read and write L1 traffic.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
L1 R/W Ratio = (Unpacker + NOC Outgoing) / (Unpacker + NOC Outgoing + Packer + NOC Incoming) * 100
```

Read ports: Unpacker (reads tiles from L1), NOC Outgoing (sends data from L1 to network).
Write ports: Packer (writes results to L1), NOC Incoming (receives data from network to L1).

- **~50%**: Balanced read/write. Matmul shows 50%.
- **>70%**: Read-heavy. Operation is primarily consuming data.
- **<30%**: Write-heavy. Operation is primarily producing data.

**Use case:** Understands data flow direction through L1.

---

**30. NOC Ring 0 Asymmetry**

Balance between outgoing and incoming NOC traffic.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
NOC Asymmetry = NOC_Outgoing / (NOC_Outgoing + NOC_Incoming) * 100
```

- **~50%**: Balanced send/receive. Matmul shows 50%.
- **>70%**: Send-heavy (core produces more than it consumes).
- **<30%**: Receive-heavy (core consumes more than it produces).

**Use case:** Identifies directional imbalance in NOC traffic which may indicate suboptimal data placement.

---

**31. L1 Contention Index**

Average backpressure across all active L1 ports.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
L1 Contention Index = avg(backpressure of Unpacker, NOC Out 0, NOC Out 1, NOC In 0, NOC In 1)
```

- **High value (>40%)**: Significant L1 port contention across the board.
- **Medium value (15-30%)**: Moderate contention. Matmul shows 22%.
- **Low value (<10%)**: Minimal L1 contention.

**Use case:** Single number summarizing overall L1 memory stress level. Easier to compare across ops than looking at individual port backpressures.

---

**32. Unpacker L1 Efficiency**

When the unpacker is busy, how often does L1 actually serve it.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 + UNPACK |

```
Unpacker L1 Efficiency = L1_0_UNPACKER_0_GRANT / UNPACK0_BUSY_THREAD0 * 100
```

- **High value (>50%)**: L1 serves unpacker requests efficiently.
- **Low value (<5%)**: L1 is frequently unable to serve the unpacker when it needs data. Matmul shows 0.25% (high contention from other ports).

**Use case:** Measures actual L1 service rate to the unpacker. Low values combined with high unpacker backpressure confirm L1 is the bottleneck for data delivery.

---

**33. Packer L1 Efficiency**

When the packer is busy, how often does L1 serve it.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 + PACK |

```
Packer L1 Efficiency = L1_0_PORT1_GRANT / PACKER_BUSY * 100
```

- **High value (>100%)**: L1 port has headroom. Values can exceed 100% because the packer port is shared with other clients (ECC on WH, other traffic on BH), so grant cycles may exceed packer busy cycles.
- **Low value (<50%)**: Packer is being starved by L1.

**Use case:** Confirms packer is not L1-bottlenecked. Low values would indicate L1 port contention affecting write-back.

---

**34. NOC vs Compute Balance**

Whether the operation is NOC-bound or compute-bound.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 + FPU |

```
NOC vs Compute = (NOC_Out + NOC_In) / (FPU_COUNTER + NOC_Out + NOC_In) * 100
```

- **>60%**: NOC-bound. More time is spent on data movement than compute. Matmul shows 68%.
- **~50%**: Balanced between compute and data movement.
- **<40%**: Compute-bound. FPU/SFPU is the bottleneck.

**Use case:** Quick diagnostic for whether to optimize compute kernels or data placement/NOC routing.

---

**35. TDMA vs NOC L1 Share**

Fraction of L1 bandwidth used by RISC/TDMA vs NOC.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 |

```
TDMA vs NOC = (TDMA_Bundle_0 + TDMA_Bundle_1) / (TDMA + NOC_Out + NOC_In) * 100
```

- **High value (>20%)**: RISC/TDMA firmware uses significant L1 bandwidth.
- **Low value (<5%)**: Most L1 bandwidth goes to NOC. Matmul shows 3%.

**Use case:** Measures firmware L1 overhead. High values may indicate firmware optimization opportunities.

---

**36. Stall Cause Overlap Factor T0/T1/T2**

Ratio of the sum of all 9 per-thread stall reason counters to the total thread stalls. Values >1.0 mean multiple stall conditions are active simultaneously.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
Stall Overlap TN = sum(all 9 WAITING_FOR_*_N) / THREAD_STALLS_N
```

- **~1.0x**: Single dominant stall cause. The thread is stalled for one reason at a time.
- **>2.0x**: Multiple hardware units are busy simultaneously when the thread stalls. Common for Thread 2 (pack) which may have move + pack stalls overlapping.

**Use case:** Tells you whether to focus on a single bottleneck or multiple interacting ones.

---

**37. Packer Load Imbalance**

Spread between the most and least utilized packer engines. WH only (PACK_COUNT=4).

| | |
|---|---|
| **Architectures** | Wormhole only |
| **Counter group** | PACK |

```
Packer Load Imbalance = (max(BUSY_0..3) - min(BUSY_0..3)) / max(BUSY_0..3) * 100
```

- **Low value (<10%)**: Work is evenly distributed across pack engines.
- **High value (>25%)**: Significant imbalance. Some engines are idle while others are busy.

**Use case:** Detects uneven work distribution across WH's 4 packer engines which may indicate suboptimal tile packing.

---

**38. Compute-to-Unpack Ratio**

Whether the operation is compute-bound or memory-bound.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | FPU + UNPACK |

```
Compute-to-Unpack Ratio = MATH_COUNTER / (UNPACK0_BUSY + UNPACK1_BUSY) * 100
```

- **>100%**: Compute-bound — math takes longer than data unpacking.
- **~50%**: Balanced between compute and data movement. Matmul shows ~53%.
- **<20%**: Memory-bound — unpackers are busy much longer than math.

**Use case:** Quick one-number diagnostic for whether to optimize compute kernels or data placement.

---

### Architecture-Specific Metrics

Some of these metrics use BH-specific fallback formulas. Others are truly Wormhole-only (marked below) because the required hardware signal is inactive on Blackhole.

**39. Math Pipeline Utilization**

Measures math instruction flow efficiency through the pipeline.

| | |
|---|---|
| **Architectures** | Wormhole |
| **Counter group** | UNPACK |

```
Math Pipeline Utilization = MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE * 100
```

On Blackhole, `MATH_INSTRN_STARTED` is empirically inactive (`o_math_instrnbuf_rden` never fires), so this metric is hidden from BH output.

- **High value (>80%)**: Math pipeline efficiently moves instructions.
- **Low value**: Math pipeline is mostly idle or stalled.



**40. SrcB Write Actual Efficiency**

Fraction of srcB write attempts that actually succeeded.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
SrcB Write Actual Efficiency = SRCB_WRITE_ACTUAL / SRCB_WRITE_AVAILABLE * 100
```

- **High value (100%)**: Every srcB write attempt succeeds.
- **Low value (<80%)**: Significant fraction of srcB writes are blocked.

**Use case:** Measures effective srcB write throughput.

---


**41. Packer Engine 0/1/2 Util**

Per-engine packer utilization.

| | |
|---|---|
| **Architectures** | Wormhole only |
| **Counter group** | PACK |

```
Packer Engine N Util = PACKER_BUSY_N / ref_cnt * 100
```

**Not available on Blackhole**: `PACK_COUNT=1`, per-engine busy signals tied to 0.

**Use case:** Detects load imbalance across WH's 4 packer engines.

---

**42. Unpacker0/1 Write Efficiency**

Source register write throughput per unpacker.

| | |
|---|---|
| **Architectures** | Wormhole only |
| **Counter group** | UNPACK |

```
Unpacker0 Efficiency = SRCA_WRITE / UNPACK0_BUSY_THREAD0 * 100
Unpacker1 Efficiency = SRCB_WRITE / UNPACK1_BUSY_THREAD0 * 100
```

- **High value (>80%)**: Unpacker spends most busy time writing data (efficient).
- **Low value (<20%)**: Unpacker is busy but not writing (stalled/waiting).

**Not available on Blackhole**: Counter signals read 0 on BH.

**Use case:** Identifies unpacker stalls due to L1 contention or data dependencies.

---

**43. FPU Execution Efficiency**

FPU active cycles as fraction of math instruction availability.

| | |
|---|---|
| **Architectures** | Wormhole only |
| **Counter group** | FPU + INSTRN |

```
FPU Execution Efficiency = FPU_COUNTER / FPU_INSTRN_AVAILABLE_1 * 100
```

- **High value (>80%)**: FPU executes whenever work is available (compute-efficient).
- **Low value (<30%)**: FPU instructions available but not executing (pipeline stalls).

**Not available on Blackhole**: `FPU_INSTRN_AVAILABLE_1` reads 0 on BH.

**Use case:** Distinguishes compute-bound (high efficiency) from memory-bound (low efficiency) workloads.

---

## Hardware Limitations

### Not Accessible from Software

| Feature | Architecture | Reason | Workaround |
|---------|-------------|--------|------------|
| `stall_cnt` (bits [127:96]) | BH | `out_fmt` is 1-bit, no mode to route bits [127:96] | Software derives stall as `req - grant` |
| DDR5 RISC L1 counters | BH | Counters instantiated but no debug register interface | None — requires RTL change |

### Counter Set

Verified against the `wormhole_rtl` and `blackhole_rtl` branches. Every counter exposed via the `hw_counters.h` arrays is driven by a real RTL signal — signals that are hardwired to a constant, or whose grant/req line is an alias of another counter we already expose, are omitted from the arrays entirely. No post-hoc filtering is applied; every emitted counter is reported as-is.

Some counters will still be 0 for a given workload — for example, `WAITING_FOR_SFPU_IDLE_{0,2}` never fires because only the math thread waits for SFPU, and `MATH_INSTRN_HF_2/4_CYCLE` fire only for HiFi math. These are workload-dependent zeros, not dead counters.
