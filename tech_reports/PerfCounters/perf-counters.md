# Hardware Performance Counters

## Quick Links
- User guide: [docs/source/ttnn/ttnn/profiling_ttnn_operations.rst](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst)
- Firmware implementation: [tt_metal/tools/profiler/perf_counters.hpp](../../tt_metal/tools/profiler/perf_counters.hpp)
- Python analysis: [tools/tracy/perf_counter_analysis.py](../../tools/tracy/perf_counter_analysis.py)
- Metric computation: [tools/tracy/process_ops_logs.py](../../tools/tracy/process_ops_logs.py)

## Overview

Each Tensix core contains hardware performance counters organized into five banks (FPU, TDMA_UNPACK, TDMA_PACK, INSTRN_THREAD, L1). These counters measure cycle-level events: how many cycles the FPU was active, how many cycles a thread was stalled, how many cycles each L1 memory port had traffic, and so on.

The counters are built from a reusable RTL module (`tt_perf_cnt`) that provides three values per event: **req_cnt** (cycles the event signal was high), **grant_cnt** (cycles the grant/ready signal was high), and **ref_cnt** (total elapsed cycles). From these raw values, the profiler computes derived metrics like utilization (`req_cnt / ref_cnt`), backpressure (`(req_cnt - grant_cnt) / req_cnt`), and cross-bank ratios that combine counters from different banks.

Ethernet (ERISC) cores also contain L1 performance counters with the same interface but different port mappings.

## How It Works

1. **Kernel starts**: The firmware calls `start_perf_counter()` which writes the start bit to all enabled counter banks. All counters begin accumulating from zero.

2. **Kernel runs**: While the kernel executes, each counter increments every cycle its input signal is high. All counters within a bank run simultaneously — there is no multiplexing during measurement.

3. **Kernel ends**: The firmware calls `stop_perf_counter()` which freezes all counters, then reads each one by cycling through `counter_sel` values (which select which counter's output is routed to the read registers). Each counter value is packed into a 64-bit profiler marker and written to the profiler buffer.

4. **Host reads**: After the kernel completes, the host reads the profiler buffer and decodes each marker into a counter type, value, and reference count.

5. **Python processes**: `process_ops_logs.py` aggregates raw counter values across cores per operation and computes derived metrics (utilization percentages, backpressure rates, composite ratios). Results are written to CSV and printed to console.

### How to Run

```bash
# Enable profiler + perf counters (47 = FPU|PACK|UNPACK|L1_0|INSTRN)
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=47 \
    pytest your_test.py -x -v

# Generate metrics report
python tools/tracy/process_ops_logs.py --device-only
```

The `TT_METAL_PROFILE_PERF_COUNTERS` value selects which counter banks to enable. Use `47` to capture FPU, PACK, UNPACK, L1_0, and INSTRN — this covers all Tensix pipeline stages and L1 Ring 0 ports. See the [user guide](../../docs/source/ttnn/ttnn/profiling_ttnn_operations.rst) for the full bitfield reference.

### Architecture Summary

| | Wormhole | Blackhole |
|---|---|---|
| Tensix raw counters | 172 | 126 |
| ERISC raw counters | 16 | 64 |
| Derived metrics | 86 | 74 |

Blackhole has fewer TDMA counters (6 signals inactive due to `PACK_COUNT=1` and `o_math_instrnbuf_rden` tied off) but more L1 mux positions (5 vs 2 for Tensix, 4 vs 1 for Ethernet).

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
Packer Efficiency = PACKER_DEST_READ_AVAILABLE / PACKER_BUSY * 100
```

- **High value (100%)**: Packer always has data when busy (no stalls). This is the normal case on BH with PACK_COUNT=1.
- **Low value (<80%)**: Packer is busy but waiting for destination register data from math stage.

**Use case:** Detects destination register stalls indicating the math stage is not keeping up with the packer.

---

**5. Math-to-Pack Handoff Efficiency**

Measures pipeline balance between math output and packer consumption.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | PACK |

```
Math-to-Pack Handoff = AVAILABLE_MATH / PACKER_BUSY * 100
```

- **High value (>100%)**: Math produces output faster than packer consumes (packer is bottleneck).
- **Value near 100%**: Math and packer are well-balanced.
- **Low value (<50%)**: Packer is busy but math output isn't ready (math is bottleneck).

**Use case:** Identifies whether math or packer is the pipeline bottleneck.

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

**8. Thread 0/1/2 IPC**

Instructions per cycle for each thread.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
Thread N IPC = THREAD_INSTRUCTIONS_N / ref_cnt
```

- **High value (>0.5)**: Thread is issuing instructions efficiently.
- **Low value (<0.1)**: Thread is mostly stalled or idle.

**Use case:** Quantifies instruction throughput per thread. Compare across threads to find the slowest one.

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

**14. Fidelity Phase Overhead**

Cycles spent on HiFi fidelity phases in the math pipeline.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
Fidelity Phase Overhead = FIDELITY_PHASE_STALLS / ref_cnt * 100
```

- **High value (>20%)**: Significant time spent on fidelity phases. Expected for HiFi4 matmul (30%).
- **Low value (<5%)**: Minimal fidelity overhead. Expected for LoFi or non-math ops.

**Use case:** Quantifies the cost of high-fidelity math. If too high, consider LoFi mode for better throughput.

---

**15. SrcA Write Port Blocked Rate**

Fraction of srcA DMA write attempts blocked by overwrite protection.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
SrcA Write Port Blocked = (SRCA_WRITE_AVAILABLE - SRCA_WRITE_NOT_BLOCKED_OVR) /
                          SRCA_WRITE_AVAILABLE * 100
```

- **High value (>30%)**: SrcA writes are frequently blocked. Data overwrite protection is active.
- **Low value (~0%)**: SrcA writes proceed without blocking.

**Use case:** Detects source register overwrite contention on the srcA path.

---

**16. SrcB Write Port Blocked Rate**

Fraction of srcB DMA write attempts blocked by port unavailability.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | UNPACK |

```
SrcB Write Port Blocked = (SRCB_WRITE_AVAILABLE - SRCB_WRITE_NOT_BLOCKED_PORT) /
                          SRCB_WRITE_AVAILABLE * 100
```

- **High value (>50%)**: SrcB write port is heavily contended. Common across many ops (50-99%).
- **Low value (<20%)**: SrcB writes proceed with minimal blocking.

**Use case:** Detects srcB write port contention. High values are common on BH and may not indicate a problem unless combined with high unpack stall rates.

---

**17. Dest Read Backpressure**

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

**18. Math Dest Write Port Stall Rate**

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

**Use case:** Detects destination register write contention from the math side.

---

**19. Math Scoreboard Stall Rate**

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

**20. CFG/SYNC/THCON/MOVE/MATH/UNPACK/PACK Instrn Avail Rate**

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

**21. THCON/MOVE Idle Stall Pct T0**

What percentage of thread 0's total stalls are caused by THCON or MOVE waits.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
MOVE Idle Stall Pct T0 = WAITING_FOR_MOVE_IDLE_0 / THREAD_STALLS_0 * 100
THCON Idle Stall Pct T0 = WAITING_FOR_THCON_IDLE_0 / THREAD_STALLS_0 * 100
```

- **MOVE Idle high (>50%)**: Most of Thread 0's stalls are XMOV waits. Expected for data movement ops like tilize (94%), silu (95%).
- **THCON Idle high (>20%)**: Thread control operations are the bottleneck.
- **Both low**: Stalls are caused by other reasons (semaphores, unpack idle, etc.).

**Use case:** Narrows down the dominant stall reason for thread 0. If thread 0 has high stall rate, this tells you why.

---

**22. MMIO/SFPU Idle Stall Pct T1**

What percentage of thread 1's total stalls are caused by MMIO or SFPU waits.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | INSTRN |

```
MMIO Idle Stall Pct T1 = WAITING_FOR_MMIO_IDLE_1 / THREAD_STALLS_1 * 100
SFPU Idle Stall Pct T1 = WAITING_FOR_SFPU_IDLE_1 / THREAD_STALLS_1 * 100
```

- **SFPU Idle high (>30%)**: Math thread is waiting for SFPU pipeline to drain.
- **MMIO Idle high (>10%)**: Math thread is waiting for config register access.
- **N/A**: Thread 1 never stalled (denominator = 0). This is common and means the math thread runs without stalls.

**Use case:** When the math thread does stall, identifies whether it's SFPU pipeline depth or config access.

---

### Write Port Analysis

**23. SrcA/SrcB Write Actual Efficiency**

Fraction of source register write attempts that actually succeeded.

| | |
|---|---|
| **Architectures** | SrcA: Both. SrcB: Wormhole only (dead on BH). |
| **Counter group** | UNPACK |

```
SrcA Write Actual Efficiency = SRCA_WRITE_ACTUAL / SRCA_WRITE_AVAILABLE * 100
SrcB Write Actual Efficiency = SRCB_WRITE_ACTUAL / SRCB_WRITE_AVAILABLE * 100
```

- **High value (100%)**: Every write attempt succeeds. No write port blocking.
- **Low value (<80%)**: Significant fraction of writes are blocked.

**Use case:** Measures effective write throughput to source registers. Low values indicate write port contention.

---

### Additional Idle Waits

**24. MMIO/SFPU/THCON/MOVE Idle Wait**

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

**25. RISC Core L1 Util**

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

**26. L1 Unpacker/Packer Port Util**

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

**27. L1 TDMA Bundle Util**

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

**28. NOC Ring 0/1 Outgoing/Incoming Util**

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

**29. NOC Ring 0/1 Outgoing/Incoming Backpressure**

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

**30. L1 Unpacker/Packer Port Backpressure**

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

**Use case:** High unpacker BP is expected (unpacker competes with other ports). Investigate only if combined with high Thread 0 stall rate.

---

### L1 Composite Metrics

These metrics combine multiple counters to provide higher-level L1 insights.

**31. L1 Total Bandwidth Util**

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

**32. L1 Read vs Write Ratio**

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

**33. NOC Ring 0 Asymmetry**

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

**34. L1 Contention Index**

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

**35. Unpacker L1 Efficiency**

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

**36. Packer L1 Efficiency**

When the packer is busy, how often does L1 serve it.

| | |
|---|---|
| **Architectures** | Wormhole, Blackhole |
| **Counter group** | L1_0 + PACK |

```
Packer L1 Efficiency = min(100, L1_0_PORT1_GRANT / PACKER_BUSY * 100)
```

- **High value (100%)**: L1 always serves packer when busy. Matmul shows 100%.
- **Low value (<50%)**: Packer is being starved by L1.

**Note:** Capped at 100% because the packer port is shared with other clients (ECC on WH, other traffic on BH).

**Use case:** Confirms packer is not L1-bottlenecked. Low values would indicate L1 port contention affecting write-back.

---

**37. NOC vs Compute Balance**

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

**38. TDMA vs NOC L1 Share**

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

### Wormhole-Only Metrics

**39. HiFi2/LoFi/HiFi4 Instrn Rate**

Fraction of math instructions at each fidelity level.

| | |
|---|---|
| **Architectures** | Wormhole only |
| **Counter group** | UNPACK |

```
HiFi2 Rate = INSTRN_2_HF_CYCLES / MATH_INSTRN_STARTED * 100
LoFi Rate = INSTRN_1_HF_CYCLE / MATH_INSTRN_STARTED * 100
HiFi4 Rate = (MATH_INSTRN_STARTED - HF2 - HF1) / MATH_INSTRN_STARTED * 100
```

- **HiFi4 dominant**: High precision math, more cycles per instruction.
- **LoFi dominant**: Fast math, fewer cycles, lower precision.

**Not available on Blackhole**: `MATH_INSTRN_STARTED` signal is inactive on BH.

**Use case:** Confirms actual fidelity mode matches expectations. Unexpected HiFi4 when LoFi was intended wastes cycles.

---

**40. Packer Engine 0/1/2 Util**

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

**41. Unpacker0/1 Write Efficiency**

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

**42. FPU Execution Efficiency**

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

### Dead Signals on Blackhole

| Signal | Counter_sel | Reason |
|--------|------------|--------|
| `o_math_instrnbuf_rden` | 3, 256-259 | Math instruction buffer read signal inactive |
| `PACKER_BUSY_0/1/2` | 15-17 | `PACK_COUNT=1`, per-engine busy signals don't exist |
| `PACKER_DEST_READ_1/2/3` | 12-14 | Single packer engine, no multi-register access |
| `DEST_READ_GRANTED_1/2/3` | 268-270 | Same reason |

