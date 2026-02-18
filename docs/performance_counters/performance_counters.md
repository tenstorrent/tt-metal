# Tensix Performance Counters Guide

This guide documents the interface for controlling and reading Tensix hardware performance counters. It covers how to configure and read them from C++ and Python, which useful derived metrics you can compute, and how to test both the interface and the metrics. It consolidates the hardware documentation and aligns it with the implemented interfaces found in this repository.

## Overview

Tensix cores contain five hardware performance counter banks (also referred to as "groups" in some hardware documentation). This guide consistently uses the term "bank".

| Bank | Description |
|------|-------------|
| INSTRN_THREAD | Instruction issue counts and stall reasons per thread |
| FPU | FPU and SFPU operation valid signals |
| TDMA_UNPACK | Unpacker busy signals and math pipeline status |
| L1 | NoC ring transactions and L1 arbitration events |
| TDMA_PACK | Packer busy signals and destination register access |

Each category has debug registers for control and two output registers:
- `PERF_CNT_*0`: Reference period (optional, only used in mode 1)
- `PERF_CNT_*1`: Mode + counter select
- `PERF_CNT_*2`: Start/stop (rising-edge triggered)
- `PERF_CNT_OUT_L_*`: Cycles counted in the measurement window
- `PERF_CNT_OUT_H_*`: Event-specific count for the selected counter

Special registers:
- `PERF_CNT_ALL`: Start/stop for FPU and INSTRN_THREAD groups simultaneously
- `PERF_CNT_MUX_CTRL`: L1 counter multiplexer control register

### L1 Mux Control Register

The L1 bank has 8 counter IDs (0–7), but the hardware provides 16 different signals to monitor. The `PERF_CNT_MUX_CTRL` register (bit 4) selects which set of 8 signals maps to those IDs:

| Bit 4 | Counter ID 0–7 Maps To |
|-------|------------------------|
| 0 | NOC Ring 0 transactions, L1 arbitration bundles, unpacker arbitration |
| 1 | NOC Ring 1 transactions, TDMA bundle arbitration, extended unpacker/packer signals |

You must set this mux bit **before starting** the L1 counters to ensure you're measuring the intended signals. The L1 mux setting is latched at measurement start time. Changing `PERF_CNT_MUX_CTRL` while counters are running results in undefined measurements.

### Mode Register Bits

The mode register (`PERF_CNT_*1`) controls counter selection:

| Bits | Field | Description |
|------|-------|-------------|
| 7:0 | mode | Counting behavior (see below) |
| 16:8 | counter_sel | Selects which counter event to read via `OUT_H` (9 bits, supports counter IDs 0–511) |
| 31:17 | reserved | Reserved for future use |

**Counting modes:**

| Mode | Behavior |
|------|----------|
| 0 (Continuous) | Counter runs until explicitly stopped. Maintains reference cycle count in `OUT_L`. |
| 1 (Reference period) | Counter automatically stops after the number of cycles specified in `PERF_CNT_*0`. |
| 2 (Continuous, no ref) | Same as mode 0, but does NOT maintain the reference cycle count. |

**Mode 0 vs Mode 2:** Both run continuously until stopped. Mode 0 tracks elapsed cycles in `OUT_L` (needed for rate calculations), while mode 2 omits reference cycle tracking. **Use mode 0 for most cases** since you typically need both event counts and cycle counts. Mode 2 should only be used when cycle counts are not required.

**Note:** The reference period register is ignored in modes 0 and 2.

### Counter Slots

A "counter slot" is a single 32-bit configuration word in the per-thread L1 configuration buffer. Each slot (one config word) specifies which bank/counter/mux combination to read out after measurement. The interface supports up to 86 slots per thread, allowing you to capture multiple events in a single measurement window. Slots are processed in increasing index order; invalid slots (valid bit = 0) are skipped and produce no output entry.

## Register Map

The following addresses are used (offsets from `RISCV_DEBUG_REGS_START_ADDR`):

| Register | Offset | Description |
|----------|--------|-------------|
| PERF_CNT_INSTRN_THREAD0 | 0x000 | Reference period |
| PERF_CNT_INSTRN_THREAD1 | 0x004 | Mode |
| PERF_CNT_INSTRN_THREAD2 | 0x008 | Start/Stop |
| PERF_CNT_TDMA_UNPACK0 | 0x00C | Reference period |
| PERF_CNT_TDMA_UNPACK1 | 0x010 | Mode |
| PERF_CNT_TDMA_UNPACK2 | 0x014 | Start/Stop |
| PERF_CNT_FPU0 | 0x018 | Reference period |
| PERF_CNT_FPU1 | 0x01C | Mode |
| PERF_CNT_FPU2 | 0x020 | Start/Stop |
| PERF_CNT_L1_0 | 0x030 | Reference period |
| PERF_CNT_L1_1 | 0x034 | Mode |
| PERF_CNT_L1_2 | 0x038 | Start/Stop |
| PERF_CNT_ALL | 0x03C | Global start/stop |
| PERF_CNT_TDMA_PACK0 | 0x0F0 | Reference period |
| PERF_CNT_TDMA_PACK1 | 0x0F4 | Mode |
| PERF_CNT_TDMA_PACK2 | 0x0F8 | Start/Stop |
| PERF_CNT_OUT_L_INSTRN_THREAD | 0x100 | Cycle count output |
| PERF_CNT_OUT_H_INSTRN_THREAD | 0x104 | Counter value output |
| PERF_CNT_OUT_L_TDMA_UNPACK | 0x108 | Cycle count output |
| PERF_CNT_OUT_H_TDMA_UNPACK | 0x10C | Counter value output |
| PERF_CNT_OUT_L_TDMA_PACK | 0x110 | Cycle count output |
| PERF_CNT_OUT_H_TDMA_PACK | 0x114 | Counter value output |
| PERF_CNT_OUT_L_DBG_L1 | 0x118 | Cycle count output |
| PERF_CNT_OUT_H_DBG_L1 | 0x11C | Counter value output |
| PERF_CNT_OUT_L_FPU | 0x120 | Cycle count output |
| PERF_CNT_OUT_H_FPU | 0x124 | Counter value output |
| PERF_CNT_MUX_CTRL | 0x218 | L1 mux control |

Implementation details are in `tests/helpers/include/counters.h`.

## Events and Counters

### FPU Bank (3 counters)

| ID | Name | Description |
|----|------|-------------|
| 0 | FPU_INSTRUCTION | Cycles that FPU instructions were executed |
| 1 | SFPU_INSTRUCTION | Cycles that SFPU instructions were executed |
| 257 | FPU_OR_SFPU_INSTRN | Cycles that either FPU or SFPU instructions were executed (combined) |

Use these to measure compute utilization: `FPU_OR_SFPU_INSTRN / cycles`. Note that all utilization metrics are cycle-based, not instruction-based—they represent the fraction of active cycles, not work completed.

### INSTRN_THREAD Bank (61 counters)

**Instruction availability counters:** These counters measure cycles where an instruction of a given type was available in the instruction buffer and ready to issue. A high count means the thread frequently had that instruction type queued; a low count (relative to total cycles) suggests the thread was idle or blocked waiting for other work. These are useful for understanding instruction mix and identifying pipeline bottlenecks.

| ID | Name | Description |
|----|------|-------------|
| 0–2 | CFG_INSTRN_AVAILABLE_[0-2] | CFG instruction available per thread |
| 3–5 | SYNC_INSTRN_AVAILABLE_[0-2] | SYNC instruction available per thread |
| 6–8 | THCON_INSTRN_AVAILABLE_[0-2] | THCON instruction available per thread |
| 9–11 | XSEARCH_INSTRN_AVAILABLE_[0-2] | XSEARCH instruction available per thread |
| 12–14 | MOVE_INSTRN_AVAILABLE_[0-2] | MOVE instruction available per thread |
| 15–17 | FPU_INSTRN_AVAILABLE_[0-2] | FPU/SFPU instruction available per thread |
| 18–20 | UNPACK_INSTRN_AVAILABLE_[0-2] | UNPACK instruction available per thread |
| 21–23 | PACK_INSTRN_AVAILABLE_[0-2] | PACK instruction available per thread |
| 24–26 | THREAD_STALLS_[0-2] | Cycles the thread was stalled per thread |

**Wait reason counters:** Identify why the thread was waiting.

| ID | Name | Description |
|----|------|-------------|
| 27 | WAITING_FOR_SRCA_CLEAR | Cycles waiting for srcA to be cleared |
| 28 | WAITING_FOR_SRCB_CLEAR | Cycles waiting for srcB to be cleared |
| 29 | WAITING_FOR_SRCA_VALID | Cycles waiting for srcA data to be valid |
| 30 | WAITING_FOR_SRCB_VALID | Cycles waiting for srcB data to be valid |
| 31–33 | WAITING_FOR_THCON_IDLE_[0-2] | Cycles waiting for THCON idle per thread |
| 34–36 | WAITING_FOR_UNPACK_IDLE_[0-2] | Cycles waiting for unpack idle per thread |
| 37–39 | WAITING_FOR_PACK_IDLE_[0-2] | Cycles waiting for pack idle per thread |
| 40–42 | WAITING_FOR_MATH_IDLE_[0-2] | Cycles waiting for math idle per thread |
| 43–45 | WAITING_FOR_NONZERO_SEM_[0-2] | Cycles waiting for semaphore > 0 per thread |
| 46–48 | WAITING_FOR_NONFULL_SEM_[0-2] | Cycles waiting for semaphore < max per thread |
| 49–51 | WAITING_FOR_MOVE_IDLE_[0-2] | Cycles waiting for MOVE idle per thread |
| 52–54 | WAITING_FOR_MMIO_IDLE_[0-2] | Cycles waiting for MMIO idle per thread |
| 55–57 | WAITING_FOR_SFPU_IDLE_[0-2] | Cycles waiting for SFPU idle per thread |

**Thread instruction counts:** (Counter IDs 256–258, using bit 8 set)

| ID | Name | Description |
|----|------|-------------|
| 256–258 | THREAD_INSTRUCTIONS_[0-2] | Total instructions executed per thread |

### TDMA_UNPACK Bank (11 counters)

**Note on HF cycles:** "HF" refers to "Half-Fidelity" cycles. In Tensix math operations, fidelity settings control precision vs speed tradeoffs. Higher fidelity (HiFi) uses more cycles per operation for better precision; lower fidelity (LoFi) uses fewer cycles. The HF cycle counts indicate how many half-fidelity-equivalent cycles the instruction consumed.

| ID | Name | Description |
|----|------|-------------|
| 1 | DATA_HAZARD_STALLS_MOVD2A | Cycles stalled due to data hazards on MOVD2A |
| 3 | MATH_INSTRN_STARTED | Math instructions started |
| 4 | MATH_INSTRN_AVAILABLE | Math instructions available in buffer |
| 5 | SRCB_WRITE_AVAILABLE | Cycles SRCB write port was available |
| 6 | SRCA_WRITE_AVAILABLE | Cycles SRCA write port was available |
| 7 | UNPACK0_BUSY_THREAD0 | Unpacker 0 busy cycles (thread 0) |
| 8 | UNPACK1_BUSY_THREAD0 | Unpacker 1 busy cycles (thread 0) |
| 9 | UNPACK0_BUSY_THREAD1 | Unpacker 0 busy cycles (thread 1) |
| 10 | UNPACK1_BUSY_THREAD1 | Unpacker 1 busy cycles (thread 1) |
| 259 | SRCB_WRITE | Actual SRCB writes completed |
| 261 | SRCA_WRITE | Actual SRCA writes completed |

### TDMA_PACK Bank (3 counters)

| ID | Name | Description |
|----|------|-------------|
| 11 | PACKER_DEST_READ_AVAILABLE | Cycles destination data was available for packer to read |
| 18 | PACKER_BUSY | Cycles packer was actively working |
| 272 | AVAILABLE_MATH | Cycles math results were available for packing |

### L1 Bank (16 counters via mux)

**Mux bit 4 = 0:**

| ID | Name | Description |
|----|------|-------------|
| 0 | NOC_RING0_INCOMING_1 | NOC ring 0 incoming channel 1 read/write |
| 1 | NOC_RING0_INCOMING_0 | NOC ring 0 incoming channel 0 read/write |
| 2 | NOC_RING0_OUTGOING_1 | NOC ring 0 outgoing channel 1 read/write |
| 3 | NOC_RING0_OUTGOING_0 | NOC ring 0 outgoing channel 0 read/write |
| 4 | L1_ARB_TDMA_BUNDLE_1 | L1 arbitration for TDMA bundle 1 |
| 5 | L1_ARB_TDMA_BUNDLE_0 | L1 arbitration for TDMA bundle 0 |
| 6 | L1_ARB_UNPACKER | L1 arbitration for unpacker |
| 7 | L1_NO_ARB_UNPACKER | L1 no-arbitration unpacker path |

**Mux bit 4 = 1:**

| ID | Name | Description |
|----|------|-------------|
| 0 | NOC_RING1_INCOMING_1 | NOC ring 1 incoming channel 1 read/write |
| 1 | NOC_RING1_INCOMING_0 | NOC ring 1 incoming channel 0 read/write |
| 2 | NOC_RING1_OUTGOING_1 | NOC ring 1 outgoing channel 1 read/write |
| 3 | NOC_RING1_OUTGOING_0 | NOC ring 1 outgoing channel 0 read/write |
| 4 | TDMA_BUNDLE_1_ARB | TDMA bundle 1 arbitration |
| 5 | TDMA_BUNDLE_0_ARB | TDMA bundle 0 arbitration |
| 6 | TDMA_EXT_UNPACK_9_10 | TDMA extended unpacker interface |
| 7 | TDMA_PACKER_2_WR | TDMA packer 2 write interface to L1 |

## Memory Layout (per TRISC thread)

Configuration and data buffers in L1 (per thread). Layout: 86 config words (344 bytes) + 172 data words (688 bytes) = 1032 bytes per thread.

| Thread | Config Address | Data Address |
|--------|----------------|--------------|
| UNPACK | 0x16A000 (86 words) | 0x16A158 (172 words) |
| MATH | 0x16A408 (86 words) | 0x16A560 (172 words) |
| PACK | 0x16A810 (86 words) | 0x16A968 (172 words) |

**Config word encoding:** Each counter slot is a single 32-bit config word with the following format:

| Bit(s) | Field | Description |
|--------|-------|-------------|
| 31 | valid | 1 if this slot is active |
| 17 | l1_mux | L1 mux bit 4 value (only meaningful for L1 bank) |
| 16:8 | counter_id | Counter ID within the bank (9 bits, supports IDs 0–511) |
| 7:0 | bank_id | Bank ID (0=INSTRN_THREAD, 1=FPU, 2=TDMA_UNPACK, 3=L1, 4=TDMA_PACK) |

**Data area:** After measurement, contains interleaved `(cycles, count)` pairs (2 words each) for each valid slot.

## Interfaces

### C++ API

Location: `tests/helpers/include/counters.h`

**Types:**
- `llk_perf::CounterBank`: Enum with values `INSTRN_THREAD`, `FPU`, `TDMA_UNPACK`, `L1`, `TDMA_PACK`
- `llk_perf::CounterResult`: Struct with `cycles`, `count`, `bank`, `counter_id`

**Class: `llk_perf::PerfCounters`**

| Method | Description |
|--------|-------------|
| `start()` | Read config from L1 and start all configured banks. Configuration buffers must be initialized before calling `start()`. |
| `stop()` | Stop counters, scan all slots, write results to L1, return results array |

**RAII wrapper: `llk_perf::ScopedPerfCounters`**

Automatically starts on construction and stops on destruction. Use when Python pre-configures the counters.

**Example (Python-provided configuration):**
```cpp
#include "counters.h"

void run_kernel(const volatile RuntimeParams* params) {
    llk_perf::ScopedPerfCounters scoped;
    // ... kernel work ...
    // Results written to L1 on destruction
}
```

### Python API

Location: `tests/python_tests/helpers/counters.py`

**Functions:**

| Function | Description |
|----------|-------------|
| `configure_counters(location)` | Configure all 86 counters on all threads (UNPACK, MATH, PACK) |
| `read_counters(location)` | Read counter results from all threads |
| `print_counters(results)` | Display formatted results |
| `export_counters(results, filename, test_params, worker_id)` | Export counter results to CSV |

**Example:**
```python
from helpers.counters import configure_counters, read_counters, print_counters, export_counters

# Configure and run with performance counters
configure_counters(location="0,0")
# ... run kernel ...
results = read_counters(location="0,0")

# Display results
print_counters(results)

# Export to CSV
export_counters(results, "my_test_counters")
```

## Derived Metrics

Location: `tests/python_tests/helpers/metrics.py`

The metrics module computes higher-level performance indicators from raw counter data. Platform-specific bandwidth parameters are automatically detected based on the connected chip architecture.

### Usage

The metrics functions automatically detect the chip architecture and apply appropriate bandwidth parameters:

```python
from helpers.metrics import print_metrics, export_metrics

# Print formatted metrics to console
print_metrics(results)

# Export metrics to CSV
export_metrics(results, "my_test_metrics")
```

**Functions:**

| Function | Description |
|----------|-------------|
| `compute_metrics(results)` | Compute utilization metrics from counter data |
| `print_metrics(results)` | Print kernel metrics to console |
| `export_metrics(results, filename, test_params, worker_id)` | Export kernel metrics to CSV |

### Platform Bandwidth Parameters

The module internally maintains bandwidth parameters for each supported architecture:

| Platform | NoC Word (bytes) | Unpacker Peak (B/cyc) | Packer Peak (B/cyc) | Notes |
|----------|------------------|----------------------|---------------------|-------|
| Wormhole B0 | 32 | 80.0 | 80.0 | 256-bit NoC flit; 80 B/cyc from tt-metal perf model |
| Blackhole | 256 | 120.0 | 120.0 | 2048-bit NoC flit; wider bus enables higher bandwidth |
| Quasar | 32 | 80.0 | 80.0 | Placeholder values |

These values are used to convert raw utilization ratios into estimated bytes/cycle throughput.

### Output Interpretation

The metrics output shows both utilization and estimated bandwidth:

| Metric | Example Output |
|--------|----------------|
| Unpacker | `util=0.228 → est ~18.21 B/cyc` |
| Packer | `util=0.382 → est ~30.57 B/cyc` |
| NoC | `45.43 B/cyc (txn/cyc 1.42)` |

- **util** = fraction of cycles the component was busy (0.0–1.0)
- **est B/cyc** = `util × peak_bytes_per_cycle` = estimated actual throughput
- Values below peak indicate headroom; values near peak indicate saturation

### Core Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| FPU Utilization | `FPU_INSTRUCTION / cycles_fpu` | FPU activity rate |
| SFPU Utilization | `SFPU_INSTRUCTION / cycles_fpu` | SFPU activity rate |
| Math Utilization | `FPU_OR_SFPU_INSTRN / cycles_fpu` | Combined FPU+SFPU activity rate |
| Unpacker0 Utilization | `SRCA_WRITE / UNPACK0_BUSY_THREAD0` | Unpacker 0 efficiency (writes per busy cycle) |
| Unpacker1 Utilization | `SRCB_WRITE / UNPACK1_BUSY_THREAD0` | Unpacker 1 efficiency (writes per busy cycle) |
| Packer Utilization | `PACKER_BUSY / cycles_pack` | Packer activity rate |

**Interpretation:**
- **Compute utilization** (FPU/SFPU/Math): Values 0.0–1.0 indicate fraction of cycles actively computing
- **Unpacker utilization**: Ratio of writes to busy cycles; values near 1.0 indicate efficient data flow with minimal stalls
- **Packer utilization**: Fraction of cycles the packer was busy; low values may indicate the kernel is compute-bound

### Bound Classification

The metrics module provides a heuristic classification of the performance bottleneck:

| Classification | Score Based On |
|----------------|----------------|
| math-bound | Compute utilization |
| unpack-bound | Unpack utilization |
| pack-bound | Pack utilization |
| risc-bound | Stall rate minus instruction issue efficiency |

The classification with the highest score is reported as the likely bottleneck.

**Note:** Bound classification is heuristic and intended for guidance, not definitive performance assessment. The "compute vs data-movement bound" classification is more relevant for tt-metal where data movement is a significant factor. In the LLK repository context, focus on the individual component bounds (math/unpack/pack/risc).
