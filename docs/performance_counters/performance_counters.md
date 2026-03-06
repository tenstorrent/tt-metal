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

## Memory Layout (Shared Buffer Architecture)

Configuration and data buffers in L1 use a **single shared buffer** accessed by all threads (UNPACK, MATH, PACK). Layout: 86 config words (344 bytes) + 172 data words (688 bytes) + 1 sync control word = 1036 bytes total.

| Buffer Component | Address | Size | Description |
|-----------------|---------|------|-------------|
| Config Buffer | 0x16A000 | 86 words (344 bytes) | Counter slot configurations (shared) |
| Data Buffer | 0x16A158 | 172 words (688 bytes) | Counter results (written by last stopper) |
| Sync Control Word | 0x16A408 | 1 word (4 bytes) | Synchronization state and last stopper ID |

**Shared Buffer Semantics:**
- All threads (UNPACK, MATH, PACK) call `start_perf_counters()` to set their start bits
- The **first thread to call start** (when all start bits are 0) initializes hardware and is recorded as the "starter"
- All threads call `stop_perf_counters()` to set their stop bits
- The **last thread to call stop** (when all 3 stop bits become set) reads hardware counters, writes results to the shared data buffer, and is recorded as the "stopper"
- Python `read_counters()` returns the snapshot captured by the stopper, along with both starter and stopper thread IDs
- **No mutex required**: Each thread atomically sets its own bit; checks are simple bit masks
- This reduces memory usage from 3096 bytes (3 × 1032) to 1036 bytes

**Sync Control Word Format (0x16A408):**

| Bit(s) | Field | Description |
|--------|-------|-------------|
| 0 | started_unpack | UNPACK thread called start_perf_counters() |
| 1 | started_math | MATH thread called start_perf_counters() |
| 2 | started_pack | PACK thread called start_perf_counters() |
| 3 | stopped_unpack | UNPACK thread called stop_perf_counters() |
| 4 | stopped_math | MATH thread called stop_perf_counters() |
| 5 | stopped_pack | PACK thread called stop_perf_counters() |
| 6 | started_global | At least one thread started counters |
| 7 | stopped_global | All threads stopped counters |
| 9:8 | starter_id | Which thread started hardware (0=UNPACK, 1=MATH, 2=PACK) |
| 11:10 | stopper_id | Which thread stopped hardware (0=UNPACK, 1=MATH, 2=PACK) |
| 31:12 | reserved | Reserved for future use |

**Config word encoding:** Each counter slot is a single 32-bit config word with the following format:

| Bit(s) | Field | Description |
|--------|-------|-------------|
| 31 | valid | 1 if this slot is active |
| 17 | l1_mux | L1 mux bit 4 value (only meaningful for L1 bank) |
| 16:8 | counter_id | Counter ID within the bank (9 bits, supports IDs 0–511) |
| 7:0 | bank_id | Bank ID (0=INSTRN_THREAD, 1=FPU, 2=TDMA_UNPACK, 3=L1, 4=TDMA_PACK) |

**Data area:** After measurement, contains interleaved `(cycles, count)` pairs (2 words each) for each valid slot.

## Performance Metrics

Location: `tests/python_tests/helpers/metrics.py`

The metrics module computes derived efficiency metrics from raw counter data. All metrics are ratios (0.0 to 1.0) where higher values indicate better efficiency.

### Base Metrics

**1. Unpacker Write Efficiency**

Measures how efficiently unpackers use their busy cycles for writing data.

```
Unpacker0 Efficiency = SRCA_WRITE / UNPACK0_BUSY_THREAD0
Unpacker1 Efficiency = SRCB_WRITE / UNPACK1_BUSY_THREAD0
Combined Efficiency = Average of both unpackers
```

- **Higher ratio (→1.0)**: Unpacker spends most busy time writing data (efficient)
- **Lower ratio (→0.0)**: Unpacker is busy but not writing (stalled/waiting)

**Use case:** Identifies unpacker stalls due to L1 memory contention or data dependencies.

**2. Packer Efficiency**

Measures how often the packer has valid destination data available when busy.

```
Packer Efficiency = PACKER_DEST_READ_AVAILABLE / PACKER_BUSY
```

- **Higher ratio (→1.0)**: Packer has data available when busy (efficient)
- **Lower ratio (→0.0)**: Packer is busy but waiting for destination data

**Note:** Only valid with HW dvalid-based synchronization, not with STALLWAIT mode.

**Use case:** Detects destination register stalls indicating math stage bottleneck.

**3. FPU Execution Efficiency**

Measures how efficiently the FPU executes when FPU instructions are available.

```
FPU Efficiency = FPU_INSTRUCTION / FPU_INSTRN_AVAILABLE_1
```

- **Higher ratio (→1.0)**: FPU executes whenever work is available (efficient)
- **Lower ratio (→0.0)**: FPU instructions available but not executing (pipeline stalls)

**Use case:** Distinguishes compute-bound (high FPU efficiency) from memory-bound (low FPU efficiency) workloads.

### Experimental Metrics

These metrics provide additional pipeline analysis but may require additional tests and reviews.

**4. Math Pipeline Utilization (EXPERIMENTAL)**

Measures math instruction flow efficiency through the pipeline.

```
Math Pipeline Utilization = MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE
```

- **Higher ratio (→1.0)**: Math pipeline efficiently moves instructions (no pipe stalls)
- **Lower ratio (→0.0)**: Instructions in pipe but not starting (pipeline stalled)

**Use case:** Detects math pipeline stalls. Values consistently at 1.0 indicate excellent pipeline health.

**5. Math-to-Pack Handoff Efficiency (EXPERIMENTAL)**

Measures pipeline balance between math and packer stages.

```
Math-to-Pack Efficiency = AVAILABLE_MATH / PACKER_BUSY
```

- **Higher ratio (→1.0)**: Math keeps up with packer demand (good balance)
- **Lower ratio (→0.0)**: Packer busy but math output isn't ready (math bottleneck)

**Use case:** Identifies math stage as bottleneck when packer is starved for data.

**6. Unpacker-to-Math Data Flow (EXPERIMENTAL)**

Measures backpressure from math stage to unpackers.

```
Unpacker0 Data Flow = SRCA_WRITE_AVAILABLE / UNPACK0_BUSY_THREAD0
Unpacker1 Data Flow = SRCB_WRITE_AVAILABLE / UNPACK1_BUSY_THREAD0
Combined Data Flow = Average of both unpackers
```

- **Higher ratio (→1.0)**: Unpacker can write when busy, no backpressure (efficient)
- **Lower ratio (→0.0)**: Unpacker busy but buffers full (math not consuming fast enough)

**Use case:** Detects math stage backpressure causing unpacker stalls. Compare with Unpacker Write Efficiency to distinguish backpressure stalls from other stall types.

### Metrics API

**Functions:**

| Function | Description |
|----------|-------------|
| `compute_metrics(df)` | Compute all metrics from counter DataFrame. Returns dictionary with raw counts, ratios, and percentages. |
| `print_metrics(results)` | Print formatted metrics report with explanations and efficiency ratios. |
| `export_metrics(results, filename, test_params, worker_id)` | Export metrics to CSV in `perf_data/` directory. |

**Example usage:**

```python
from helpers.counters import configure_counters, read_counters
from helpers.metrics import print_metrics

# Configure and run test
configure_counters(location="0,0")
# ... run kernel ...
results = read_counters(location="0,0")

# Display metrics
print_metrics(results)
```

**Output format:**

```
======================================================================
PERFORMANCE METRICS
======================================================================

──────────────────────────────────────────────────────────────────────
  UNPACKER WRITE EFFICIENCY
──────────────────────────────────────────────────────────────────────
  Measures the fraction of unpacker busy cycles spent writing data.
  Higher ratio (→1.0) = efficient, unpacker writes when busy
  Lower ratio (→0.0) = inefficient, unpacker busy but stalled/waiting
──────────────────────────────────────────────────────────────────────
  Metric                           Writes         Busy   Efficiency
  ────────────────────────────── ──────────── ──────────── ────────────
  Unpacker0 (SRCA):                   1234.0      2468.0         0.50
  Unpacker1 (SRCB):                   1356.0      2468.0         0.55
  Combined Average:                                                0.52

[... similar sections for each metric ...]
```

## Interfaces

### Python API

Location: `tests/python_tests/helpers/counters.py`

**Key Features:**
- Single source of truth: All constants derived from `TestConfig`
- Comprehensive validation: Detects missing start/stop calls with specific thread identification
- Shared buffer support: Returns last stopper's counter snapshot
- Pre-built counter configurations for common measurement scenarios

**Functions:**

| Function | Description |
|----------|-------------|
| `configure_counters(location="0,0")` | Write counter configuration to shared L1 buffer. Configures all 94 counter definitions (61 INSTRN_THREAD + 3 FPU + 11 TDMA_UNPACK + 3 TDMA_PACK + 16 L1). Note: Hardware has 86 slots; L1 counters are mux-dependent (8 active at once). Clears data buffer and sync control word. |
| `read_counters(location="0,0")` | Read counter results from shared buffer. Returns DataFrame with columns: `starter_thread`, `stopper_thread`, `bank`, `counter_name`, `counter_id`, `cycles`, `count`, `l1_mux`. Validates sync state and identifies missing start/stop calls. |
| `print_counters(results)` | Print counter results in human-readable format with thread identification. |
| `export_counters(results, filename, test_params, worker_id)` | Export counter DataFrame to CSV in `perf_data/` directory. |

**Validation:**

The `read_counters()` function performs comprehensive validation:
- **Zero sync word**: Detects if counters were never started (all threads forgot to call `start_perf_counters()`)
- **Missing global start**: At least one thread must call `start_perf_counters()`
- **Missing global stop**: All threads must call `stop_perf_counters()`. Reports which specific threads (UNPACK, MATH, PACK) are missing stop calls
- **Invalid thread IDs**: Validates both starter and stopper thread IDs are in valid range (0-2)
- **Error messages include sync_ctrl value**: All validation errors display the raw sync control word for debugging

**Example validation error:**
```
RuntimeError: Perf counters were not stopped by all threads.
Missing stop_perf_counters() call from: MATH. sync_ctrl=0x0000006c
```

**Pre-built Counter List:**

The module provides `ALL_COUNTERS` list with 94 pre-configured counter definitions:
- INSTRN_THREAD: 61 counters (instruction availability, wait reasons, thread instruction counts)
- FPU: 3 counters (FPU, SFPU, combined)
- TDMA_UNPACK: 11 counters (unpack busy, write availability, math pipeline)
- L1: 16 counters (8 counter IDs × 2 mux settings)
  - mux=0: 8 counters (NOC Ring 0, L1 arbitration)
  - mux=1: 8 counters (NOC Ring 1, TDMA arbitration)
- TDMA_PACK: 3 counters (packer busy, dest available, math available)

**Note:** The hardware supports 86 counter slots. L1 counters require setting the mux bit before measurement, so only 8 of the 16 L1 counter definitions can be active simultaneously. This means maximum concurrent counters = 61 + 3 + 11 + 3 + 8 = 86.

### C++ API

Location: `tests/helpers/include/counters.h`

**Types:**
- `llk_perf::counter_bank`: Enum with values `instrn_thread`, `fpu`, `tdma_unpack`, `l1`, `tdma_pack`

**Singleton Manager: `llk_perf::PerfCounterManager`**

The `PerfCounterManager` class manages performance counter lifecycle using a singleton pattern. Direct access to the singleton is not necessary; use the public API functions instead.

**Public API Functions:**

| Function | Description |
|----------|-------------|
| `llk_perf::start_perf_counters()` | Read config from shared L1 buffer and start all configured banks. Atomically sets the thread's start bit in sync control word. First thread to call this (when all start bits are 0) initializes hardware and is recorded as the "starter". **All threads must call this.** Thread-safe via atomic bit operations (no mutex needed). |
| `llk_perf::stop_perf_counters()` | Stop all configured banks and atomically set the thread's stop bit. Last thread to call this (when all 3 stop bits become set) reads hardware counters, writes results to the shared data buffer, and is recorded as the "stopper". **All threads must call this.** Thread-safe via atomic bit operations (no mutex needed). |

**Example usage:**
```cpp
#include "counters.h"

void llk_unpack_main() {
    llk_perf::start_perf_counters();
    // ... unpack work ...
    llk_perf::stop_perf_counters();
}

void llk_math_main() {
    llk_perf::start_perf_counters();
    // ... math work ...
    llk_perf::stop_perf_counters();
}

void llk_pack_main() {
    llk_perf::start_perf_counters();
    // ... pack work ...
    llk_perf::stop_perf_counters();
}
```
