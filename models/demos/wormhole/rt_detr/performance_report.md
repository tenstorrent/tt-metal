
# RT-DETR Performance Analysis Report

## Profiling Environment

| Field | Value |
| :--- | :--- |
| **Model Name** | RT-DETR (Real-Time Detection Transformer) |
| **Batch Size** | 1 |
| **Hardware** | Wormhole B0 (wormhole_b0) |
| **Model Type** | Object Detection |
| **Date** | 2026-05-25 |
| **Total Operations Traced** | 1,860 |
| **Max Available Worker Core Count** | 64 |
| **Device Architecture** | Wormhole B0 |

---

## Performance Overview

| Metric | Value |
| :--- | :--- |
| **Host Time** | 139.95 ms |
| **Device Time (FW)** | 61.11 ms |
| **Total Device Kernel Time** | 59.05 ms |
| **FW Overhead** | ~2.06 ms (~3.4%) |
| **Host-to-Device Efficiency** | ~43.7% of host wall-clock is on-device compute |

The device finishes its share of the work in 61.1 ms, but the host clock shows 139.95 ms by the time everything wraps up. That extra ~79 ms is Python overhead, dispatch scheduling, and synchronisation between passes — not idle device time. The device is executing efficiently; the gap reflects the cost of keeping it fed from the host side.

Zooming into the device side, the 1,860 ops traced add up to 59.05 ms of actual kernel execution, landing almost exactly at the 61.11 ms firmware time. The ~3.4% difference is bookkeeping — kernel entry and exit overhead — and is about as low as expected for a well-utilised trace.

---

## Perf Report Header Reference

The CSV trace follows the standard TT-NN profiler schema. Each row represents one dispatched operation. The key columns and their meaning:

| Header | Description |
| :--- | :--- |
| **OP CODE** | Class name of the C++ operation (e.g. `MatmulDeviceOperation`) |
| **OP TYPE** | Where the op ran: `tt_dnn_device` = on Tensix, `tt_dnn_cpu` = C++ on CPU, `python_fallback` = host Python |
| **GLOBAL CALL COUNT** | Monotonically increasing index of the op in the execution pipeline |
| **DEVICE ID** | Which physical device handled this op (relevant for multi-chip N300 configs) |
| **ATTRIBUTES** | Op-specific configuration: dtype, memory layout, fused flags, etc. |
| **MATH FIDELITY** | Precision mode used by Tensix compute: LoFi → HiFi2 → HiFi3 → HiFi4 |
| **CORE COUNT** | Number of Tensix cores used for this dispatch |
| **PARALLELIZATION STRATEGY** | How the kernel tiles work across cores (e.g. interleaved, sharded) |
| **HOST START / END TS** | System clock timestamps bracketing the full op including dispatch |
| **HOST DURATION [ns]** | Wall-clock time from the host side — includes Python, dispatch, and device wait |
| **DEVICE FW START / END CYCLE** | Raw Tensix cycle counts from first-to-start and last-to-finish RISC cores |
| **DEVICE FW DURATION [ns]** | Time from earliest FW entry to latest FW exit across all cores and RISCs |
| **DEVICE KERNEL DURATION [ns]** | Pure compute time from first kernel start to last kernel end, across all cores |
| **DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]** | Per-core spread; large min-max gaps indicate load imbalance |
| **DEVICE KERNEL FIRST TO LAST START [ns]** | Stagger between when the earliest and latest cores begin — a measure of dispatch skew |
| **DEVICE BRISC KERNEL DURATION [ns]** | Time on the Bridge RISC (data movement reader/writer paths) |
| **DEVICE NCRISC KERNEL DURATION [ns]** | Time on the NoC RISC (NoC data movement) |
| **DEVICE TRISC0/1/2 KERNEL DURATION [ns]** | Time on the three compute RISCs (TRISC0 = unpack, TRISC1 = math, TRISC2 = pack) |
| **DEVICE COMPUTE CB WAIT FRONT [ns]** | Time TRISC0 spent waiting for input circular buffers — indicates starvation |
| **DEVICE COMPUTE CB RESERVE BACK [ns]** | Time TRISC2 spent waiting to write output — indicates backpressure |
| **DISPATCH TOTAL CQ CMD OP TIME [ns]** | Time to push the op's command queue entries |
| **DISPATCH GO SEND WAIT TIME [ns]** | Time from GO signal sent to execution start |
| **OP TO OP LATENCY [ns]** | Gap between consecutive ops as seen from the device |
| **INPUT_x / OUTPUT_x** | Per-tensor shape (W/Z/Y/X), layout (TILE/ROW\_MAJOR), dtype, and memory location |
| **PM IDEAL / COMPUTE / BANDWIDTH [ns]** | Performance model estimates under ideal, compute-bound, and bandwidth-bound conditions |
| **PM FPU UTIL (%)** | Estimated floating-point utilisation on the Tensix FPUs |
| **NOC UTIL (%) / MULTICAST NOC UTIL (%)** | Network-on-chip utilisation (unicast and multicast) |
| **DRAM BW UTIL (%)** | DRAM bandwidth utilisation |
| **PROGRAM CACHE HIT** | Whether the compiled program was re-used from cache (True) or recompiled (False) |
| **COMPUTE / DATA MOVEMENT KERNEL SOURCE & HASH** | Kernel file paths and cache keys |

---

## Section 1 — Device Time Breakdown

### Cumulative device time by operation class

| Operation | Calls | Total Kernel Time | Share | Avg Cores |
| :--- | ---: | ---: | ---: | ---: |
| MatmulDeviceOperation | 206 | 16.23 ms | 27.5% | 30.3 |
| TilizeWithValPaddingDeviceOperation | 194 | 11.97 ms | 20.3% | 3.0 |
| Conv2dDeviceOperation | 66 | 6.53 ms | 11.1% | 61.3 |
| BinaryNgDeviceOperation | 278 | 6.31 ms | 10.7% | 64.0 |
| CopyDeviceOperation | 154 | 3.45 ms | 5.8% | 64.0 |
| UnaryDeviceOperation | 122 | 2.73 ms | 4.6% | 64.0 |
| TilizeDeviceOperation | 96 | 2.44 ms | 4.1% | 17.3 |
| ReshapeViewDeviceOperation | 54 | 2.04 ms | 3.5% | 64.0 |
| TypecastDeviceOperation | 250 | 1.79 ms | 3.0% | 28.6 |
| InterleavedToShardedDeviceOperation | 74 | 1.15 ms | 2.0% | 60.8 |
| ShardedToInterleavedDeviceOperation | 74 | 1.00 ms | 1.7% | 60.8 |
| SDPAOperation | 14 | 0.73 ms | 1.2% | 64.0 |
| LayerNormDeviceOperation | 40 | 0.60 ms | 1.0% | 10.3 |
| HaloDeviceOperation | 74 | 0.49 ms | 0.8% | 60.8 |
| Other (9 op types) | 162 | 1.27 ms | 2.2% | — |

Matmul and `TilizeWithValPadding` together account for 47.8% of total device kernel time and are the primary targets for optimisation.

---

## Section 2 — Hardware and Compute Configuration

### Core utilisation

1,042 of 1,860 ops (56%) run at the full 64-core count. Notable exceptions:

- `TilizeWithValPaddingDeviceOperation` averages 3 cores per call despite consuming 20.3% of device time. This is the most underutilised op in the pipeline. The input tensors at this stage are likely small enough that the multicore tiling strategy is not triggering, or `use_multicore` is not being set for the `ValPadding` variant. This is worth investigating — widening the shard grid could recover meaningful time.
- `MatmulDeviceOperation` averages ~30 cores. Expected for the smaller feature maps in the transformer decoder, but larger backbone attention matrices should be reaching 62–64 cores. The HiFi2 matmuls (122 calls) average 4× longer than HiFi4 (84 calls), which at first appears counterintuitive. In practice, those longer matmuls correspond to larger tensor shapes, and the shape complexity dominates the fidelity difference.
- `LayerNormDeviceOperation` averages 10.3 cores across 40 calls. LayerNorm's channel-wide reduction makes full parallelisation inherently harder, but 10 out of 64 cores is on the low side and may warrant a closer look at the reduction sharding strategy.

### Math fidelity

1,084 ops (58%) run at HiFi4 and 268 (14%) at HiFi2. HiFi4 is applied to the attention mechanism, LayerNorm, and precision-sensitive decoder paths. HiFi2 is used in the convolutional backbone where throughput is prioritised over precision. The split is appropriate for this model type.

### Data types and memory layout

BFLOAT16 dominates: 1,360 input tensors and 1,610 output tensors. FLOAT32 appears on 500 input tensors, primarily in initial tilize steps and accumulation paths. The 250 `TypecastDeviceOperation` calls (FLOAT32 → BFLOAT16) each average ~7.2 µs, so the conversion overhead is not a bottleneck.

| Location | Input tensors | Output tensors |
| :--- | ---: | ---: |
| DEV_0_DRAM_INTERLEAVED | 862 | 864 |
| DEV_0_L1_INTERLEAVED | 714 | 712 |
| DEV_0_L1_BLOCK_SHARDED | 196 | 196 |
| DEV_0_L1_HEIGHT_SHARDED | 88 | 88 |

Roughly 46% of tensors reside in L1, with 284 using sharded layouts. The 74 `InterleavedToSharded` and 74 `ShardedToInterleaved` transitions cost ~2.16 ms combined — acceptable given the memory bandwidth benefit for the ops in between.

---

## Section 3 — Firmware Overhead Analysis

The delta between `DEVICE FW DURATION` and `DEVICE KERNEL DURATION` captures per-op firmware bookkeeping. High ratios typically flag short-running kernels where the fixed overhead is disproportionate.

| Operation | FW overhead |
| :--- | ---: |
| NLPConcatHeadsDeviceOperation | 22.9% |
| MoveDeviceOperation | 22.5% |
| HaloDeviceOperation | 12.6% |
| TypecastDeviceOperation | 12.1% |
| TransposeDeviceOperation | 11.7% |
| LayerNormDeviceOperation | 7.7% |
| CopyDeviceOperation | 5.6% |
| MatmulDeviceOperation | 2.3% |
| SDPAOperation | 1.9% |
| Conv2dDeviceOperation | 1.8% |
| TilizeWithValPaddingDeviceOperation | 1.4% |

`MoveDeviceOperation` and `NLPConcatHeadsDeviceOperation` both exceed 22% overhead. Both are short-running (3.4 µs and 4.4 µs average kernel time), so the fixed firmware cost represents a large fraction of their FW duration. At current call counts (62 and 2 respectively), the absolute impact is small (~213 µs and ~9 µs total kernel time), but fusing either into adjacent operations would improve the ratio if they appear in critical paths.

`MatmulDeviceOperation`, `SDPAOperation`, and `Conv2dDeviceOperation` all sit below 3% — the expected range for longer-running, well-utilised kernels.

---

## Section 4 — PyTorch Fallbacks

All 1,860 operations carry `OP TYPE: tt_dnn_device` and a valid `DEVICE FW DURATION [ns]` value. No operations fell back to PyTorch CPU execution, meaning the full inference graph has been lowered to Tenstorrent device kernels. There are no device-to-host or host-to-device transfer penalties from fallback ops, and no CPU synchronisation stalls during the forward pass.

---

## Section 5 — Program Cache

All operations show `PROGRAM CACHE HIT: False`. This is expected for a first-run trace — the cache is populated on the first inference pass and reused on subsequent runs. Per Tenstorrent's profiling documentation, only host timings from the second run should be treated as steady-state, since the first pass includes compilation overhead.

The 139.95 ms host time captured here likely includes this compilation cost. Device kernel times are already representative of steady-state execution, as they measure actual Tensix cycles independent of cache state. A cache-warm trace would be needed to establish a reliable host-side baseline.

---

## Section 6 — Op-to-Op Latency

Op-to-op latency measures the gap between consecutive operations as seen from the device. Large spikes indicate idle periods on the device, typically caused by host scheduling delays, synchronisation waits, or layout transitions.

### Largest observed spikes

| OP CODE | Global Call Count | Op-to-Op Latency | Kernel Duration |
| :--- | ---: | ---: | ---: |
| TilizeWithValPaddingDeviceOperation | 254,976 | 4,699.5 ms | 1.21 ms |
| TilizeWithValPaddingDeviceOperation | 254,977 | 4,699.5 ms | 1.21 ms |
| TilizeWithValPaddingDeviceOperation | 761,856 | 857.6 ms | 0.016 ms |
| TilizeWithValPaddingDeviceOperation | 761,857 | 857.6 ms | 0.016 ms |
| MatmulDeviceOperation | 538,624 | 401.9 ms | 0.17 ms |
| MatmulDeviceOperation | 538,625 | 401.9 ms | 0.17 ms |

The ~4.7-second spikes at call counts 254,976–254,977 are inter-inference idle time rather than intra-inference stalls. Global call counts in the 250,000+ range indicate this trace covers multiple inference passes; the gap represents the host preparing and loading the next input before the device resumes. The same pattern applies to the Matmul spikes at call counts 538,624–538,625.

For real-time deployment, pipelining input preparation in parallel with device inference would eliminate this idle time.

---

## Section 7 — Key Operations

### Matmul (27.5% of device time)

The 206 matmul calls split into two fidelity bands:
- HiFi2 (122 calls): average kernel duration 113.7 µs. Larger transformer attention and projection matmuls.
- HiFi4 (84 calls): average kernel duration 28.0 µs. Smaller, precision-sensitive decoder matmuls.

The 4× duration difference is driven primarily by tensor shape. Whether any of the 84 HiFi4 calls could tolerate HiFi2 without accuracy regression is worth evaluating. Average core count of 30 across all matmuls also suggests the larger-shape variants may not be fully parallelised — checking `use_matmul_1d_systolic_array` or output tiling options for the bigger attention matrices is a reasonable next step.

### TilizeWithValPadding (20.3% of device time)

Converts ROW_MAJOR input tensors to TILE layout while padding to tile boundaries. At 3 cores on average across 194 calls, this op is the most underutilised in the trace relative to its time cost. Confirming that `use_multicore: true` is set for the `ValPadding` variant (it is set for `TilizeDeviceOperation` per the CSV attributes) and verifying input shapes are large enough to benefit from parallelisation would be the first step.

### Conv2d (11.1% of device time)

66 calls averaging 99 µs each, with 61.3 cores on average and 1.8% FW overhead. Convolutions are well-parallelised and the memory layout transitions to block-sharded format are functioning as intended.

### SDPA (1.2% of device time)

14 calls, all using 64 cores, averaging 51.8 µs kernel time and 1.9% FW overhead. The attention mechanism is executing cleanly on-device. Low DRAM pressure here suggests KV tensors are fitting within L1 shards for the sequence lengths in use.

### LayerNorm (1.0% of device time)

40 calls averaging 14.9 µs, with 10.3 cores on average and 7.7% FW overhead. The sequential reduction dependency limits parallelism, but the core count is low enough that exploring width-sharded reduction strategies may be worthwhile.

---

## Section 8 — Recommendations

### Near-term

1. **TilizeWithValPadding parallelisation.** Averaging 3 cores while consuming 20% of device time is the most actionable gap in this trace. Widening the tiling strategy to 16–32 cores could recover 6–10 ms per inference pass.

2. **Cache-warm profiling run.** The current host time includes first-run compilation overhead. A second-pass trace with `PROGRAM CACHE HIT: True` is needed to establish a reliable steady-state host latency baseline.

3. **Host-side input pipelining.** The multi-second op-to-op latency spikes at inference boundaries indicate the host is not preparing inputs in parallel with device execution. A double-buffered preprocessing pipeline would eliminate this idle time.

### Medium-term

4. **Fidelity evaluation for HiFi4 matmuls.** 84 calls averaging 28 µs each. If any subset can tolerate HiFi2 without accuracy regression, switching would reduce their contribution to the 16.2 ms matmul total.

5. **Memory layout transition reduction.** 74 Interleaved-to-Sharded and 74 Sharded-to-Interleaved calls cost ~2.16 ms combined. Where adjacent ops can share a layout, some of these transitions can be eliminated.

6. **MoveDeviceOperation fusion.** 62 calls with 22% FW overhead and 3.4 µs average kernel time. Short enough to consider folding into adjacent copy or reshape operations where the op graph permits.

---
