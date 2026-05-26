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

The most immediate takeaway from this profiling run: the device is executing in **61.1 ms**, but the host wall-clock reads **139.95 ms**. That gap - roughly **79 ms** - is not wasted device time. It represents host-side scheduling, dispatch overhead, Python runtime, and any synchronisation points between inference passes. The device itself is doing work efficiently; the challenge is keeping it fed without interruption.

Across all 1,860 traced operations, total device kernel execution adds up to **59.05 ms**, which closely matches the device FW time of 61.11 ms. The ~3.4% difference between kernel and FW duration represents genuine firmware overhead (entry/exit bookkeeping around each kernel), which is healthy - it means the compute is doing real work rather than spinning in firmware preamble.

---

## Perf Report Header Reference

The CSV trace follows the standard TT-NN profiler schema. Each row represents one dispatched operation. The key columns and their meaning:

| Header | What It Tells You |
| :--- | :--- |
| **OP CODE** | Class name of the C++ operation (e.g. `MatmulDeviceOperation`) |
| **OP TYPE** | Where the op ran: `tt_dnn_device` = on Tensix, `tt_dnn_cpu` = C++ on CPU, `python_fallback` = host Python |
| **GLOBAL CALL COUNT** | Monotonically increasing index of the op in the pipeline - useful for correlating spikes across devices |
| **DEVICE ID** | Which physical device handled this op (relevant for multi-chip N300 configs) |
| **ATTRIBUTES** | Op-specific config blob: dtype, memory layout, fused flags, etc. |
| **MATH FIDELITY** | Precision mode used by Tensix compute: LoFi → HiFi2 → HiFi3 → HiFi4 (higher = more cycles, more accuracy) |
| **CORE COUNT** | How many Tensix cores were used for this specific dispatch |
| **PARALLELIZATION STRATEGY** | How the kernel tiles work across cores (e.g. interleaved, sharded) |
| **HOST START / END TS** | System clock timestamps bracketing the full op (including dispatch) |
| **HOST DURATION [ns]** | Wall-clock time seen from the host - includes Python, dispatch, and device wait |
| **DEVICE FW START / END CYCLE** | Raw Tensix cycle counts from first-to-start and last-to-finish RISC cores |
| **DEVICE FW DURATION [ns]** | Time from earliest FW entry to latest FW exit across all cores and RISCs |
| **DEVICE KERNEL DURATION [ns]** | Pure compute time - from first kernel start to last kernel end, across all cores |
| **DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]** | Per-core spread; large min-max gaps indicate load imbalance |
| **DEVICE KERNEL FIRST TO LAST START [ns]** | Stagger between when the earliest and latest cores begin - a measure of dispatch skew |
| **DEVICE BRISC KERNEL DURATION [ns]** | Time on the Bridge RISC (data movement reader/writer paths) |
| **DEVICE NCRISC KERNEL DURATION [ns]** | Time on the NoC RISC (NoC data movement) |
| **DEVICE TRISC0/1/2 KERNEL DURATION [ns]** | Time on the three compute RISCs (TRISC0 = unpack, TRISC1 = math, TRISC2 = pack) |
| **DEVICE COMPUTE CB WAIT FRONT [ns]** | How long TRISC0 waited for input circular buffers to be filled (starvation signal) |
| **DEVICE COMPUTE CB RESERVE BACK [ns]** | How long TRISC2 waited to write output (backpressure signal) |
| **DISPATCH TOTAL CQ CMD OP TIME [ns]** | Time to push the op's command queue entries - dispatch bottleneck indicator |
| **DISPATCH GO SEND WAIT TIME [ns]** | Time from GO signal sent to execution start - latency of the dispatch path |
| **OP TO OP LATENCY [ns]** | Gap between consecutive ops, measured from the BR/NRISC perspective |
| **INPUT_x / OUTPUT_x** | Per-tensor shape (W/Z/Y/X), layout (TILE/ROW\_MAJOR), dtype, and memory location |
| **PM IDEAL / COMPUTE / BANDWIDTH [ns]** | Performance model estimates: how long this op *should* take under ideal, compute-bound, and bandwidth-bound conditions |
| **PM FPU UTIL (%)** | Estimated floating-point utilisation - how saturated the Tensix FPUs are |
| **NOC UTIL (%) / MULTICAST NOC UTIL (%)** | Network-on-chip utilisation (unicast and multicast) |
| **DRAM BW UTIL (%)** | DRAM bandwidth utilisation - high values indicate memory pressure |
| **PROGRAM CACHE HIT** | Whether the compiled program was re-used from cache (True) or re-compiled (False) |
| **COMPUTE / DATA MOVEMENT KERNEL SOURCE & HASH** | Kernel file paths and cache keys - useful for debugging or matching kernel variants |

---

## Section 1 - Where the Time Actually Goes

### Cumulative Device Time by Operation Class

The 59.05 ms of total device kernel time breaks down as follows:

| Operation | Calls | Total Kernel Time | Share | Avg Cores |
| :--- | ---: | ---: | ---: | ---: |
| MatmulDeviceOperation | 206 | 16.23 ms | **27.5%** | 30.3 |
| TilizeWithValPaddingDeviceOperation | 194 | 11.97 ms | **20.3%** | 3.0 |
| Conv2dDeviceOperation | 66 | 6.53 ms | **11.1%** | 61.3 |
| BinaryNgDeviceOperation | 278 | 6.31 ms | **10.7%** | 64.0 |
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
| Other (9 op types) | 162 | 1.27 ms | 2.2% | - |

**Matmul and TilizeWithValPadding together account for nearly half (47.8%) of all device execution time.** These are the two obvious targets for any optimisation effort.

---

## Section 2 - Hardware and Compute Configuration

### Core Utilisation

The majority of ops - 1,042 out of 1,860 - run at the full 64-core count. This is the ideal scenario: the op has enough parallelism to keep every Tensix worker busy. Notable exceptions:

- **TilizeWithValPaddingDeviceOperation** averages just 3 cores per call. This is the single most underutilised op in the pipeline. Despite consuming 20.3% of total device time, it is severely under-parallelised, likely because its input tensors are small or its tiling strategy does not scale. This is worth investigating - if the tiling or sharding strategy can be changed to fan out across more cores, there is meaningful time to be recovered here.
- **MatmulDeviceOperation** averages ~30 cores. That is roughly half the available pool. This is expected for matmuls operating on smaller feature maps in the transformer decoder head, but for the backbone's larger activations, the expectation would be 62–64 cores. The HiFi2 variant of this op (122 calls) averages 4× longer than the HiFi4 variant (84 calls), which is counterintuitive - HiFi2 is supposed to be faster. This suggests those larger, slower matmuls happen to also be in HiFi2 precision territory (likely larger shapes), and the shape complexity is dominating the precision savings.
- **LayerNormDeviceOperation** averages just 10.3 cores across 40 calls. LayerNorm is inherently more difficult to parallelise because the reduction must happen across the full channel dimension before the normalisation is applied. Even so, 10 cores out of 64 is quite sparse, and depending on the reduction strategy in use, there may be room to widen the shard grid.

### Math Fidelity

Across all 1,860 ops, **1,084 (58%) run at HiFi4** and **268 (14%) at HiFi2**. The remaining 508 ops either have no compute RISC component (data movement only) or the fidelity is not populated.

HiFi4 is the highest fidelity mode - full precision on Tensix FPUs. It is appropriate for the attention mechanism, LayerNorm, and the transformer decoder where numerical precision matters for detection box regression. HiFi2 is correctly applied in the convolutional backbone where tolerance for slight precision reduction is higher and throughput is more valuable.

### Data Types and Precision

- **BFLOAT16** dominates: 1,360 input tensors and 1,610 output tensors use it.
- **FLOAT32** appears on 500 input tensors, primarily in the initial tilize step and in paths that preserve FP32 for accumulation accuracy.
- The typecast chain from FLOAT32 → BFLOAT16 accounts for 250 ops - these are the `TypecastDeviceOperation` entries. Each one is very fast (~7.2 µs average kernel time), so the conversion overhead is not a bottleneck; the strategy of processing in FP32 then casting to BF16 before further compute is paying off.

### Memory Layout

| Location | Input Tensor Count | Output Tensor Count |
| :--- | ---: | ---: |
| DEV_0_DRAM_INTERLEAVED | 862 | 864 |
| DEV_0_L1_INTERLEAVED | 714 | 712 |
| DEV_0_L1_BLOCK_SHARDED | 196 | 196 |
| DEV_0_L1_HEIGHT_SHARDED | 88 | 88 |

About **54% of tensors live in DRAM** and **46% in L1**. The L1 tensors benefit from lower-latency access and reduced NoC traffic. The sharded variants (block and height sharded, totalling 284 tensors each way) are the most bandwidth-efficient - they distribute data across cores so each core's local L1 holds exactly what it needs, eliminating inter-core NoC fetches for those activations.

The 74 `InterleavedToShardedDeviceOperation` and 74 `ShardedToInterleavedDeviceOperation` calls reflect the layout transitions needed to move between these regimes. They collectively cost ~2.16 ms of device time - not free, but the payoff in reduced memory latency for the ops in between typically justifies it.

---

## Section 3 - Firmware Overhead Analysis (FW vs Kernel Duration)

The difference between DEVICE FW DURATION and DEVICE KERNEL DURATION reveals per-op firmware bookkeeping - things like kernel launch setup, cycle-counter initialisation, and teardown. This is unavoidable to some degree, but large ratios flag either very short kernels (where the fixed overhead dominates) or internal firmware inefficiencies.

| Operation | FW Overhead |
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

`MoveDeviceOperation` and `NLPConcatHeadsDeviceOperation` have the worst firmware overhead ratios - over 22% of their FW time is overhead, not actual computation. These are short-running ops (3.4 µs and 4.4 µs average kernel time respectively), which means the fixed firmware entry/exit cost is a large fraction. If these ops appear in critical paths, fusing them into adjacent operations is the recommended fix. At their current call count (62 and 2 respectively), their absolute impact is manageable (~213 µs and ~9 µs total kernel time), but this ratio is worth monitoring as model complexity scales.

`MatmulDeviceOperation`, `SDPAOperation`, and `Conv2dDeviceOperation` all have sub-3% overhead - these are the well-utilised, long-running kernels where the fixed cost is amortised properly. This is the correct regime.

---

## Section 4 - PyTorch Fallbacks

**There are none.**

Every single one of the 1,860 ops in this trace has OP TYPE `tt_dnn_device` and carries a valid `DEVICE FW DURATION [ns]` value. Zero operations fell back to PyTorch CPU execution. This is an excellent result - it means the full inference graph has been successfully lowered to Tenstorrent device kernels with no host-side intervention during the forward pass.

This also means there are no device-to-host or host-to-device data transfer penalties from fallback ops, and no synchronisation stalls caused by waiting for CPU math to complete before resuming device execution.

---

## Section 5 - Program Cache Analysis

**Every operation shows `PROGRAM CACHE HIT: False`.**

This is expected behaviour for a first-run profiling trace. The program cache is populated during the first inference pass; subsequent runs will hit the cache and avoid recompilation. The profiling recommendation from Tenstorrent's documentation specifically notes that only the host timings from the *second* run should be taken as representative of steady-state performance, because the first run is building the cache.

The 139.95 ms host time captured here likely includes compilation overhead. For a production deployment or benchmarking scenario, this model should be run at least twice in the same process, with the second run's timings used for comparison. The device kernel times (which are traced at the hardware level) are already representative of steady-state device execution, since they measure actual Tensix cycles regardless of the cache state.

---

## Section 6 - Op-to-Op Latency Spikes

Op-to-op latency measures the gap between consecutive operations as seen from the device - essentially, dead time between one kernel finishing and the next starting. Large spikes here indicate pipeline bubbles, usually caused by host-side scheduling delays, synchronisation waits, or the overhead of transitioning between memory layouts.

### Largest Observed Spikes

| OP CODE | Global Call Count | Op-to-Op Latency | Kernel Duration |
| :--- | ---: | ---: | ---: |
| TilizeWithValPaddingDeviceOperation | 254,976 | **4,699.5 ms** | 1.21 ms |
| TilizeWithValPaddingDeviceOperation | 254,977 | **4,699.5 ms** | 1.21 ms |
| TilizeWithValPaddingDeviceOperation | 761,856 | 857.6 ms | 0.016 ms |
| TilizeWithValPaddingDeviceOperation | 761,857 | 857.6 ms | 0.016 ms |
| MatmulDeviceOperation | 538,624 | 401.9 ms | 0.17 ms |
| MatmulDeviceOperation | 538,625 | 401.9 ms | 0.17 ms |

The first pair of spikes - nearly **4.7 seconds** of op-to-op latency - is striking. These are not execution time; they represent a window where the device sat idle between the preceding op completing and these two `TilizeWithValPadding` ops starting. Given that both entries share almost identical latency values and occur on adjacent global call counts (254,976 and 254,977), this is almost certainly a large image input being processed - possibly a batch boundary or a preprocessing step where the host is loading or resizing input data before feeding the next inference. If this trace spans multiple inference calls (which the global call count of 254,000+ strongly suggests - far higher than a single pass would generate), then the 4.7-second gap is the inter-inference idle time.

The **MatmulDeviceOperation** spikes at call counts 538,624 and 538,625 (401 ms latency each) follow a similar pattern. These are almost certainly at the start of another inference invocation after a host-side idle period.

**The practical implication:** if this model is being used for real-time detection, the host pipeline needs to be pipelining input preparation in parallel with device inference. These latency gaps represent direct throughput loss.

---

## Section 7 - Key Operations Deep Dive

### Matmul (27.5% of device time)

The 206 matmul calls split into two fidelity bands:
- **HiFi2 (122 calls):** Average kernel duration **113.7 µs**. These are the larger transformer attention and projection matmuls.
- **HiFi4 (84 calls):** Average kernel duration **28.0 µs**. These are smaller, precision-sensitive matmuls in the decoder.

The 4× speed gap between HiFi2 and HiFi4 is primarily driven by tensor shape rather than fidelity, but this does raise a question: if HiFi4 is being used for smaller, faster matmuls, could HiFi2 be substituted for any of them without regression in detection accuracy? Even a subset of those 84 HiFi4 calls moved to HiFi2 could recover a few microseconds per inference.

Average core count of 30 across all matmuls leaves room for improvement on the larger-shape variants. Checking whether `use_matmul_1d_systolic_array` or a higher degree of output tiling could unlock more parallelism on the bigger attention matrices is recommended.

### TilizeWithValPadding (20.3% of device time)

This operation - which converts ROW_MAJOR input tensors to TILE layout while padding to tile boundaries - is the second largest consumer of device time and arguably the most poorly utilised. At only **3 cores on average** across 194 calls, it is leaving 61 cores idle for a fifth of the model's device execution time.

The likely cause: the input tensors at this stage are either small in spatial extent or the multicore tiling strategy is not being triggered. Verifying that `use_multicore: true` is consistently being set in the attributes (the first-row entries in the CSV show it is enabled for `TilizeDeviceOperation`, but this should be confirmed for the `ValPadding` variant specifically) and ensuring the input shapes are large enough to benefit from parallelisation would be the first debugging step.

### Conv2d (11.1% of device time)

66 convolution calls averaging **99 µs** each, with very high core utilisation (**61.3 cores** on average). Convolutions are executing well - the memory layout transition to block-sharded format is clearly working as intended. FW overhead is also low at 1.8%. This is one of the healthier op classes in the trace.

### SDPA - Scaled Dot-Product Attention (1.2% of device time)

14 calls, each using all 64 cores, with an average kernel time of **51.8 µs** and negligible firmware overhead (1.9%). The attention mechanism is executing on-device cleanly and efficiently. Given that SDPA is often a candidate for memory-intensive operations, the low DRAM-to-L1 pressure here suggests the KV tensors may be fitting in L1 shards for the sequence lengths in use.

### LayerNorm (1.0% of device time)

40 calls at an average of **14.9 µs** kernel time, but with low average core count (10.3). Despite HiFi4 precision, the firmware overhead is 7.7% - the second-worst of the larger ops. LayerNorm's sequential reduction dependency makes full core parallelisation hard, but exploring width-sharded strategies for the reduction axis may yield improvement.

---

## Section 8 - Recommendations

### Immediate wins

**1. Investigate TilizeWithValPadding parallelisation.** This single op class is using only 3 cores on average while consuming 20% of device time. If the tiling strategy can be widened to 16–32 cores, the potential saving is on the order of 6–10 ms per inference pass.

**2. Profile a second-run, cache-warm trace.** All operations show `PROGRAM CACHE HIT: False`. The current host time (139.95 ms) includes compilation. A cache-warm run will show the true steady-state host latency and may reveal a significantly smaller gap between host and device time.

**3. Pipeline host preprocessing.** The 4.7-second op-to-op gaps at the TilizeWithValPadding boundaries suggest the host is stalling on input preparation between inference calls. A double-buffered preprocessing pipeline would eliminate this dead time.

### Medium-term

**4. Evaluate fidelity downgrade for selected matmuls.** The 84 HiFi4 matmul calls average 28 µs each. If accuracy analysis shows they can tolerate HiFi2, switching would bring them closer to the HiFi2 average and reduce their contribution to the 16.2 ms matmul total.

**5. Reduce memory layout transitions.** The 74 Interleaved-to-Sharded and 74 Sharded-to-Interleaved calls cost ~2.16 ms combined. Restructuring adjacent ops to share the same memory layout where possible eliminates the need for some of these transitions.

**6. Investigate MoveDeviceOperation fusion.** 62 calls with 22% firmware overhead and average kernel time of 3.4 µs. These are short enough to warrant folding into adjacent copy or reshape operations if the op graph allows it.

---

*Report generated from trace: `ops_perf_results_rtdetr_2026_05_25_05_51_41.csv` - 1,860 operations on Wormhole B0*