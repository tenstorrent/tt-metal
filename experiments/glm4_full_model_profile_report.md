# GLM-4.7-Flash Full Model Profile Report

**Date:** 2026-03-11
**Model:** zai-org/GLM-4.7-Flash (47 layers, MLA attention, MoE)
**Hardware:** 4x Wormhole (mesh_shape=1,4, FABRIC_1D)
**Profiler:** Tracy (TT_METAL_DEVICE_PROFILER=1)
**Test:** `debug_run_full_tt_greedy.py --max-new-tokens 4 --mesh-cols 4`

---

## Executive Summary


| Metric                           | Value                                            |
| -------------------------------- | ------------------------------------------------ |
| Decode throughput                | **1.98 tokens/second**                           |
| Decode latency                   | **504.6 ms/token**                               |
| Prefill time                     | 1017.3 s (iterative single-token, not optimized) |
| Prompt length                    | 8 tokens                                         |
| New tokens generated             | 4                                                |
| Device kernel time (per device)  | 44.2 ms avg                                      |
| Device kernel utilization        | 8.8% of decode latency                           |
| Total device ops (per device)    | 1,870                                            |
| Total device ops (all 4 devices) | 7,496                                            |
| Ops per decode step (est.)       | ~156 ops/device/token                            |


---

## Per-Device Summary


| Device      | Ops       | Total Kernel (us) | Total Kernel (ms) |
| ----------- | --------- | ----------------- | ----------------- |
| 0           | 1,870     | 44,225            | 44.2              |
| 3           | 1,870     | 44,421            | 44.4              |
| 4           | 1,878     | 47,571            | 47.6              |
| 7           | 1,878     | 44,651            | 44.7              |
| **Average** | **1,874** | **45,217**        | **45.2**          |


Device 4 is ~7% slower than the other devices (47.6 vs 44.2-44.7 ms), likely due to being a remote ethernet device.

---

## Per-Op Breakdown (Device 0, representative)


| #         | Op Name                              | Count     | Kernel (us) | %       | Avg (us) | Max (us) |
| --------- | ------------------------------------ | --------- | ----------- | ------- | -------- | -------- |
| 1         | MatmulDeviceOperation                | 206       | 12,754      | 28.8    | 62       | 221      |
| 2         | FillPadDeviceOperation               | 75        | 4,662       | 10.5    | 62       | 223      |
| 3         | TilizeDeviceOperation                | 21        | 4,376       | 9.9     | 208      | 210      |
| 4         | BinaryNgDeviceOperation              | 242       | 2,702       | 6.1     | 11       | 43       |
| 5         | SparseMatmulDeviceOperation          | 62        | 2,521       | 5.7     | 41       | 98       |
| 6         | PermuteDeviceOperation               | 24        | 2,174       | 4.9     | 91       | 99       |
| 7         | RepeatDeviceOperation                | 48        | 1,975       | 4.5     | 41       | 93       |
| 8         | CloneOperation                       | 319       | 1,489       | 3.4     | 5        | 25       |
| 9         | TransposeDeviceOperation             | 175       | 1,441       | 3.3     | 8        | 26       |
| 10        | UnaryDeviceOperation                 | 86        | 1,382       | 3.1     | 16       | 25       |
| 11        | LayerNormDeviceOperation             | 44        | 1,342       | 3.0     | 30       | 44       |
| 12        | MoeExpertTokenRemapDeviceOperation   | 24        | 1,303       | 2.9     | 54       | 105      |
| 13        | PadDeviceOperation                   | 25        | 1,092       | 2.5     | 44       | 45       |
| 14        | ReduceScatterDeviceOperation         | 17        | 795         | 1.8     | 47       | 120      |
| 15        | RotaryEmbeddingLlamaDeviceOperation  | 50        | 609         | 1.4     | 12       | 33       |
| 16        | ReshapeViewDeviceOperation           | 45        | 523         | 1.2     | 12       | 17       |
| 17        | AllGatherDeviceOperation             | 17        | 480         | 1.1     | 28       | 35       |
| 18        | GatherDeviceOperation                | 9         | 466         | 1.1     | 52       | 52       |
| 19        | TopKDeviceOperation                  | 9         | 392         | 0.9     | 44       | 44       |
| 20        | SliceDeviceOperation                 | 183       | 360         | 0.8     | 2        | 5        |
| 21        | FastReduceNCDeviceOperation          | 24        | 283         | 0.6     | 12       | 14       |
| 22        | ConcatDeviceOperation                | 45        | 271         | 0.6     | 6        | 11       |
| 23        | SdpaDecodeDeviceOperation            | 25        | 241         | 0.5     | 10       | 20       |
| 24        | UntilizeWithUnpaddingDeviceOperation | 18        | 234         | 0.5     | 13       | 14       |
| 25        | TilizeWithValPaddingDeviceOperation  | 20        | 124         | 0.3     | 6        | 8        |
| 26        | PagedUpdateCacheDeviceOperation      | 11        | 94          | 0.2     | 9        | 9        |
| 27        | ScatterDeviceOperation               | 21        | 85          | 0.2     | 4        | 4        |
| 28        | InterleavedToShardedDeviceOperation  | 11        | 18          | 0.0     | 2        | 2        |
| 29        | ReduceDeviceOperation                | 9         | 16          | 0.0     | 2        | 2        |
| 30        | UntilizeDeviceOperation              | 2         | 9           | 0.0     | 4        | 4        |
| 31        | EmbeddingsDeviceOperation            | 3         | 9           | 0.0     | 3        | 3        |
| **TOTAL** |                                      | **1,870** | **44,225**  | **100** |          |          |


---

## Op Category Analysis


| Category              | Ops | Kernel (us) | %    | Description                                  |
| --------------------- | --- | ----------- | ---- | -------------------------------------------- |
| **Compute (Matmul)**  | 268 | 15,275      | 34.5 | Dense + sparse matmuls                       |
| **Data Movement**     | 530 | 10,850      | 24.5 | Pad, Fill, Permute, Repeat, Clone, Transpose |
| **Layout Conversion** | 62  | 4,743       | 10.7 | Tilize, Untilize, Reshape                    |
| **Element-wise**      | 328 | 4,084       | 9.2  | Binary, Unary ops                            |
| **MoE Routing**       | 42  | 1,695       | 3.8  | TopK, ExpertRemap, Gather                    |
| **Normalization**     | 44  | 1,342       | 3.0  | LayerNorm/RMSNorm                            |
| **Multi-device Comm** | 34  | 1,275       | 2.9  | ReduceScatter, AllGather                     |
| **Attention**         | 75  | 850         | 1.9  | SDPA, RoPE, KV cache                         |
| **Other**             | 487 | 2,111       | 4.8  | Slice, Concat, Reduce, Scatter, Embed        |


---

## Bottleneck Analysis

### 1. Host-side dispatch overhead dominates (91.2%)

Device kernel time is only 44.2 ms per forward pass, but decode latency is 504.6 ms. This means **91.2% of time is host-side overhead** (dispatch, data copies, Python framework, synchronization). This is the single biggest optimization opportunity.

### 2. Data movement is 24.5% of kernel time

FillPad (10.5%), Permute (4.9%), Repeat (4.5%), Clone (3.4%), Transpose (3.3%) together consume 10,850 us. Many of these could potentially be eliminated through:

- Better tensor layout choices to avoid permute/transpose
- Fusing padding into compute ops
- In-place operations to avoid clone

### 3. Layout conversion is 10.7% of kernel time

TilizeDeviceOperation alone is 9.9% (4,376 us). This suggests input tensors are frequently in the wrong layout. Pre-tilizing weights and keeping activations in tile layout would help.

### 4. Matmul utilization can be improved

206 dense matmuls average only 62 us each -- this includes both small attention projections and large MLP matmuls. The largest matmuls (221 us, MLP gate/up) use 54 cores effectively. Smaller matmuls using fewer cores could benefit from DRAM-sharded strategies.

### 5. MoE overhead is significant

MoE-specific ops (ExpertRemap, TopK, Gather, SparseMatmul, FastReduceNC) total ~5,655 us (12.8%). The SparseMatmulDeviceOperation at 2,521 us (5.7%) is the MoE expert computation. Optimization paths:

- Expert parallelism across devices (already using 4-device sharding)
- Fused MoE kernels to reduce dispatch overhead

### 6. Multi-device communication is modest

ReduceScatter + AllGather = 1,275 us (2.9%). With FABRIC_1D topology, inter-device communication is not a major bottleneck.

---

## Recommendations for Optimization


| Priority | Target                 | Est. Impact     | Approach                                       |
| -------- | ---------------------- | --------------- | ---------------------------------------------- |
| **P0**   | Host dispatch overhead | ~10x throughput | Enable trace mode (metal trace capture/replay) |
| **P1**   | Data movement (24.5%)  | ~10 ms savings  | Fuse padding, eliminate permute/clone          |
| **P2**   | Tilize overhead (9.9%) | ~4 ms savings   | Pre-tilize inputs, maintain tile layout        |
| **P3**   | MoE kernel fusion      | ~3 ms savings   | Fused sparse matmul + remap                    |
| **P4**   | Matmul optimization    | ~2 ms savings   | DRAM-sharded for small projections             |


---

## Output Files


| File                                                    | Description                                                                          |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `experiments/glm4_full_model_ops_profile.csv`           | Raw per-op data: 7,496 rows (all devices), with op name, kernel duration, core count |
| `experiments/glm4_full_model_ops_summary.csv`           | Per-op summary by device: count, total/avg/min/max kernel time, percentage           |
| `generated/profiler/.logs/tracy_profile_log_host.tracy` | Raw Tracy capture (can open in Tracy Profiler GUI)                                   |
| `generated/profiler/.logs/cpp_device_perf_report.csv`   | Raw device-side perf report                                                          |
| `generated/profiler/.logs/profile_log_device.csv`       | Raw device profiler log (468 MB)                                                     |


---

## Test Environment

- **Machine:** 4x Wormhole (PCIe IDs: 0,1,2,3; Remote: 4,5,6,7)
- **TTNN:** Built from source (tt-metal)
- **PyTorch:** 2.7.1+cpu
- **Transformers:** 4.53.0
- **KV Cache dtype:** bfloat16
- **Block size:** 64
- **Fabric:** FABRIC_1D (1D ring topology)

---

## Notes

- Prefill was done via iterative single-token decode (1017s for 8 tokens). This is a bring-up path, not production-optimized prefill.
- Tracy report generation (`process_ops_logs.py`) crashed with multi-device assertion error ("Device data missing for device 4"). This report was generated by manually parsing the raw profiler logs.
- Profiler DRAM buffers overflowed on device 4, which may cause slight undercount of some ops on that device.
- The 1.98 tok/s throughput is without trace mode -- enabling metal trace capture/replay should dramatically improve throughput by eliminating host dispatch overhead.
