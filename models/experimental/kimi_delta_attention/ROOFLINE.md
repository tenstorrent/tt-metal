# KDA roofline

## Scope and ceilings

The measured prefill point is `B=1,T=640,H=32,K=V=128,hidden=2304`,
chunk size 32, BF16 activations, FP32 recurrence state, and HiFi4 on one
Blackhole at 1.35 GHz. Results are warm serialized device-kernel time.

The repository matmul model defines 4096 FLOP/core/cycle at LoFi and divides
throughput by the fidelity multiplier. HiFi4 therefore gives
`110 * 4096 / 4 * 1.35 GHz = 152.064 TFLOP/s` for the full chip. The repository
Blackhole DRAM ceiling is 512 GB/s. Sources:

- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2646-2662`
- `ttnn/api/ttnn/operation.hpp:130-137`
- `ttnn/core/operation.cpp:36-42`

The resulting ridge point is `152064 / 512 = 297 FLOP/byte`.

## Full-layer result

Profile:
`/tmp/kda_layer_flat_decay_profile/reports/2026_07_23_09_29_14/ops_perf_results_2026_07_23_09_29_14.csv`.
Three warm iterations averaged 4.641325 ms.

| Work | Executed/algorithmic GFLOP | Mean time | Utilization |
|---|---:|---:|---:|
| Five projections | 50.510 | 693.744 us | 47.86% of chip compute peak |
| KDA prep matrix work | 1.007 | 321.193 us | 2.06% of chip compute peak |
| KDA scan matrix work | 2.349 | 181.788 us | 29.21% of its 32-core peak; 8.50% of chip peak |
| Four-tap depthwise FIR | 0.055 | included below | n/a |
| Whole layer | 53.920 | 4641.325 us | 11.617 TFLOP/s, 7.64% of chip compute peak |

Projection FLOPs use `2*M*K*N`. The projection-only profiler roofline
(`sum(PM IDEAL) / sum(measured)`) is 49.36%. Its per-call results are:

| Projection | Mean time | PM ideal | PM utilization |
|---|---:|---:|---:|
| QKV, `2304 -> 12288` | 395.041 us | 238.314 us | 60.33% |
| Auxiliary, `2304 -> 288` | 57.296 us | 5.586 us | 9.75% |
| Decay, `128 -> 4096` | 27.281 us | 9.536 us | 34.96% roofline, 16.18% FPU |
| Output gate, `128 -> 4096` | 27.281 us | 9.536 us | 34.96% roofline, 16.18% FPU |
| Output, `4096 -> 2304` | 186.845 us | 79.438 us | 42.52% |

The QKV projection meets the 60% aspiration. The layer does not: layout and
small pointwise programs consume most elapsed time, and the small auxiliary
projections do not fill the machine.

## Custom recurrence accounting

For each `(head,chunk)` prep item, the kernel executes 24 full 32x32 tile
matmuls: four each for the two gate accumulations and two pairwise products,
plus eight for the doubling inverse. Across 640 items this is 1.006633 GFLOP.
The explicit DRAM path reads q/k/v, vector gate, beta, and six per-core
constants, then writes 22 FP32 intermediate tiles per item. It moves 89.211 MB,
or 11.28 FLOP/byte. At 321.193 us it sustains 277.75 GB/s, 54.25% of the DRAM
ceiling. The bandwidth lower bound is 174.24 us; the 6.62 us matrix-compute
lower bound is not limiting. Small-matrix setup, SFPU normalization/exp, and
packing explain the remaining gap.

For each `(head,chunk)` scan item, the kernel executes 56 tile matmuls. Across
640 items this is 2.348810 GFLOP. Initial/final state, seven prep tensors, and
the FP32 output move 72.352 MB, or 32.46 FLOP/byte. At 181.788 us scan sustains
398.00 GB/s, 77.73% of the DRAM ceiling. Its bandwidth lower bound is
141.312 us. Scan is DRAM-bound.

The counts follow:

- `device/kernels/compute/chunk_kda_prep.cpp:374-508`
- `device/kernels/compute/chunk_kda_scan.cpp:125-181`
- `device/kernels/dataflow/reader_chunk_kda_prep.cpp:84-178`
- `device/kernels/dataflow/writer_chunk_kda_prep.cpp:71-81`
- `device/kernels/dataflow/reader_chunk_kda_scan.cpp:90-102`
- `device/kernels/dataflow/writer_chunk_gdn_scan.cpp:45-72`

## Eight-device CCL roofline

The proposed TP=8 replicated-output path all-reduces a BF16
`[1,1,640,2304]` partial: 2.949120 MB/device. Mirroring the sparse-MLA CCL
model, an eight-device reduce-scatter plus all-gather has
`2 * payload * (P-1)/P = 5.160960 MB` on the critical path. The live LoudBox
ceiling is `400 Gbit/s/link/direction * 2 links = 100 GB/s`, hence a
51.610 us theoretical time.

Command:
`scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_kda_ccl_perf.py -q -s`

Result: PASS, 1/1. The real-time profiler measured 219.169 us, 23.5% fabric
roofline utilization. This does not meet the 40% aspiration. The next
optimization must avoid treating a standalone DRAM-to-DRAM all-reduce as the
final path: fuse output matmul with reduce-scatter and overlap the collective
with local work, then remeasure the same critical-path byte model.

The topology/critical-path convention mirrors
`models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_ccl_perf.py`.
