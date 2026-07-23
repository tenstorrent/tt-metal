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

## Measured TP=8 fused layer

Profile:
`/tmp/kda_tp_layer_t640_grid8x8_r10/reports/2026_07_23_10_41_10/ops_perf_results_2026_07_23_10_41_10.csv`.
The full target-shape TP=8 layer passed for ten warm measured iterations.
Excluding the first sample, the median device critical path was 5.749 ms; the
signposted host interval averaged 6.116 ms. Device kernels sum to only
1.20-1.23 ms/device, so host dispatch gaps and unfused
layout/pointwise operations, rather than recurrence or CCL alone, dominate the
layer wall time.

TP executes 59.205 GFLOP across the mesh. This includes the 53.920 GFLOP
single-device algorithm plus 5.285 GFLOP from replicating the two 128-wide
low-rank auxiliary outputs on every device. At 5.749 ms, the mesh sustains
10.30 TFLOP/s, or 0.85% of the eight-chip HiFi4 ceiling. This whole-layer
number includes idle gaps and is intentionally not renormalized to active
cores.

| Program | Slowest-chip warm time | Observation |
|---|---:|---|
| KDA prep, 80 cores/device | 84.295 us | 3.8x faster than single-chip H=32 prep |
| KDA scan, 16 cores/device | 96.497 us | Matches the predicted H=4 split mapping |
| Fused output matmul + reduce-scatter | 166.069 us | Slowest device median; device medians span 140.349-166.069 us |

The fused program reduces an FP32 `[1,1,640,2304]` partial. Its 5.898240 MB
payload has `payload * (P-1)/P = 5.160960 MB` on the reduce-scatter critical
path. At two links, the 100 GB/s lower bound is again 51.610 us. The measured
slowest-device median gives 31.1% effective fabric-roofline utilization. This
is a conservative combined-program metric because its 166.069 us also includes
the local output matmul; it nevertheless misses the 40% goal, which requires
at most 129.0 us. Controlled grid/worker-placement sweeps did not improve the
8x8 matmul plus row-8 reduce-scatter baseline.

## Traced TP=8 execution

Profile:
`/tmp/kda_tp_layer_t640_trace_r10/reports/2026_07_23_10_49_42/ops_perf_results_2026_07_23_10_49_42.csv`.
One warm trace replay preceded ten measured target-shape replays. Their median
slowest-device critical path was 1.263 ms, while the signposted host interval
was 12.999 ms, or 1.300 ms/layer. This is 4.55x faster than the 5.749 ms eager
device span. Each device spends a median 1.213-1.216 ms in kernels, leaving
only about 47 us between programs. The experiment therefore validates host
dispatch as the cause of the eager gap; trace replay is now the performance
baseline.

At the device critical path, the mesh sustains 46.89 TFLOP/s from the same
59.205 GFLOP executed work, or 3.85% of the eight-chip HiFi4 ceiling. At the
host-observed 1.300 ms it sustains 45.55 TFLOP/s, or 3.74%. Both figures
remain far below the 60% aspiration because the layer is a serial chain of 51
small programs/device, most of which are layout or pointwise operations, and
because whole-chip peak is not renormalized to each program's active cores.

The measured slowest-device program medians are 84.602 us for 80-core prep,
96.252 us for 16-core scan, and 148.023 us for fused output matmul +
reduce-scatter. The traced fused collective reaches `51.610 / 148.023 =
34.9%` effective fabric-roofline utilization. It still misses the 40% goal and
129.0 us target, but the improvement from the eager 166.069 us shows that
launch jitter was part of the earlier combined-program measurement.

Aggregated over the measured ten replays and eight devices, the largest active
kernel groups were untilize-with-unpadding (12.941 ms), tilize (12.468 ms),
local matmul (11.843 ms), fused matmul + reduce-scatter (11.752 ms), scan
(7.693 ms), reshape/view (7.219 ms), and prep (6.749 ms). The next whole-layer
optimization should therefore remove layout round trips and fuse adjacent
pointwise work; adding recurrence cores or changing the retained collective
grid cannot close the measured gap.

## Native depthwise-convolution result

Profile: `/tmp/kda_tp_layer_t640_native_conv_r10/reports/2026_07_23_11_02_40/ops_perf_results_2026_07_23_11_02_40.csv`.
Replacing the shifted-FIR wrapper with the repository trace-safe native
`ttnn.conv1d` pattern reduced the ten-replay median critical path from
1.263 ms to 0.987 ms (21.8%). The signposted host interval was 10.232 ms, or
1.023 ms/layer (21.3% below 1.300 ms). Median active kernels fell from
1.213-1.216 ms to 0.940-0.942 ms/device, proving removed layout work accounts
for the gain.

The native convolution measures 26.081 us median and removes six
programs/device plus about 274 us of active time. At 0.987 ms the mesh sustains
59.99 TFLOP/s, or 4.93% of eight-chip peak; host-observed throughput is
57.86 TFLOP/s, or 4.76%.

The new active-time order per device/layer is local matmuls (147.781 us),
fused output matmul + reduce-scatter (146.585 us), scan (96.771 us),
reshape/view (89.783 us), prep (84.274 us), untilize-with-unpadding
(81.276 us), tilize-with-padding (56.508 us), and untilize (39.035 us).
The fused output slowest-device median is 147.859 us, or 34.9% effective CCL
roofline; convolution does not change the collective distribution decision.

## Head-major output-gate result

Profile: `/tmp/kda_tp_layer_t640_batched_gate_r10/reports/2026_07_23_11_08_44/ops_perf_results_2026_07_23_11_08_44.csv`.
A batched `[H,128,128]` projection now emits recurrence native `[H,T,128]`
layout. It removes the prior 84.7 us reshape and about 22 us transpose while
adding about 13 us matmul time. Median device latency falls 0.987 -> 0.890 ms
(9.8%); host latency is 0.924 ms/layer and active kernels are
0.844-0.845 ms/device.

The mesh sustains 66.50 TFLOP/s or 5.47% of eight-chip peak by device span;
host-observed throughput is 64.06 TFLOP/s or 5.27%. Prep, scan, and fused
output slowest-device medians remain 84.659, 96.535, and 148.238 us. Effective
CCL utilization remains 34.8%.

## Fused input-projection result

Profile: `/tmp/kda_tp_layer_t640_fused_input_r10/reports/2026_07_23_11_13_49/ops_perf_results_2026_07_23_11_13_49.csv`.
Grouping each device Q/K/V, replicated low-rank, and local beta columns into one
projection reduced matmul active time by 51.3 us while adding 12.6 us slices.
Median device latency falls 0.890 -> 0.874 ms (1.8%), host latency is
0.913 ms/layer, and active kernels are 0.806-0.807 ms/device.

Mesh throughput is 67.71 TFLOP/s or 5.57% of eight-chip peak by device span;
host-observed throughput is 64.86 TFLOP/s or 5.33%.

The KDA-specific fused-output 1x3 subblock sweep was negative. Against the
matched 1x1 profile above, median device span changed 0.87433 -> 0.87484 ms and
slowest-chip fused time changed 146.778 -> 147.706 us. Retain 1x1.

## Fused decay-bias result

Profile:
`/tmp/kda_tp_layer_t640_fused_decay_r10/reports/2026_07_23_11_26_27/ops_perf_results_2026_07_23_11_26_27.csv`.
Passing the pre-expanded decay bias to the aligned chunk projection reduced the
program count from 44 to 43 per device. Median device latency falls
0.87433 -> 0.85469 ms (2.25%), host latency falls 0.91280 -> 0.89225 ms/layer,
and active kernels are 0.799-0.802 ms/device.

The profiler rejects the hypothesis that both following pointwise programs
were eliminated: binary programs fall from 24 to 16 per layer across the mesh,
while unary programs remain at 32. Thus the projection absorbs the bias add,
but softplus remains a device program. Mesh throughput is 69.27 TFLOP/s or
5.69% of eight-chip peak by device span; host-observed throughput is
66.36 TFLOP/s or 5.45%.
