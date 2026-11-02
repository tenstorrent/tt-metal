# Fused corrected log_softmax prototype

For branch state, transfer instructions, build caveats and next steps, see the
[handoff document](LOG_SOFTMAX_HANDOFF.md).

This branch keeps the corrected composite as `ops::log_softmax_composite` and routes
supported `ops::log_softmax` calls through one device program for forward and one
for backward. The composite branch and draft PR #63429 remain unchanged.

## Arithmetic and storage

Following [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642),
Section 2, select exactly one first maximum, omit its exponential from the tail,
and compute `log1p(tail)` in FP32. Forward emits `x - max(x) - log1p(tail)`.
Backward computes non-selected gradients in FP32 and reconstructs the selected
one as minus their sum. FP32 input and row-state loads use unpack-to-destination
mode; routing these through FPU copy would truncate them to TF32.

A core streams 32 rows through bounded L1 buffers. Forward reads logits four
times: maximum, selected index, exponential tail, output. Backward makes three
passes to sum upstream, sum non-selected gradients, and emit gradients. It
recomputes probabilities from input and saved statistics. All tensor arithmetic
runs on device, with no host fallback. Device permutations move non-last axes to
the last axis and restore the output and input gradient.

The retained context contains three FP32 vector tiles per 32 physical rows:
maximum, selected index, and log normalizer. This costs 384 bytes per physical
row, or **6 MiB for NanoLlama3**, instead of full-size FP32 log probabilities and
maximum-selection masks. Output and input gradient retain the input dtype;
FP32 upstream gradients are supported with BF16 inputs.

This implements the probability-collapse correction. Section 3's separate
teacher/student cancellation correction requires a logits-aware loss; it cannot
be supplied by a log_softmax API returning rounded log probabilities. Existing
fused training loss kernels are independent and are not changed here.

## NanoLlama3 measurement

Blackhole P150a, AMD Ryzen 9 9950X3D, Release runtime, 2026-10-30. Shape
`[64, 1, 256, 32000]`, reducing the vocabulary axis, from
`training_shakespeare_nanollama3.yaml` and `nanollama3.yaml`. This measures the
standalone BF16 operation and its backward, not a model training step.

| Implementation | Forward ms | Backward ms | Forward + backward ms | Forward peak GiB | Step peak GiB | Forward retained GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 19.52 | 19.36 | 38.81 | 2.933 | 3.907 | 0.977 |
| composite | 291.10 | 86.62 | 377.72 | 9.775 | 14.650 | 4.883 |
| fused | 21.24 | 34.67 | 55.79 | 0.982 | 1.959 | 0.982 |

Peaks are incremental live DRAM tensor allocations traced by TTNN graph capture.
Input and upstream are preallocated and excluded: together they add 1.953 GiB.
Including those buffers gives approximately 5.86 / 16.60 / 3.91 GiB of tensor
storage at step peak for original / composite / fused. Runtime reservations,
allocator fragmentation, and other model tensors are outside this measurement.
Forward retained includes the native output and tensors saved for backward.
Phases are timed independently; their medians need not add up exactly.

The fused step is about **6.8x faster than the corrected composite**, while taking
about **44% longer than the original implementation**, which has the arithmetic
bug. Fused forward is close to original. Backward performs more passes and
recomputes probabilities; no device profile was collected to attribute the
remaining latency precisely.

## Other shapes and remaining trade-offs

Forward + backward median latency, same device and BF16 inputs:

| Shape | Axis | Original ms | Composite ms | Fused ms |
| --- | ---: | ---: | ---: | ---: |
| [1, 1, 32, 128] | 3 | 0.091 | 0.259 | 0.097 |
| [1, 1, 128, 1024] | 3 | 0.105 | 0.570 | 0.467 |
| [1, 1, 1024, 4096] | 3 | 0.470 | 6.375 | 1.739 |
| [1, 1, 32, 32000] | 3 | 1.074 | 1.847 | 13.244 |
| [1, 1, 256, 32000] | 3 | 1.518 | 6.586 | 13.256 |
| [1, 1, 1024, 4096] | 2 | 0.447 | 80.293 | 0.716 |

This is not a uniform performance improvement. A 32-row group occupies one
core, so 32 or 256 rows with a 32000-wide vocabulary use only one or eight cores.
They are slower than the corrected composite. Splitting wide rows across cores
is a future optimization. Narrow shapes can also favor existing operations.
Non-last axes add permutation allocations and launches; their cost is included
above. Recomputing statistics/probabilities trades device work and DRAM reads
for lower live DRAM storage. The prototype currently uses no cross-core row
reduction and does not materialize full-size FP32 intermediates.

The fused path supports rank-four BF16/FP32, standard 32x32 TILE, interleaved
DRAM inputs, and finite logits with reduction width at most 2^24. Other input
layouts, memory configurations, tile sizes, dtypes, or ranks keep the corrected
composite path. Numerical tests covered all four axes, negative-axis notation,
tile padding, tied maxima, singleton axes, confidence gaps 8/18/24, large class
indices, and repeated gradient accumulation. Two additional tests cover FP32
upstream with BF16 input and 32003-wide rows with saved-context/cache reuse.
**All 45 selected tests passed** without relaxing existing tolerances. Wormhole,
multi-device execution, full-model training, and a clean whole-repository build
remain unverified.

## Method and reproduction

The C++ Google Benchmark executable uses `steady_clock` and explicit device
synchronization after warmup and each sample. This includes eager host dispatch,
not just kernel execution. It is not a Tracy device profile or a Python timer.
NanoLlama3 uses one warmup and one invocation per sample, with three samples;
other shapes use five warmups, twenty invocations per sample and seven samples.
Reported values are medians, with min/max in the accompanying JSON. The
original, corrected composite, and fused implementations rotate execution order
and share the same runtime, input buffers, and seeded data. Allocation tracing
runs separately from timing.

Changed host objects, tests, and benchmark were rebuilt and relinked against the
available Release TTNN/Metal libraries; new kernels were JIT compiled. Host
sources were also syntax checked against this worktree's headers. This was not
a clean build of all dependencies. Runtime checkout and build details are
recorded in the [measurement data](../tests/benchmark/results/log_softmax_fused_blackhole_2026_10_30.json).

After configuring tt-train in a build for this worktree:

```bash
cmake --build build --target ttml_tests log_softmax_benchmark
export TT_METAL_HOME="$PWD"
TT_METAL_OPERATION_TIMEOUT_SECONDS=10 flock /tmp/tt-device.lock timeout 300 \
  build/tt-train/tests/ttml_tests \
  --gtest_filter='UnaryOpsTest.*:PrecisionAndAxes/LogSoftmaxTest.*:TrivialTnnFixedTest.TestMax*:TrivialTnnFixedTest.TestStableSoftmax*'
TT_METAL_OPERATION_TIMEOUT_SECONDS=10 flock /tmp/tt-device.lock timeout 900 \
  build/tt-train/tests/log_softmax_benchmark \
  --benchmark_filter='^LogSoftmax(NanoLlama3)?/'
```
