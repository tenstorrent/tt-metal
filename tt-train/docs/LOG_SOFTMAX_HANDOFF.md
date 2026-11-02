# log_softmax handoff

Prepared 2026-11-02. Implementation and hardware measurements are from
2026-10-30. Start here when continuing on another machine; the detailed
[fused-operation report](LOG_SOFTMAX_FUSED.md) and
[measurement JSON](../tests/benchmark/results/log_softmax_fused_blackhole_2026_10_30.json)
are also committed on this branch.

## Status and branch ownership

The corrected composite is the numerical reference. A separate fused prototype
implements the same collapse correction with much less saved state. Both have
been committed; no implementation changes are waiting in their worktrees.

| Item | Branch or commit | Location/status on the source machine |
| --- | --- | --- |
| Common upstream base | `62443b624e29d4fe032d37aa3f6a055ae0e26d2d` | Parent of the first composite fix |
| Corrected composite | `nmaurice/tt-train-log-softmax-collapse` at `348687a5df3966df958a7582198c26d200ee0192` | `/workspace/tt-metal-log-softmax-collapse` |
| Fused implementation checkpoint | `nmaurice/tt-train-fused-log-softmax` at `c0acf61b5ff` | `/workspace/tt-metal-fused-log-softmax`; this handoff is a subsequent documentation commit |
| Composite draft PR | [#63429](https://github.com/tenstorrent/tt-metal/pull/63429) | Still draft; head verified as `348687a5df3` on 2026-11-02 |

The fused branch contains all five composite commits plus the fused checkpoint.
It was not published to the upstream remote as of this handoff. The composite
branch exists there. Use the bundle procedure below to transfer both branches
and this document without publishing anything.

User constraints to preserve:

- Keep the composite implementation and its draft PR separate from the fused
  experiment. Do not replace or force-update that PR with the fused code.
- Fixing standalone log_softmax remains useful even without current trainer
  callers: it provides a reference for subsequent fused loss fixes.
- Retain correctness for the paper's failure cases while optimizing performance
  and memory; do not relax the existing numerical tolerances.

The main source workspace, `/workspace/tt-metal`, contains six uncommitted
tt-train files that exactly matched the committed composite at the last check.
They are duplicate working copies, not additional work to port. It also has
unrelated scratch files and a dirty Tracy submodule. Neither is part of this
handoff. The two feature worktrees are the authoritative sources.

## Numerical problem and scope

Reference: [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642).
Keep the two reported problems distinct:

1. **Softmax collapse, Section 2:** adding a small exponential tail to the
   dominant `exp(0) = 1` can round the denominator to one. The dominant
   log-probability and its gradient then become zero prematurely. This is the
   problem corrected by both branches.
2. **Teacher/student softmax cancellation, Section 3:** computing a loss from
   separately rounded teacher probabilities/log-probabilities and student
   log-probabilities can lose their small differences. A logits-aware loss
   accepting both inputs and applying the paper's `expm1`/`log1p` formulation is
   still required. Neither standalone implementation completes that work.

For each row, choose exactly one first maximum `k`, including when maxima tie:

```text
m = x[k]
t = sum(exp(x[j] - m), j != k)
l = log1p(t)
y[j] = (x[j] - m) - l

s = sum(dy)
dx[j] = dy[j] - exp(y[j]) * s, j != k
dx[k] = -sum(dx[j], j != k)
```

Normalization and intermediate gradients use FP32. Output and input gradient
retain the native input dtype. Final dtype rounding and exponential underflow
still apply; this does not promise arbitrary precision.

Other TTNN log_softmax/Moreh and tt-train fused or vocabulary-parallel CE paths
are independent and have not been fixed by this prototype. Earlier discussion
proposed one master issue and three children: TTNN standalone/Moreh, tt-train
standalone, and tt-train fused operations. Each should describe both paper bugs
and the expectation to address both. No issue IDs are recorded here; verify
existing tracking before creating duplicates. PR #63429 explicitly documents
the cancellation work as outstanding.

## Code map and implementation details

Paths below are relative to `tt-train/`.

| File/directory | Role |
| --- | --- |
| `sources/ttml/ops/unary_ops.cpp` | `log_softmax` selects the fused path; `log_softmax_composite` retains the corrected reference and fallback; autograd integration |
| `sources/ttml/ttnn_fixed/trivial_ttnn_ops.{cpp,hpp}` | Composite FP32 normalization, selected-maximum mask, and saved context |
| `sources/ttml/metal/ops/log_softmax/log_softmax.{cpp,hpp}` | Device operation validation, tensor specs, program factory, cache key and address overrides, forward/backward API |
| `sources/ttml/metal/ops/log_softmax/kernels/compute/log_softmax.cpp` | Four-pass forward and three-pass backward compute |
| `sources/ttml/metal/ops/log_softmax/kernels/compute/log_softmax_sfpu.h` | Padding masks, selected-index handling, pivot-gradient replacement |
| `sources/ttml/metal/ops/log_softmax/kernels/dataflow/` | Bounded, double-buffered DRAM streams and output/state writes |
| `sources/ttml/metal/ops/log_softmax/sources.cmake` | Host source registration, included by the ttml CMakeLists |
| `tests/ops/unary_ops_test.cpp` | Numerical regressions and the two fused-specific tests |
| `tests/benchmark/log_softmax_benchmark.cpp` | Original/composite/fused comparisons and allocation tracking |

Supported fused inputs are rank-four BF16/FP32, standard 32x32 TILE layout and
interleaved DRAM. The reduction width must be positive and at most `2^24`, so
FP32 column indices remain exact. Tests use finite logits. Unsupported input
formats/ranks take the corrected composite path, subject to that path's own
input requirements. Non-last axes use device permutations before/after the
last-axis kernel, including the inverse permutation in backward.

`metal::log_softmax_fw(input)` returns `{output, state}`.
`metal::log_softmax_bw(input, upstream, state)` returns the native-dtype gradient.
BF16 input with FP32 upstream is supported without first rounding upstream to
BF16. The cache key includes the upstream dtype; cached address overrides update
input, upstream, state and output buffers. Validation also runs on cache hits.

One core handles a group of 32 physical rows. Forward streams logits four times:
max, first-max index, excluded exponential tail, then output. Backward sums
upstream, sums non-selected gradients, then emits gradients. It recomputes
probabilities from logits and state. Both are single device programs; there is
no host computation of logits, normalization or gradients in the fused path.

State is three FP32 vector tiles per 32 physical rows: maximum, selected index,
and log normalizer. Its tensor shape is `[1, 1, physical_rows, 96]`; the useful
values are columns 0, 32 and 64. It costs 384 bytes per physical row, **6 MiB**
for the full NanoLlama3 case. L1 stream buffers do not grow with vocabulary width.

Important implementation lessons:

- FP32 destination accumulation alone is insufficient. FP32 loads must use
  `UnpackToDestFp32`; the FPU copy path otherwise truncates them to TF32. This
  caused real FP32 regressions in the initial fused prototype.
- Use SFPU FP32 reductions and broadcasts rather than a reduction path that
  silently loses mantissa bits. Respect the four FP32 DST-tile register budget.
- Select only one maximum; masking every tied maximum changes the normalizer.
- Reinitialize SFPU operations when their programmable constants change.
- The original ten-second forward was a scalar TILE argmax scan on one device
  RISC core, not a readback of logits to the host. The composite now reuses
  ROW_MAJOR multicore argmax, grouping at most 256 rows to bound L1 scratch.
- `TTML_LOG_SOFTMAX_KERNEL_ROOT` was a local build override for compiling this
  worktree's kernels against another checkout's runtime. A normal build should
  use the committed default relative path with runtime roots set to this tree.

## Verification already performed

On one Blackhole P150a, **45 selected C++ tests passed** with existing tolerances:
43 inherited from the composite work plus two fused-specific tests. This is a
selected suite, not all tt-train tests.

- `PreservesConfidentLogProbabilitiesAndGradients`: BF16/FP32, axes 0/1/2/3/-1,
  gaps 8/18/24, padded shape `[3,3,5,35]`, positive and negative logit offsets.
  Checks output and backward against float64 arithmetic on the actual quantized
  inputs. Explicitly requires the dominant log-probability and gradient to stay
  negative/nonzero; absolute tolerance is zero in these confidence cases.
- `TiesAndGeneralUpstream`, `SingletonAxis`, large indices above BF16's exact
  consecutive-integer range, and repeated gradient accumulation.
- `LogSoftmaxTiedMaximumSelection` checks the composite helper's mask directly.
  It should not be mistaken for a direct fused-state test.
- `FusedLogSoftmaxPreservesFp32Upstream`: catches loss of FP32 upstream bits with
  BF16 input using a gradient that would become zero after premature rounding.
- `FusedLogSoftmaxWideRowsAndCacheReuse`: `[1,1,35,32003]`, both input dtypes,
  FP32 upstream, tied maxima, changing buffer addresses, and keeping one saved
  context live while a second call executes. Directly checks fused pivot state.

The 43 also include unrelated unary and reduction/softmax checks. They are not
43 independent paper-bug reproductions. None tests or fixes Section 3's
teacher/student loss cancellation.

Commit hooks and whitespace checks passed. Changed host sources were syntax
checked against this branch's headers. Hardware execution rebuilt/relinked
changed tt-train objects against the available Release TTNN/Metal libraries from
a checkout at `e2bf1a08417c9e635965630210e140c96fd273cc`; kernels were JIT compiled.
**A clean build of this branch and its dependencies has not yet been verified.**
Wormhole, multi-device execution and complete model training are also unverified.

## Performance and DRAM evidence

Single P150a, AMD Ryzen 9 9950X3D, BF16 `[64,1,256,32000]`, last-axis reduction.
Geometry comes from
[`training_shakespeare_nanollama3.yaml`](../configs/training_configs/training_shakespeare_nanollama3.yaml)
(batch 64) and [`nanollama3.yaml`](../configs/model_configs/nanollama3.yaml)
(256 tokens, vocabulary 32000). These are standalone operation measurements,
not measured full-model training time.

| Implementation | Forward ms | Backward ms | Combined ms | Forward peak GiB | Combined peak GiB | Forward retained GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original arithmetic | 19.52 | 19.36 | 38.81 | 2.933 | 3.907 | 0.977 |
| Corrected composite | 291.10 | 86.62 | 377.72 | 9.775 | 14.650 | 4.883 |
| Corrected fused | 21.24 | 34.67 | 55.79 | 0.982 | 1.959 | 0.982 |

The fused combined operation is about 6.8x faster than the composite and takes
44% longer than the original, buggy implementation. DRAM is incremental live
tensor allocation, measured separately using TTNN graph allocation capture.
Input and upstream were allocated before capture and together add 1.953 GiB.
Thus op-plus-input/upstream combined peaks are roughly 5.86 / 16.60 / 3.91 GiB.
These are not board-wide peaks including runtime reservations or other model
state. Phases are timed independently, so medians need not sum exactly.

Timing uses C++ `steady_clock`, eager host dispatch and explicit device
synchronization; it is **not Tracy or Python timing**. Nano uses one warmup and
one timed invocation per sample, three samples. Other cases use five warmups,
twenty calls per sample and seven samples. Original/composite/fused rotate order
using shared seeded input buffers; allocation tracing is outside timing.

Small-row, wide-vocabulary cases regress: `[1,1,32,32000]` combined is about
13.24 ms fused vs 1.85 ms composite; `[1,1,256,32000]` is 13.26 vs 6.59 ms.
Only one/eight cores handle those rows. The prototype has no cross-core row
reduction or dispatch heuristic selecting the fastest implementation. Keep this
limitation visible when describing its performance. See the full report/JSON
for the complete seven-case sweep and timing ranges.

## Transfer to another machine

Transfer the source changes with a Git bundle containing both feature refs and
the handoff commit. It depends on the common upstream base above; it is not a
standalone clone of all tt-metal history. No build directories or JIT caches
need to be copied.

On the source machine, from either feature worktree, after committing handoff:

```bash
git bundle create /workspace/tt-train-log-softmax-handoff.bundle \
  refs/heads/nmaurice/tt-train-fused-log-softmax \
  refs/heads/nmaurice/tt-train-log-softmax-collapse \
  ^62443b624e29d4fe032d37aa3f6a055ae0e26d2d
git bundle verify /workspace/tt-train-log-softmax-handoff.bundle
```

Copy the bundle to the other machine. In a full clone of tenstorrent/tt-metal:

```bash
git bundle verify /absolute/path/tt-train-log-softmax-handoff.bundle
git fetch /absolute/path/tt-train-log-softmax-handoff.bundle \
  refs/heads/nmaurice/tt-train-fused-log-softmax:refs/heads/nmaurice/tt-train-fused-log-softmax \
  refs/heads/nmaurice/tt-train-log-softmax-collapse:refs/heads/nmaurice/tt-train-log-softmax-collapse
git switch nmaurice/tt-train-fused-log-softmax
git submodule update --init --recursive
```

If verification reports the prerequisite commit missing, fetch
`62443b624e29d4fe032d37aa3f6a055ae0e26d2d` from upstream first. A shallow clone
may need that fetch or unshallowing. Existing diverged local feature branches
should be preserved/imported under different names rather than force-overwritten.

An optional `tt-train-log-softmax-evidence.tar.gz` accompanies the bundle with
the original final gtest and benchmark logs. The portable results are already
committed in the JSON. Old `/tmp/build_fused_logsoftmax*.py` helpers, executables,
and `/workspace/tt-metal/source-env.sh` contain machine-specific paths and are
not the recommended build or environment setup on the new machine.

## First steps on the destination

1. Confirm the branch contains `c0acf61b5ff` and this handoff. Preserve the
   composite ref/PR. Read repository build instructions and the tt-train path
   instructions; use the Metal/SFPU kernel skills when editing device code.
2. Install the normal tt-metal dependencies for the destination hardware and
   perform a clean Release build from this branch. The repository supports
   `./build_metal.sh --build-tt-train`. Then build the specific targets below.
   Resolving any clean-build incompatibility is the first unverified task.
3. Run the same numerical suite before profiling or optimizing. Match the
   recorded BF16/FP32 tolerances and the nonzero confidence assertions.
4. Rerun the timing/allocation benchmark and save results with destination
   hardware, runtime revision, build settings, and timing method. Use Tracy
   separately if attributing the remaining latency to kernels or transfers.

From the destination checkout root, after its normal environment setup:

```bash
export TT_METAL_HOME="$PWD"
export TT_METAL_RUNTIME_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cmake --build build --target ttml_tests log_softmax_benchmark -j 8

TT_METAL_OPERATION_TIMEOUT_SECONDS=10 flock /tmp/tt-device.lock timeout 300 \
  build/tt-train/tests/ttml_tests \
  --gtest_filter='UnaryOpsTest.*:PrecisionAndAxes/LogSoftmaxTest.*:TrivialTnnFixedTest.TestMax*:TrivialTnnFixedTest.TestStableSoftmax*'

TT_METAL_OPERATION_TIMEOUT_SECONDS=10 flock /tmp/tt-device.lock timeout 900 \
  build/tt-train/tests/log_softmax_benchmark \
  --benchmark_filter='^LogSoftmax(NanoLlama3)?/'
```

The Google Benchmark CPU column is not device kernel latency. Consume the
`LOG_SOFTMAX_CONFIG`, `LOG_SOFTMAX_TIME` and `LOG_SOFTMAX_MEMORY` records.
The full filter includes the non-last-axis case. To run just Nano, use
`^LogSoftmaxNanoLlama3/`. Do not accidentally include the historical serial
argmax diagnostic with a broad filter.

## Remaining work, in suggested order

1. Clean-build and reproduce on destination hardware; add Wormhole verification
   if available. Check standard tt-train CI placement before publishing the
   fused branch; no fused PR has been created by this effort.
2. Profile the fused backward and the wide/few-row cases. Consider splitting
   rows across cores, reducing repeated passes/loads, or choosing a dispatch
   strategy by geometry. Treat these as proposals, not measured improvements.
3. Extend correctness coverage as supported formats/devices expand. Keep
   padding, ties, selected-gradient reconstruction, native precision, upstream
   precision and cache-address reuse checks.
4. Design the logits-aware teacher/student loss needed for the second bug.
   Do not claim that a rounded log_softmax output fixes that loss by itself.
5. Apply the arithmetic to independent TTNN/Moreh and tt-train fused/vocabulary-
   parallel losses, using the corrected standalone operation as a reference.
   Benchmark memory and time for each integration and verify model loss/gradients.

No implementation, numerical tolerance or composite PR changes are part of
the handoff documentation update.
