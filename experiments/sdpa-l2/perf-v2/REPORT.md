# SDPA accuracy/performance follow-up — September 9, 2026

Implemented and rebuilt the retained candidate. FP32 remains below 0.5% on
the tested normal-input aggregate metric, but its overhead is still large.
BF16 streaming gains substantially better accuracy for about 5.5% overhead.

Latest FP32 optimization: [perf-v3/REPORT.md](../perf-v3/REPORT.md),
4597.94 ms at 0.491831% L2 versus a fresh 5095.71 ms control.
The results below describe the preceding implementation.

Follow-up: [distribution sweep and maximum relative error](../distributions/REPORT.md)
tests six matched input distributions. Scaled Q/K and outliers exceed 0.5%
FP32 L2; the normal-input result must not be generalized to those cases.

## Final matched results

Non-causal H=10/S=256K/D=128, seed 1236; 20 warmups and seven timed replays:

| Destination/path | Relative L2 % | Median trace ms | Overhead vs same-destination main |
|---|---:|---:|---:|
| Main FP32 | 9.080440 | 3315.52 | — |
| Improved FP32 | 0.488394 | 5095.30 | +53.68% |
| Main BF16 streaming | 18.964064 | 2777.65 | — |
| Improved BF16 streaming | 3.189697 | 2930.22 | +5.49% |

The FP32 schedule is about 5.6% faster than the previous improved version
(5395.74 ms), with unchanged measured L2. It is **not** close to main's
performance. BF16 relative L2 falls by 83.18% (5.95x smaller).

Raw final timing files: main-fp32-steady.jsonl, candidate-fp32-steady.jsonl,
main-stream-steady.jsonl, candidate-stream-steady.jsonl. BF16 measured ranges
were 2770.93–2792.39 ms (main) and 2925.80–2932.28 ms (candidate); FP32 ranges
were 3312.52–3321.85 ms and 5094.96–5095.74 ms. Do not overinterpret hundredths
of a percent in the overhead ratios.

## Scope and measurement

Blackhole P100A, reservation 214149, yyzo-bh-26; not a Galaxy measurement.
Base main commit: 2ba6fc2339d53300ae87c5202f335ef56492cfb3.
Primary problem: non-causal B=1, H=10, Q=K=262144, D=128,
Q chunk=128, K chunk=512, BF16 Q/K/V/output, HiFi2.
The FP32 variant enables destination accumulation; the BF16 variant still uses
compute_streaming with BF16 destination accumulation.

All matched runs retain the earlier bit-ceiling Q preprocessing and scale
compensation (c=1.0027). Reference inputs are the original BF16 tensors, not
preprocessed Q. Relative L2 is 100*||actual-reference||2/||reference||2.
Host Q preprocessing costs about 105 ms for this shape and is excluded from
the trace times. FP32 still needs this preprocessing for the stated target.
BF16 does not: a separate same-input sampled-device check with **no** Q
preprocessing gives 3.149637% L2 (final-bf16-no-preprocess.jsonl). The BF16
compensation can therefore be used with untouched inputs and no added host
preprocessing step. Its no-preprocessing full-prefill time was not remeasured.
The FP64 reference checks 128 query rows per head; this is not an exhaustive
reference of all 256K output rows. Timings are warmed full-operation trace
replays, excluding host preprocessing, compilation, and transfers.

BF16 needs a longer warmup on this power-limited card: its replay latency
ramps upward over the first several iterations. Main FP32 also drifts with
short warmup. Final comparisons use 20 unmeasured replays followed by seven
measured replays, on both main and candidate. Do not compare the candidate's
steady-state time with main's earlier two-warmup result.
These are blocking trace-replay wall times, not newly collected FPU counters.

The added --sampled-device mode generates the same full Q/K/V tensors but
executes only the selected non-causal query rows. Its accuracy matched the
full run exactly in the checked cases. Its timings are diagnostics, NOT
full-prefill measurements.

## Retained FP32 optimization

Keep the previous numerical recipe: 10-bit negative-logit fast-exp grid,
cubic refinement, six-fraction-bit P rounding, compensated FP32 L1 numerator
update, accurate denominator update/correction, and final normalization.
Large QK/PV matmuls remain HiFi2.

The exp refinement now interleaves two independent Horner chains and unrolls
four pairs. Three polynomial constants move to programmable SFPU registers
to avoid working-register pressure. Those registers overlap the fast-exp
grid constants, so the grid is restored before each exp tile.

This preserved the measured L2: 0.48839445% on seed 1236.
Full-shape time fell from the earlier 5395.74 ms to 5094.85 ms.
This is a modest improvement, not a low-overhead solution: against the earlier
3287.43 ms main FP32 baseline, substantial overhead remains.

The cubic refinement still runs per score tile, while the state corrections
add work every K chunk. On this compute-intensive shape that work is no
longer hidden by the other stages. The faster quadratic alternatives miss
the numerical target, and reducing the small delta multiply to HiFi2 also
fails. The retained schedule is the fastest passing implementation among
the tested variants, not a proof of globally minimum overhead.

## BF16 streaming: compensate both states

Merely improving small correction multiplies or correction exp does not fix
the long-context accumulation problem. Simply widening buffers also does not
preserve values through a BF16 destination update.

Each running state is now represented as two BF16 parts, hi + lo.
The SFPU computes:

    value = (old_hi + old_lo) * correction + current_chunk
    new_hi = RNE_BF16(value)
    new_lo = RNE_BF16(value - new_hi)

Both the denominator and numerator use this update. The QK/PV computation,
fast logit exp, BF16 destination flag, and streaming path are unchanged.
The extra two-part storage costs 80 KiB/core for the primary geometry.
No Float32 circular-buffer format is required.

The update runs on PACK, allowing UNPACK/MATH to prepare the next DST half.
Current-chunk L1 writes are fenced once per row group, not once per tile.
Packer setup/accumulation toggles are hoisted, and pairs of numerator tiles
share a broadcast correction (seven BF16 DST tiles per batch).

Why both states matter:

- Denominator-only compensation: full-shape normal-input L2 4.1544%, at
  about the original streaming time. But constant-V L2 at 256K was 25.97%.
  Accurate normalization exposed the still-drifting numerator. Rejected as
  the final implementation; historical snapshot: denominator-only.patch.
- Compensating both: full-shape normal-input L2 3.1897%; constant-V diagnostic
  L2 0.5304% at 256K. The initial MATH-thread implementation took 3085.80 ms.
- Moving compensation to PACK and hoisting synchronization/setup reduced
  the two-warmup full-shape time to about 2836 ms without changing measured
  normal-input L2. With 20 warmups, the retained version measures 2930.22 ms
  against matched main at 2777.65 ms: **5.49% overhead for 83.18% lower L2**.

Activation is deliberately narrow: Blackhole, BF16 Q/K/V, HiFi2 with both
approximation flags, Q chunk=128, K chunk=512, Q/V head dimensions=128,
at least 64 K chunks, and no MLA, chunked/windowed mode, explicit mask,
sliding window, or attention sink. Other BF16 configurations retain main's
path. Broader deployment needs separate validation.

## Rejected FP32 alternatives

These are experimental measurements, not interchangeable configurations.
Full measurements below use the primary shape/seed unless noted.

| Experiment | L2 % | Trace ms | Reason not retained |
|---|---:|---:|---|
| Quadratic, 10-bit grid | 0.502588 | 5170.11 | Fails 0.5% |
| Quadratic, 11-bit grid, paired | 0.502531 | 5025.98 | Fails 0.5%; narrower range |
| Move exp+refinement to PACK | 0.488394 | 5639.74 | Slower |
| Paired cubic, no unrolling | 0.488394 | 5190.52 | Unrolling is faster |
| Full DST sync / larger blocks | 0.488394 | 5754.39 | Slower |
| Six-piece LUT | 0.495527 | 5304.45 | Slower than paired cubic |
| Hand-scheduled paired cubic | 0.488394 | 5096.24 | No benefit over simpler SFPI |
| HiFi2 small delta correction | 0.559026 | 5061.20 | Fails 0.5% |
| Relative-error fitted quadratic | 0.502316 | 4964.89 | Fails 0.5% |

Q-scale sweeps with the quadratic also approached the threshold, but were not
accepted by rounding a result down to 0.5%. Compiler/format failures and
invalid intermediate experiments are retained in the raw logs, not counted
as successful optimizations.

## Reproduction

From the built repository in the reserved container:

    export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
    export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
    python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
      --kv-lens 262144 --full --heads 10 --variants hifi2 fp32_hifi2 \
      --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 \
      --benchmark-iters 7 --label candidate --output candidate.jsonl

Use only fp32_hifi2 with --max-l2-pct 0.5 for the FP32 accuracy gate.
Use --sampled-device only for non-causal accuracy sweeps, never for the
full-prefill performance comparison.

For the steady-state comparison, add --benchmark-warmup 20.

## Regression results

- Normal H=10/S=256K, seeds 1234/1235/1236, FP32: 0.489757%,
  0.491865%, 0.488394% L2. The first two use sampled-device execution;
  seed 1236 also matches the full operation exactly. All pass the explicit
  --max-l2-pct 0.5 gate. Per-row p95 is about 0.56%, not below 0.5%.
- Constant V, BF16 streaming, Q=128/H=1: 0.570475% at S=32K and
  0.530409% at S=256K. This catches the denominator-only regression.
- Uniform attention plus constant V, BF16: 0.78125% at both S=32K and
  S=256K. Compensation does not eliminate the remaining chunk arithmetic
  and normalization errors, but this result does not grow with K length.
- Outside the BF16 activation gate, D=64/S=4K: smoke/finite/trace-equality
  checks passed (L2 2.5256%). This is not a new accuracy claim for that shape.
- Full causal H=4/S=32K, 128 reference rows spread across the sequence:
  BF16 aggregate L2 0.925643%, FP32 0.109949%. Early query rows have
  larger output norms, so this aggregate is not comparable to tail-only
  sampling; row-median L2 is 2.4091% and 0.4854%, respectively.
- The repro validates its blockwise FP64 reference against dense softmax,
  checks finite outputs, and requires trace output to equal ordinary output
  exactly. These checks passed in the reported final regression runs.

Raw results: final-seed-1234.jsonl, final-seed-1235.jsonl, final-seeds.jsonl
(seed 1236), final-constant-v.jsonl, final-causal.jsonl.

## Source and verification

The retained combined source delta against main is final-candidate.patch.
The repro is tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py.
Main comparisons temporarily restored all four changed tracked source files
to HEAD, verified an empty diff for those files, and rebuilt. The final
candidate was then restored and rebuilt in the same container:

    CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install

Build logs are main-final-build.log, main-steady-build.log, and
restored-final-build.log. C++ formatting used git-clang-format with
clang-format 19.1.4; Python passed black 23.10.1. No product dependency,
toolchain, or power/clock changes were made. Local/remote source diffs match
(apart from Git's abbreviated object-ID lengths), and git diff --check passes.

## Limitations

This remains an experimental worktree, not a production-readiness claim.
The FP32 target is a normal-input aggregate-L2 result; per-row tails and
outlier/model-derived distributions need separate acceptance criteria.
BF16 streaming improves substantially but does not meet the FP32 0.5% goal.
The remaining FP32 overhead is still large on this compute-intensive shape;
no near-main-performance claim is justified by these experiments.
