# Faster improved FP32 SDPA — September 9–10, 2026

Historical iteration. The newer FP32 streaming implementation and the 3.32-second
target work are in [perf-v4/REPORT.md](../perf-v4/REPORT.md).

## Outcome

Retained a faster **non-streaming FP32-destination** implementation. Both large
QK and PV matmuls remain **HiFi2**; approximate exp and the existing Q
preprocessing are retained. The three normal-input seeds pass the explicit
0.5% aggregate L2 gate: **0.496206%, 0.495448%, and 0.491831%**.

At the same full 256K shape, the final matched comparison is **4597.94 ms
versus 5095.71 ms: 9.77% less time (1.108x speedup)**.

| FP32 implementation | Median trace ms | Relative L2 % | Protocol |
|---|---:|---:|---|
| Previous improved, freshly rebuilt control | 5095.71 | 0.488394 | 40 warmups / 10 measured |
| New improved | 4597.94 | 0.491831 | 40 warmups / 10 measured |
| Unmodified main, earlier perf-v2 run | 3315.52 | 9.080440 | 20 warmups / 7 measured |

The new path remains approximately **38.68% slower than main FP32**, versus
53.69% for the previous improved path. It removes approximately 27.96% of
that overhead; this is progress, not near-main performance. Main was not
remeasured in this round. The control range is 5094.84–5098.63 ms; the new
range is 4594.89–4603.79 ms. Raw data: control-w40.jsonl and final-w40.jsonl.
Full-operation output and sampled-device output agree on the referenced
tail rows, including exactly matching L2 0.4918307524%.

This is not distribution-independent accuracy. The outlier case increases
from 0.629698% to 0.683394%; common-Q/K failures remain. The 0.5% requirement
is met on the tested normal inputs, not every row or every distribution.
The BF16 streaming implementation is unchanged by this round.

Review [incremental-fp32.patch](incremental-fp32.patch) for this round only:
three C++ files, 110 insertions and two deletions. The original accurate
refiner is retained as the fallback. [final-candidate.patch](final-candidate.patch)
contains the complete worktree delta from main, including the prior BF16 work.
Both patches passed reverse-apply checks against the final sources.

## Measurement contract

Blackhole P100A, yyzo-bh-26, reservation 214149; **not Galaxy**.
Worktree: cglagovich/blackhole-work-20260908, based on main
2ba6fc2339d53300ae87c5202f335ef56492cfb3.

Primary timing: noncausal B=1, H=10, Q=K=262144, D=128,
Q chunk=128, K chunk=512, BF16 Q/K/V/output, FP32 DST.
Reference is FP64 on the original BF16 inputs, before Q preprocessing.
We reference 128 query rows/head, not every output row.
Relative L2 = 100 * ||actual-reference||2 / ||reference||2.

Blocking trace-replay wall times exclude compilation, transfers, and host Q
preprocessing. They are not raw device-cycle durations. Preprocessing is
still the existing six-fraction-bit bit-ceiling operation with c=1.0027 and
attention scale 1/(sqrt(D)*c); it takes approximately 105–115 ms on this host
for full Q, excluded from the trace measurement.

Early full-shape experiments use 20 warmups / 7 measured replays, matching
perf-v2. Final control/candidate comparison uses 40 warmups / 10 replays;
the power-limited card's timing drifts with a short warmup.
Short-Q and sampled-device timings are diagnostic only.

## What changed

### Make HiFi2 truncation do the weight rounding

On the BH FPU, for BF16 operands:

- HiFi2 uses all seven SrcA fraction bits, but only the high six SrcB bits.
- In PV, P is SrcB and V is SrcA. P is stored in an FP32 CB, but the TF32
  unpack plus HiFi2 phases consume only those six high fraction bits.
- LoFi with a SrcA value of exactly one produces exactly the same effective P:
  SrcA's low bits are zero, so HiFi2's additional phase adds nothing.

The old implementation explicitly rounded each exp output to six fraction
bits using shifts and a BF16 conversion, then summed those rounded values.
The new polynomial includes a half-six-bit-ULP bias. The FPU truncation
therefore approximates rounding without those per-element instructions.

**The denominator must use those same effective weights.** Summing the
untruncated FP32 values would overestimate the normalization and introduce
bias. The new reduce_effective_p helper performs P @ column_ones in LoFi,
batched over four query tiles. It is a small extra FPU reduction, not a change
of the large QK/PV matmuls to LoFi.

### Factor the exponential polynomial to avoid exponent bookkeeping

Let z be the fast macro's positive, normal float encoding and
m = setexp(z, 127). The previous refinement formed p(m), extracted z's
exponent, added 96, and installed that exponent on p.

Instead, fit a cubic q such that

    m * q(m) ~= 0.995 * 2^(m-1) + 1/128,  m in [1,2]
    result = z * (2^96 * q(m))

The common 0.995 factor cancels in softmax normalization. The additive 1/128
is half one six-bit mantissa ULP. Folding 2^96 into the coefficients replaces
EXEXP + integer add + SETEXP with one multiplication per vector. Two Horner
chains are still interleaved, unrolled four pairs, as in the prior candidate.

The fit is input-independent; it uses a uniform 65,536-point grid, not the
test seeds. check_exp_model.py reproduces the constants. Its relative RMS
error against the biased target is 0.03955%, with extrema -0.14261%/+0.05656%.
This is a deliberately restricted SDPA negative-logit approximation, not a
general exp implementation or a guarantee over arbitrary dynamic range.
The negative underflow encodings remain negative and packer ReLU clamps them.
The CPU model does not emulate SFPU rounding or subnormal behavior.

### Simplify denominator state and final normalization

Each chunk denominator is now fully row-reduced. Therefore:

- Rescale only its first column, and form correction - 1 in the same SFPU pass.
- Skip the old final matrix reduction.
- Keep the denominator's full-FP32 unpack view through reciprocal, removing
  the old copy through a TF32 view that was needed for that matrix reduction.

Retain the existing accurate small correction exp, two-Newton reciprocal,
FP32 L1 numerator update, and SFPU normalization. The small numerator-delta
multiply is still HiFi4, as in the previous improved version. “HiFi2” refers
to the two large matmuls; this is not a claim that every operation uses HiFi2.

### Activation and fallback

The new SDPA_HIFI2_EFFECTIVE_WEIGHTS define is limited to Blackhole FP32 DST,
BF16 Q/K/V, HiFi2 with both approximation flags, noncausal Q/K chunks 128/512,
Dq=Dv=128, at least 64 K chunks, and no MLA, chunked/windowed mode, explicit
mask, sliding window, generated padding mask, or attention sink.

Other configurations retain the preceding improved FP32 implementation.
The BF16 streaming guards and compensation remain unchanged.
This is an experimental worktree, not a broadly validated production patch.

## Accuracy regression sweep

Generate full Q/K/V at S=256K and execute 128 sampled query rows/head.
The normal seed gate uses tail rows, matching the full timing reference.
All other entries below use seed 1236 and evenly spread query rows, matching
the earlier distribution/common-mode reports. Each case covers 163,840
output elements, not maxima over the full 256K-query output.

All entries below are percentages. Previous values are from perf-v2's
distribution/common-mode follow-ups; they were not all rerun as controls.

| Input | Previous FP32 L2 | New FP32 L2 | New max elementwise relative error |
|---|---:|---:|---:|
| normal | 0.488340 | 0.494108 | 140130.149829 |
| scaled_qk | 0.865743 | 0.878116 | 150389.295122 |
| outliers | 0.629698 | 0.683394 | 363381.743798 |
| biased_v | 0.178769 | 0.181385 | 0.609702 |
| uniform | 0.186573 | 0.187338 | 50.974007 |
| constant_v | 3.621e-14 | 3.621e-14 | 1.110e-13 |
| uniform_constant_v | — | 0 | 0 |
| common_q +8 | 1.144766 | 1.156820 | 2120301.894395 |
| common_q +32 | 2.752265 | 2.753570 | 120899.328091 |
| common_k +8 | 0.567969 | 0.576196 | 40659.845287 |
| common_k +32 | 1.120380 | 1.124906 | 91991.205195 |
| common_v +8 | 0.040825 | 0.041629 | 0.317038 |
| common_v +32 | 0.010510 | 0.010510 | 0.051337 |

Maximum relative error has no epsilon floor. Near-zero reference elements
make it enormous; low aggregate L2 does not imply a small maximum.
Raw files include the worst element, RMS, filtered maximum, per-row L2,
BF16 rounding floor, and the residual-normalized metric for common V.
At V+32, the ordinary L2 matches the ideal BF16 output rounding floor; that
does not mean the small residual signal is preserved.

Regression results:

- Normal seeds 1234/1235/1236: 0.496206/0.495448/0.491831% L2, all pass.
- BF16 streaming, no Q preprocessing: 3.162034% L2, identical to the earlier
  matched normal/spread result.
- Causal fallback, H=4/S=32K/spread: 0.108631% L2.
- D=64/S=4K fallback: 0.481489% L2.
- Additional active-path lengths, seed 1236: 32K/64K/128K give
  0.497303/0.495615/0.490623% L2, respectively; all pass the 0.5% gate.
- Finite checks, FP64-reference self-checks, and exact ordinary/trace output
  equality pass in all 19 device cases.
- Constant V produces one to FP64-reference roundoff; Q=0,V=1 is exact.

Raw final sweep: final-accuracy-all.jsonl, extracted from the complete
final-full32-recip-accuracy.log. Individual numbered JSONL files are also
retained. The extra three length checks are in final-lengths.jsonl.
validate.sh regenerates these 22 cases and gates normal L2 at 0.5%.

## Profiling and alternatives

128K full-operation profiles avoid overflowing the 32-bit counters that
would be unsafe for these multi-second 256K operations. These are profiler
busy-cycle utilization metrics, not throughput-derived peak-TFLOP estimates.

| Version, H=10/S=128K | FPU utilization | SFPU utilization | MATH utilization |
|---|---:|---:|---:|
| Previous improved FP32 | 25.62% | 48.37% | 73.99% |
| Factored effective-weight candidate, before final normalization cleanup | 31.72% | 35.79% | 67.52% |
| Final rebuilt candidate | 31.80% | 35.87% | 67.67% |

Median SFPU busy cycles fall from 826,880,484 to 521,711,400 (-36.9%).
FPU cycles rise from 438,049,344 to 462,404,928 (+5.6%) due to the new
effective-weight reduction. The reduced SFPU work more than compensates.
Raw profiles/logs: start-profile.*, factored-final-profile.*, final-profile.*.

I implemented and tested FP32 streaming variants, including PACK-thread exp,
full-FP32 state movement, accurate state updates, L1 accumulation, paired SFPU
state updates, format-reconfiguration reduction, larger query chunks, and a
BF16 P alias. Correct streaming required handling FP32/TF32 unpack formats
explicitly; the existing BF16 assumptions were not safe to reuse directly.

The tested FP32 streaming schedules did not beat the retained standard path.
The first correct full 256K streaming implementation took 6146.09 ms at
0.487855% L2. Later short-Q refinements reached 27.77 ms; the retained
non-streaming implementation is about 23.2 ms in that diagnostic. A Q=256
streaming chunk required single-buffered K/V to fit L1 and measured
1267.67 ms for full 128K (only two warmups). Those are distinct experiments,
not a full-shape matched comparison of the best possible streaming design.
Streaming remains a possible future direction, not a demonstrated win here.

Other informative trials:

| Trial | Scope/protocol | Result |
|---|---|---|
| Prebiased cubic + effective-weight denominator, explicit exponent restore | Full 256K, 20/7 | 4757.27 ms, 0.490490% L2 |
| Factored cubic before final TF32-copy removal | Full 256K, 20/7 | 4579.69 ms, 0.493248% L2 |
| HiFi2 QK / HiFi3 PV, unrounded cubic exp | Full 128K, 2/5 | 1195.50 ms, 0.365756% L2 |
| HiFi2 QK / HiFi3 PV, unrounded quadratic exp | Full 128K, 2/5 | 1192.67 ms, 0.382087% L2 |
| BF16 P alias in FP32 streaming | Short Q=128 / K=256K | 28.48 ms, no benefit |
| Increase original refiner unroll to 16 | Short Q=128 / K=256K | 27.26 ms, no benefit |
| Increase factored refiner unroll to 8 | Q=1408 / K=256K | 23.25 ms, no clear benefit; retained 4 |

The mixed HiFi3 variants are alternatives, not “all-HiFi2” results. Their
JSON configuration still says HiFi2 because PV was overridden within the
kernel. Likewise experimental FP32-streaming JSON files retain the repro's
default streaming=false field; their labels/patches identify the actual
experimental path. Final retained-kernel metadata is consistent.

Raw failed/debug experiments are retained for audit but are not accepted
results. The retained implementation is the fastest passing option among
the tested variants, not proof of a global optimum.

## Reproduction

From the built reserved-container repo:

    export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
    export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
    python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
      --kv-lens 262144 --full --heads 10 --variants fp32_hifi2 \
      --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 \
      --benchmark-warmup 40 --benchmark-iters 10 --max-l2-pct 0.5 \
      --label final --output final.jsonl

    bash experiments/sdpa-l2/perf-v3/validate.sh final
    python_env/bin/python experiments/sdpa-l2/perf-v3/check_exp_model.py

The final Release build/install passed (final-build.log). C++ changes and
Python tools were formatted; git diff --check, shell syntax, Python syntax,
and patch reverse-apply checks passed. The rebuilt candidate passed the
three extra length gates, full 256K L2 gate and exact trace-equality check,
128K profiler run, and polynomial-model self-check. The earlier 19-case
sweep used numerically identical code before whitespace/comment cleanup.
Both the local worktree and allocated container retain the final candidate.
