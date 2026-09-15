# Full-HiFi4 SDPA: residual accuracy investigation

Status (2026-09-11): investigation, qualification, holdout, timing controls,
source restoration, and regression checks complete. The accurate candidate
does not pass the unchanged qualification contract; this is not a release sign-off.

## Conclusions established by the ablations

HiFi4 is the appropriate starting point for an accurate tier, but it is not
sufficient by itself. There are two distinct findings:

1. **SDPA-specific losses are avoidable.** Remove the HiFi2-specific probability
   bias, retain original BF16 Q, and preserve FP32 scores through max subtraction.
   Full-FP32 subtraction matters more than replacing the improved approximate
   exponential with accurate exp.
2. **The remaining outlier failures are reproduced by ordinary HiFi4 matmul.**
   Taking device-computed QK scores and doing everything after QK in FP64 closely
   reproduces the failing SDPA rows. FP32 destination and all four multiplication
   phases do not imply IEEE-FP32-quality multi-term dot products.

Do not explain the residual failures away as PCC artifacts, near-zero-element
relative errors, or excessive context length. They include ordinary head L2
and vector-valued row L2 failures against the agreed gates. Nor should we call
them an unavoidable silicon limit: these experiments isolate generic matmul
arithmetic, but do not distinguish every possible LLK configuration issue from
the FPU's internal reduction behavior.

## Controlled SDPA ablations

All entries are relative L2 percentages. These are the same six failure-directed
heads used in the preceding investigation, not averages or maxima over the
expanded qualification suite. Input hashes and reference positions are preserved.
N is both Q and K length; B=1, D=128, H=5 or 10 as recorded in each JSON.

| N | Distribution | Retained HiFi2 | HiFi4, old exp | HiFi4, unbiased exp | HiFi4, accurate exp | HiFi4, FP32 subtraction + accurate exp |
|---:|---|---:|---:|---:|---:|---:|
| 32768 | Normal | 0.495150 | 0.237896 | 0.205676 | 0.200985 | **0.174234** |
| 32768 | Q/K scaled 2x | 0.857172 | 0.319625 | 0.298166 | 0.294727 | **0.193989** |
| 32768 | Sparse outliers | 0.851759 | 0.305752 | 0.301638 | 0.294394 | **0.267216** |
| 262144 | Normal | 0.493544 | 0.233993 | 0.204182 | 0.200119 | **0.173400** |
| 262144 | Q/K scaled 2x | 0.838897 | 0.318737 | 0.303351 | 0.299988 | **0.197174** |
| 262144 | Sparse outliers | 0.933626 | 0.378522 | 0.379924 | 0.373932 | **0.301248** |

Retained HiFi2 uses the earlier Q bit-ceiling and 1.0027 scale compensation.
Every HiFi4 column instead uses original Q and the original attention scale.
HiFi4 applies to QK, PV, and the probability-times-ones denominator, so numerator
and denominator consume matching probability precision. Destination registers,
scores, numerator, denominator, and correction storage are FP32; maxima remain
BF16. Q/K chunks are 128/1024 in these ablations. Input buffering is unchanged
from the retained FP32-streaming layout.

The old polynomial deliberately compensates six-bit HiFi2 probability effects.
The unbiased version fits exp2(m-1)/m over m in [1,2] on a uniform grid, without
using qualification inputs. Both use the same 10-bit logit grid and cutoff.
Accurate exp uses the existing `_sfpu_exp_fp32_accurate_` implementation, not
an exp coefficient fit to this suite.

Merely selecting accurate exp still routes scores through the FPU operand path
for subtraction, narrowing them to TF32. The FP32-subtraction diagnostic adds
a direct-to-DST alias of the existing score CB and subtracts the row maximum
in SFPU FP32. The alias shares storage: no score copy, larger Q/K chunks, or
additional Q/K/V traffic is introduced. Its conservative one-tile schedule is
not a completed performance optimization.

With that subtraction fixed, unbiased approximate exp gives respectively
0.179224%, 0.194956%, 0.266810%, 0.178844%, 0.200298%, and 0.299185% L2 on the
same six heads. Accurate exp is only a small further improvement on normal
inputs and does not resolve the outlier residual. The six-head experiment alone
passes the strict gates, but the expanded sweep finds failures; it must not be
used as a substitute for qualification.

Raw ablations: `unbiased-v2.jsonl`, `accurate-exp.jsonl`,
`fp32-sub-accurate.jsonl`, `fp32-sub-unbiased.jsonl`. Earlier HiFi2/HiFi4 controls
are in `../stress-analysis/`.

## Attribution of the remaining error

For each selected input, run full SDPA, then an independent HiFi4 QK matmul on
the sampled queries. Use those device scores with FP64 softmax and FP64 PV,
rounding only the final output to BF16. This is a different matmul kernel, not
an exact dump of SDPA's score CB, so agreement of the error vectors is important.

| Case | Head | SDPA L2 % | QK-device-only model L2 % | SDPA worst row % | Model worst row % | Error-vector cosine |
|---|---:|---:|---:|---:|---:|---:|
| 32K outliers, H10, seed 1235 | 9 | 0.267216 | 0.265542 | 1.239667 | 1.228631 | 0.983801 |
| 256K outliers, H10, seed 1234 | 3 | 0.301248 | 0.301234 | 1.216221 | 1.220274 | 0.990474 |
| 32K outliers, H10, seed 1237 | 0 | 0.389476 | 0.388085 | **2.829064** | **2.828903** | 0.988812 |
| 256K outliers, H5, seed 1238 | 1 | 0.389803 | 0.390524 | **2.484923** | **2.482404** | 0.994152 |
| 256K outliers, H10, seed 1234, final suite's worst | 2 | **0.662220** | **0.663082** | **3.459265** | **3.475663** | **0.997819** |

The last three heads were selected from newly observed qualification failures;
the last was added after completing the full suite.
The model reproduces the failing rows without any streaming state, approximate
exp, device reciprocal, device denominator, or device PV arithmetic.

This is consistent with softmax sensitivity, not a requirement for error to
grow with context length. For a score perturbation ds, the first-order output
change is `dO = sum_j p_j * ds_j * (V_j - O)`. Large absolute score errors near
competing attention peaks can change the mixture appreciably even when relative
QK-matrix L2 is small. In the 32K/seed-1237 failing head, an FP64-score model with
ten-bit probabilities and BF16 output has only 0.144785% head L2 and 0.268833%
worst-row error, versus 2.829064% worst-row error on the device. The row failure
is not imposed by BF16 output resolution.

Additional controls:

- Replacing V with ones gives exactly one on all sampled elements in all nine
  diagnostic cases. Constant V remains excluded from release qualification;
  this is an attribution probe, not a change to the suite.
- Reversing K and V together preserves the large outlier error; the outlier
  original/reversed output difference is only 0.027–0.036% L2 in these probes.
  This argues against accumulated online-softmax drift as the dominant source.
  Normal-output permutation differences are larger (~0.14%) and are not claimed
  to be bitwise invariant.
- A CPU-only ten-bit probability-quantization model stays near the final BF16
  rounding floor and does not reproduce the failing rows.
- Removing a single global gain error does not materially remove the residual.

Raw data and reproducer: `residual.jsonl`, `residual-final-worst.jsonl`, `residual.py`.

## Standalone matmul controls

All these tests use HiFi4, FP32 destination, FP32 output, and BF16-valued inputs.
No SDPA code or softmax is involved.

An isolated product tests the low-bit cross term directly:

    (1 + 1/128) * (1 + 7/128) = 1.06292724609375

HiFi2 returns 1.0546875; HiFi3 returns 1.0625; HiFi4 returns the exact product.
Thus HiFi4 really consumes the BF16 bits missing from HiFi3. But dense random
dot products still have 0.031789% L2 at inner dimension 32 and 0.032029% at
inner dimension 128. The constructed high-dynamic-range case has 0.029971% L2.
Promoting the same BF16 values to FP32 input storage does not improve these
errors; reported metrics and sample outputs match the BF16-input runs.

All 12 BF16-input matmul probe cases were also rerun on the exact unmodified
main sources, after rebuilding with a fresh kernel cache. Every recorded metric,
sample output, and low-bit statistic matches the earlier probe. Thus this
matmul behavior is not introduced by our SDPA changes. These comparisons are
of the recorded statistics and samples, not full-tensor hash comparisons.

To isolate multi-term reduction, mask Q lanes into interleaved partitions,
run the same matmul separately for each partition, then sum its FP32 outputs
in FP64 on the CPU:

| Passes | Normal dot-product L2 % | High-dynamic-range dot-product L2 % |
|---:|---:|---:|
| 1 | 0.0320286 | 0.0299712 |
| 2 | 0.0234847 | 0.0222423 |
| 4 | 0.0141476 | 0.0118590 |
| 8 | **0.00000281** | **0.00002320** |
| 16 | 0.00000168 | 0.00001246 |
| 128, one product per pass | 0 | 0 |

The eight-pass result sharply localizes the error to the multi-term matmul
reduction rather than final output packing or omitted BF16 multiplication
phases. It does **not** establish that eight passes are necessary, optimal,
or an acceptable production solution. Nor does it establish an exact internal
bit width without an independently validated hardware model.

Reproducers: `matmul_probe.py`, `partition_probe.py`; raw data:
`matmul-probe.jsonl`, `matmul-probe-main.jsonl`, `matmul-fp32-input.jsonl`,
`partition-probe.jsonl`.
The ISA checkout used was `../tt-isa-documentation` at
`5287a62727350bcef35f7b411d1b8a706172ec4c`, specifically the shared MatrixUnit,
SrcASrcB, and MVMUL documentation. Its floating-point functional model is
explicitly approximate, not a bit-accurate guarantee of FP32 reduction.

## Expanded qualification

**162 PASS, 8 FAIL, 220 UNSUPPORTED**, out of 390 requested cases after the agreed
exclusions. That is **95.29% of the 170 executed cases**, versus **97/170 = 57.06%**
for the preceding improved HiFi2 FP32 implementation on the same supported
subset. Unsupported cases are not passes. The accurate tier still does **not**
qualify under the unchanged strict contract.

| Group | Executed cases passing | Worst head L2 % | Lowest defined PCC | Maximum row p99 % | Worst sampled row % |
|---|---:|---:|---:|---:|---:|
| Normal lengths | 40/40 | 0.176597 | 0.999998436 | 0.210696 | 0.231857 |
| Supported boundary, N=33280 | 10/10 | 0.176825 | 0.999998430 | 0.210524 | 0.235929 |
| Q/K scaled 0.5x | 20/20 | 0.175017 | 0.999998458 | 0.212617 | 0.222343 |
| Q/K scaled 2x | 20/20 | 0.205271 | 0.999997907 | 0.383769 | 0.548982 |
| Sparse outliers | **12/20** | **0.662220** | **0.999978089** | 0.986190 | **3.459265** |
| Uniform attention | 20/20 | 0.206353 | 0.999997874 | 0.206353 | 0.206353 |
| Zero V | 20/20 | N/A | N/A | 0 | 0 |
| Exact cancellation | 20/20 | N/A | N/A | 0 | 0 |

Each metric column is its own extremum over the group, not necessarily the
same head. All eight failing cases fail the worst-row gate. Three also fail
head L2, and one of those fails PCC. No p99 gate fails. The largest failure is
N=262144, H=10, seed 1234, head 2: 0.662220% head L2 and 3.459265% worst row.
Normal inputs cover H=5/10 and all five original seeds at 32K/64K/128K/256K;
the other normal lengths remain unsupported by this FP32-streaming specialization.
PCC alone would catch only one of these eight failures; five are caught only
by the worst-row gate. This is a concrete reason to retain more than PCC and
whole-head average error in operator acceptance.

Same original BF16 inputs, hashes, heads, seeds, and
512 reference positions per head as qualification-v1. Common Q/K/V and constant
V are excluded as agreed. Unsupported FP32-streaming inputs are recorded, not
executed on a fallback. A fresh-seed holdout is additional to the frozen suite.
Thresholds remain 0.5% head L2, PCC >=0.99998, row p99 <=1%, and worst sampled
row <=2%, plus the original structural gates and undefined-PCC exception.
All full outputs are finite; original input hashes are unchanged. First-seed
trace checks compare entire outputs, not just reference samples. Raw records
are in `qualification.jsonl`; no failed result was rescored as a pass.

The additional holdout is **11/12 PASS**: H=5, fresh seeds 1239/1240,
N=32768/262144, normal/scaled-QK/outliers, all heads and 512 reference rows each.
All holdout heads satisfy 0.5% L2 (worst 0.440491%). One outlier case fails
the row maximum: N=262144, seed 1240, head 0 reaches **2.463148%**. This
independently confirms that the residual tail failure is not confined to the
original tuning/qualification seeds. Raw data: `holdout.jsonl`.

Together the frozen suite and holdout execute 182 cases, compare 1,335 heads
and 683,520 sampled query rows against FP64, and perform 40 full-output trace
equality checks. There are no missing/extra frozen-manifest IDs, duplicate
records, or executed fallbacks. These counts do not turn sampled worst-row
metrics into exhaustive long-context guarantees.

Three direct host-guard probes (N=2048, 25920, and 32769) all correctly reject
unsupported diagnostic streaming requests before fallback execution:
`guard-probe.jsonl`. Unsupported manifest entries are not device accuracy tests.

## Performance

All timing controls are complete. The benchmark uses B=1, H=10, Q=K=262144,
D=128, noncausal normal BF16 inputs, seed 1236, no Q preprocessing. All variants
use those same original inputs. Forty warmups and ten blocking trace replays
exclude compilation, transfers, and reference work. These are trace-replay
wall times, not newly collected raw device-cycle counters or FPU activity.

| Implementation | Median ms | Normal aggregate L2 % | PCC | Useful TFLOP/s |
|---|---:|---:|---:|---:|---:|
| Main BF16 streaming | 2790.163 | 18.647639 | 0.997511717 | 126.102 |
| Improved BF16 streaming, compensation enabled | 2840.444 | 3.159530 | 0.999711225 | 123.869 |
| Main FP32, non-streaming | 3323.778 | 8.794174 | 0.999399556 | 105.857 |
| HiFi4 FP32 streaming, full-FP32 subtraction + accurate exp | 9561.096 | 0.174016 | 0.999998485 | 36.800 |

Improved BF16 reduces this input's L2 by 5.90x for **1.80% time overhead**
(50.28 ms), preserving main's 128/512 chunks and input double buffering.
Main FP32 also uses 128/512; the HiFi4 streaming diagnostics use 128/1024
and the retained FP32 input-buffering layout. The accuracy-oriented diagnostic
is 2.88x main FP32's time. These are useful operational comparisons, not
isolated fidelity changes: the algorithms and FP32 chunk sizes differ.

The HiFi4 ablations separate fidelity from the conservative FP32/SFPU loop's
cost. All three use original Q, HiFi4 QK/PV/denominator, and FP32 state:

| Subtraction | Exp | Median ms | Normal aggregate L2 % | PCC | Useful TFLOP/s |
|---|---|---:|---:|---:|---:|
| Existing FPU/TF32 operands | Unbiased approximate | **4820.833** | 0.204591 | 0.999997934 | 72.984 |
| Full-FP32 SFPU | Unbiased approximate | **6327.669** | 0.179230 | 0.999998393 | 55.604 |
| Full-FP32 SFPU | Accurate SFPU | **9561.096** | 0.174016 | 0.999998485 | 36.800 |

Approximate exp with FP32 subtraction is 33.82% faster in elapsed time than
accurate exp, with a small normal-input L2 difference. The existing TF32
subtraction is faster still but adds a score-precision loss. These are measured
algorithm/scheduling controls, not a claim that HiFi4 intrinsically costs 9.56 s.
The naive one-score-tile-at-a-time FP32 subtraction schedule has optimization
room; no production-optimal latency is claimed.

The three HiFi4 configurations cost respectively 45.04%, 90.38%, and 187.66%
more time than main FP32. Replay ranges were 2783.859–2796.573 ms (main BF16),
2836.090–2852.233 ms (improved BF16), 3320.224–3329.523 ms (main FP32),
4808.619–4832.779 ms (HiFi4 TF32 subtraction), 6319.287–6337.089 ms
(HiFi4 FP32 subtraction/approximate exp), and 9560.636–9561.647 ms
(HiFi4 FP32 subtraction/accurate exp).

**Only the accurate-exp/FP32-subtraction variant received the expanded 170-case
qualification plus holdout.** The two approximate-exp variants received the
six-head ablations and this full-shape normal benchmark; do not assign them
the 162/170 pass rate. No extra performance acceptance threshold was introduced.

Useful throughput counts `4 * H * N^2 * D = 351.84372088832e12` attention
matmul FLOPs per invocation, divided by trace-replay time; it excludes softmax
and housekeeping instructions. This is not an FPU activity-counter percentage.
Raw timing and numerical records: `main-bf16.jsonl`, `improved-bf16.jsonl`,
`main-fp32.jsonl`, `hifi4-unbiased-tf32sub.jsonl`,
`hifi4-unbiased-fp32sub.jsonl`, and `accurate-hifi4.jsonl`.

## Engineering recommendation

Keep the two-tier direction, and make full HiFi4 part of the accurate tier.
Also preserve FP32 scores through subtraction; otherwise enabling FP32 DST
still hides a lossy operand conversion. Use original Q and remove the old
HiFi2 probability bias. Accurate exp is a useful trusted control, but exp
coefficient work is not the solution to the remaining outlier failures.

Do not sign off the accurate tier solely because normal inputs are near BF16
rounding error. Keep the failing stress cases and the standalone matmul repro.
Investigate the inner-product reduction/LLK behavior separately before calling
this an unconditional accuracy implementation. A reduction-precision workaround
must be measured against the accuracy/performance tradeoff; the diagnostic
eight-pass decomposition is not proposed for the PR.

For the next implementation step, qualify the unbiased approximate-exp variant
with full-FP32 subtraction across the same frozen suite before selecting it as
the accurate tier's exp implementation. Optimize the subtraction schedule while
retaining the accurate-exp control. Separately take the standalone main-matmul
repro to the LLK/FPU investigation: reducing that error has more leverage on the
remaining stress failures than additional exp fitting. The BF16 fast tier can
remain a separate engineering change, with its measured gains and known
qualification limitations disclosed rather than inheriting the accurate label.

No FA3/FA4 GPU execution was performed in this investigation. We found a TT-side
source that reproduces the failures, so a cross-vendor comparison would not
justify waiving those failures. Model captures, end-to-end evals, Galaxy,
causal/GQA/multibatch, other head dimensions, and unsupported streaming shapes
remain unqualified.

## Reproduction, provenance, and handoff

Blackhole P100A, yyzo-bh-26, reservation 216406, 110 compute cores. This is not
a Galaxy measurement. Main control is the original base
`2ba6fc2339d53300ae87c5202f335ef56492cfb3`, not newly fetched main.

`diagnostic.patch` is incremental from the retained four-file snapshot recorded
in `../qualification-v1/RESTORED-SHA256.txt`; `DIAGNOSTIC-SHA256.txt` records the
tested diagnostic sources. Modes are described in `PLAN.md`. The temporary host
factory selects the validated FP32 streaming geometry using a HiFi2 request,
then emits actual HiFi4 kernels; the JSON/experiment labels disclose this
override. It is not a production API, and each mode runs in a fresh process.

`full-diagnostic-from-main.patch` instead contains the complete four-file delta
from the main base. Use one patch appropriate to the starting source state,
not both. The current worktree is the retained snapshot, not the diagnostic.
The incremental patch passes `git apply --check` against that snapshot.

On the allocated machine, from the repository root with its existing Python
environment, a fresh diagnostic run can be reproduced as follows. Choose new
output names; qualification resumes existing completed records.

```bash
git apply --check experiments/sdpa-l2/accuracy-investigation/diagnostic.patch
git apply experiments/sdpa-l2/accuracy-investigation/diagnostic.patch
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
export TT_METAL_HOME="$PWD" ARCH_NAME=blackhole
python_env/bin/python experiments/sdpa-l2/accuracy-investigation/qualify_full.py --output /tmp/hifi4-qualification-new.jsonl
python_env/bin/python experiments/sdpa-l2/accuracy-investigation/qualify_full.py --holdout --output /tmp/hifi4-holdout-new.jsonl
TT_SDPA_ACCURACY_DIAG=3 bash experiments/sdpa-l2/accuracy-investigation/measure.sh hifi4-timing-new fp32_hifi2 1024
```

Qualification, residual, and guard scripts verify the diagnostic source hashes
before opening a device; diagnostic timing also checks the manifest. Hash checks
do not replace rebuilding the host library. Do not change the diagnostic mode
inside a process with cached programs. To reproduce the independent matmul
finding on main, no SDPA patch is needed: run `matmul_probe.py --output
/tmp/main-matmul-new.jsonl` with the same Python environment on the main build.

Host build succeeded with `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build
build_Release --target install`; device JIT and actual executions exercise the
modified headers. Initial compile-only failures are retained in logs; corrected
runs use the `-v2` names where applicable. Qualification was safely interrupted
during reference calculation to run residual diagnostics, then resumed from
complete records; its original and resumed logs are both retained.

All four operator source files are restored locally and remotely to the
retained hashes in `../qualification-v1/RESTORED-SHA256.txt`, preserving the
pre-investigation changes. The restored host build succeeds. Four regression
cases (BF16 fast and retained HiFi2 FP32 accurate at N=32768 and 262144,
H=10, seed 1234) match the frozen full-output hashes and their full trace outputs
exactly: `restored-regression.jsonl`, `restored-build.log`.
No diagnostic C++ change has been promoted into the retained implementation.

Main controls used all four original source files, verified by
`MAIN-SHA256.txt`, a forced host rebuild (`main-build.log`), and the fresh
kernel-cache directory `/localdev/cglagovich/sdpa-accuracy-main-cache.Rux3om`.
Python syntax compilation, Black formatting, shell syntax, and diff-whitespace
checks pass. No commits or external messages/PRs have been made.
