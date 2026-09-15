# Accurate FP32 SDPA: utilization optimization, 2026-09-11

## Scope and target

Continue from the accepted **mode 4** algorithm: HiFi4 QK/PV, FP32 destination,
full-FP32 score subtraction, unbiased approximate exp, FP32 numerator/sum state,
original BF16 inputs and BF16 output. The maximum CB is BF16, not FP32.
This is a performance optimization of that algorithm, not a fresh accuracy claim.

Measurements use reservation 216406, `yyzo-bh-26`, one Blackhole P100A,
110 compute cores (11x10). This is not a Galaxy measurement. Batch 1,
noncausal, D128, Q/K chunks 128/1024; input buffering and movement are unchanged.
The original FP32 configuration already single-buffers K/V at K1024.

Useful attention FLOPs are `4 * B * H * N^2 * D`, excluding softmax and
denominator work. The user's approximate 34% estimate at 55.65 TFLOP/s implies
an 80% target near **131 TFLOP/s / 2.69 seconds at H10, N262144**.
That nominal peak denominator is not independently calibrated. FPU activity
counters and useful attention FLOP utilization are different metrics.

## Retained candidate

All changes are behind investigation guards; normal defaults are preserved.
Source [candidate-env.sh](candidate-env.sh) before a fresh Python process.

1. Process two score tiles together, sharing one maximum tile in FP32 DST.
2. Fuse subtraction and the existing exp grid/polynomial instruction sequence,
   preserving the separate FP32 subtraction rounding step.
3. Initialize coefficients and replay body once per K chunk; skip unused
   fast-exp macro setup. Temporarily use programmable LREG11 for the grid bias,
   restoring SFPI's reserved -1 value before returning.
4. Coalesce score unpack and pack handshakes across both tiles. Preserve
   Blackhole's extra zero-flag clear and all producer/consumer barriers.
5. Use denominator phases **0+2**, not ordinary HiFi2's 0+1. Its SrcA operand is
   zero/one, so the low-SrcA products in phases 1 and 3 vanish. QK/PV remain
   full HiFi4. Complete BF16 output equality is required in paired checks.

No Q preprocessing, new scratch CB, Q/K chunk change, or input-buffer change.

## Validation methodology

`validate.py --label final-v1 --qk-width 4` compares the original mode 4
baseline with the combined candidate, in fresh processes. Performance runs
use 40 warmups and 10 blocking trace replays at N32768/131072/262144, H10,
seed1236. Reference calculations use FP64 attention on 512 spread query rows
per head with full K/V. Timing excludes reference calculations and hashing.
Reported L2/PCC are sampled-reference metrics, not full-output FP64 comparisons.

Separately, ten distributions at N32768/H10/seed1236 and three holdouts at
N65536/H5/seed1237 check full-output finiteness, trace equality, and exact BF16
output SHA256 equality between baseline and candidate. Their one-replay,
zero-warmup timings are **not** used for performance claims. This preserves
existing accuracy, including existing failure cases; it is not a replacement
for the wider qualification suite.

### Sustained paired results

| N | Original mode4 ms | Candidate ms | Less time | Candidate useful TFLOP/s | L2 % (both) | PCC (both) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 94.204 | 81.104 | 13.91% | 67.78 | 0.179594 | 0.999998386 |
| 131,072 | 1551.946 | 1292.383 | 16.72% | 68.06 | 0.179231 | 0.999998394 |
| 262,144 | 6208.460 | 5157.544 | 16.93% | 68.22 | 0.179230 | 0.999998393 |

All three complete BF16 outputs match their baseline hashes exactly; all
trace replays reproduce the corresponding complete nontrace output.
These comparisons use this session's matched controls, not older timings.
At 256K the speedup is 1.204x. On the provisional 55.65/0.34 TFLOP/s scale,
the candidate is approximately 41.7% utilized. **The 80% goal is not met:**
another approximately 1.92x speedup / 48% time reduction is required.

All **16 paired cases** passed exact full-output equality, trace equality,
finiteness, and unchanged sampled L2/PCC checks (3 sustained + 10 stress +
3 holdouts). Raw paired JSONL/logs are the authority. The normal32K case occurs
in both timing and stress sections, so these are 16 runs, not 16 unique inputs.

| Stress case, N32768 H10 | L2 % (both versions) | PCC (both versions) |
| --- | ---: | ---: |
| Normal | 0.179594 | 0.999998386 |
| Q/K scaled by 2 | 0.197226 | 0.999998057 |
| Sparse outliers | 0.259614 | 0.999996632 |
| Uniform attention (Q=0) | 0.173736 | 0.999998488 |
| Constant V=1 | ~0 (FP64 reference residual) | Undefined |
| Q=0, V=1 | 0 | Undefined |
| V bias +1 | 0.176753 | 0.981648634 |
| Common Q +32 | 0.232221 | 0.999997304 |
| Common K +32 | **0.731996** | 0.999973698 |
| Common V +32 | 0.028662 | Undefined |

The H5/seed1237/N65536 holdouts have L2 0.179143% normal, 0.198372%
scaled-QK, and 0.279635% outliers, unchanged in each pair. PCC is undefined
when a compared output has zero variance; no artificial PCC=1 is assigned.
The common-K result remains above 0.5%. This optimization neither fixes nor
worsens that existing mode4 limitation. No claim of universal accuracy or
full qualification is made.

## Screened alternatives

Short screens used 10 warmups/5 replays, H10 normal seed1236, N32K/128K.
Every successful screen retained complete baseline BF16 output equality.
These are ablations, not the sustained final comparison.

| Configuration | 32K ms | 128K ms |
| --- | ---: | ---: |
| Original mode 4 | 93.32 | 1542.81 |
| Two-score batching | 82.90 | 1419.48 |
| Above + fused subtraction/exp | 80.80 | 1383.21 |
| Above + coefficient/replay reuse | 80.17 | 1362.21 |
| Above + extra programmable constant | 78.98 | 1340.58 |
| Above + paired unpack | 77.28 | 1324.22 |
| Above + paired pack / skip unused init | 76.83 | 1316.90 |

The exact 0+2 denominator was independently beneficial (1341.99 ms versus
1362.21 ms for the coefficient-reuse configuration at 128K) and is included
in the final combined candidate. BF16 maximum broadcast caching, an extra
16 KiB FP32 maximum scratch CB, and internal QK 1x2/2x2 schedules were correct
but slower; they are disabled. The first replay-reuse attempt hung due to
swapped REPLAY execute/load flags. The process was stopped and the allocated
card reset; the corrected version passed subsequent controls. Failed logs
remain available.

## Bottleneck evidence

An L1-resident matching-primitive calibration measured **140.21 TFLOP/s QK**
and **140.89 TFLOP/s PV**, HiFi4/FP32 DST, 110 cores, with output verification.
This uses 1x4 no-MOP matmul plus pack, repeated over resident inputs. It is
neither an SDPA result nor a theoretical peak, and its thermal workload differs
from a sustained 256K operator. It indicates significant overhead outside the
matmul primitive, not a guarantee of 131 TFLOP/s attention.

Stage profiles use N65536 to avoid 32-bit counter wrap. LLK CB wait/reserve
instrumentation also uses sum slots: discard UNPACK slot0 and PACK slot1.
`stages.py` retains only uncontaminated thread/slot pairs. Inclusive per-thread
times overlap and must not be added into an exclusive critical-path budget.
Production performance comparisons have profiling disabled.

Final-candidate stage1 counters: **55.61% FPU active, 42.03% SFPU active,
19.84% both active, 22.20% neither active**, averaged over 110 cores with no
reference-counter wrap. Stage2 gives 55.07%, 41.62%, 19.63%, and 22.94%,
respectively. Do not substitute these activity percentages for the 41.7%
provisional useful-FLOP utilization estimate above.

Uncontaminated inclusive stage sums for the final candidate:

| Thread | Stage | Mean share of that thread's kernel interval |
| --- | --- | ---: |
| MATH | QK | 33.91% |
| MATH | PV | 28.33% |
| MATH | score subtraction/exp | 20.51% |
| MATH | denominator | 3.91% |
| PACK | score subtraction/exp | 55.29% |
| UNPACK | PV | 18.80% |
| UNPACK | denominator | 4.25% |

Register-resident fused subtraction/exp calibration (`exp-v2.jsonl`) achieves
372.34 billion logits/second over 110 cores. Applying that rate to H10/256K's
logit count gives **1.846 seconds of arithmetic-only work**. The test excludes
recurring reload/pack, matmuls, reductions, and online state work. It repeatedly
processes negative logits against a fixed maximum, checks the expected fixed
point including the exp grid's common normalization factor, and checks all
cores and complete trace equality. This is an instruction-throughput probe,
not SDPA performance or a universal lower bound. `exp-v1.log` retains the
initial failed check that incorrectly omitted the grid's common scale factor.

## What remains to reach 80%

The next substantial change is an **explicit asynchronous DST-half pipeline**,
not another chunk-size sweep or arithmetic-precision reduction:

1. Overlap QK on one DST half with subtraction/exp and pack on the other,
   with explicit ownership and visibility tokens. Today the synchronous
   per-pair reload/reconfigure helpers leave substantial MATH wait time.
   Prototype a steady-state independent-subblock microkernel first; require
   an improvement over the current combined stage, not just fewer instructions.
2. Carry that schedule through the last-row/PV drain, preserving the existing
   partial-product grouping and online update ordering. Keep the existing Q/K
   chunk geometry and input buffers. Check full-output equality after each
   scheduling change before any sustained timing.
3. Reduce SFPU issue overhead further with Blackhole load macros and retained
   coefficient/address setup, while preserving the subtraction and polynomial
   rounding sequence. Load macros require explicit dependency spacing; raw
   instruction-count reductions alone do not prove a safe or faster schedule.
4. Revisit FPU bubbles and normalization work after those changes. The current
   denominator is already down to about 4% of inclusive MATH-thread time;
   focusing only there cannot supply the remaining 48% time reduction.

The target is aggressive. At the profiled shape, eliminating all neither-unit
time and hiding all SFPU-only time would still leave approximately 55.6% of
the original interval if FPU-active time stayed fixed: only about 1.80x.
This is a conditional scheduling model, not a hardware bound. The requested
additional 1.92x therefore likely also needs better FPU issue efficiency or
less auxiliary FPU work, not just overlap.

Similarly, the isolated QK/PV rates imply roughly 2.50 seconds for the two
matrix products at 256K, before denominator and state work. They omit the
split-drain's short PV groups and recurring operator traffic. A 2.69-second
operator would leave very little unhidden overhead. The 1.85-second SFPU
probe does not rule the target out, but no result yet demonstrates that
these stages can overlap sufficiently. **80% is still an experimental target,
not a promised outcome.**

## Reproduction and provenance

Base commit: `2ba6fc2339d53300ae87c5202f335ef56492cfb3`, with the retained
preexisting SDPA changes. `candidate.patch` is relative to that retained
working state (`../qualification-v1/RESTORED-SHA256.txt`), **not pristine main**.
It includes investigation/ablation controls and is not a minimal PR patch.
`SOURCE-SHA256.txt` records the four final C++ source files. Per-case metadata
also records source hashes and all option values. Public `fp32_hifi2` naming
is historical: mode4 explicitly overrides QK/PV to HiFi4 and records the override.

Build on the allocated container:

```bash
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
python_env/bin/python experiments/sdpa-l2/fp32-util-v1/validate.py \
    --label fresh-label --qk-width 4
```

Use the repository's established Blackhole environment (`TT_METAL_HOME`,
`ARCH_NAME=blackhole`, `PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages`).
The harness refuses existing result labels. Final build log: `final-build.log`.
