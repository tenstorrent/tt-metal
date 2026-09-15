# HiFi2 SDPA with rounded inputs and refined approximate exp

September 9, 2026. Base `2ba6fc2339d53300ae87c5202f335ef56492cfb3`.

**Subsequent performance qualification:** [NONCAUSAL_H10.md](NONCAUSAL_H10.md)
remeasures full non-causal H=10. The candidate remains accurate (0.4884% L2)
but is 64% slower than main FP32 and about 2x BF16 streaming. The small overhead
below is specific to the causal configuration, not the candidate generally.

## Outcome

**The measured full 256K causal workload reaches the 0.5% aggregate L2 goal
without switching the large QK/PV matmuls to HiFi4 or using accurate logit exp.**
Three seeds give 0.4930%, 0.4950% and 0.4899%, at about 1.638 s device execution versus
1.584 s for main (+3.4%). Simple CPU Q preprocessing adds about 43 ms, making
preprocessing plus execution roughly 6% slower than main's device execution.
This is not a complete end-to-end application timing: transfers and reference
computation are excluded from both device timings.

This is a narrow success, not a universal guarantee. Short-query results remain
around 0.50%, some above the threshold. Outlier inputs fail the goal. Per-row
p95 L2 is about 0.56% even in the passing normal full-prefill cases. The patch is
experimental and the margin is small.

## Setup and scope

- Reservation 214149, `yyzo-bh-26`, Blackhole P100A, firmware 19.12.0, KMD 2.9.0.
  These are **single-card Blackhole results, not Galaxy results**. No NVIDIA
  measurement was made.
- Remote checkout: `/localdev/cglagovich/tt-metal-blackhole-20260908` in
  `yyzo-bh-26-special-cglagovich-for-reservation-214149`.
- BF16 Q/K/V and output; FP32 destination; QK/PV HiFi2; both approximation flags
  true; Q chunk 128, K chunk 512, D=128. The factory selects the standard compute
  kernel for FP32 destination, **not `compute_streaming`**.
- Full causal: B=1, H=4, S=262144. The entire operation executes; FP64 reference
  normally samples the last 128 rows per head. Short diagnostic: H=1, Q=128,
  noncausal, varying KV length.
- Relative L2 is `100 * ||actual - reference|| / ||reference||`, not squared L2.
  The reference always uses the **original BF16 Q/K/V**, before preprocessing.
  Thus preprocessing error is included, not hidden by changing the reference.
- Trace timings exclude compilation, transfers and preprocessing. Two warmups
  precede five or seven measured blocking replays. Traced and ordinary outputs
  are checked for exact equality. Small differences are timing noise.

## Why HiFi2 needs asymmetric treatment

The ISA's phase table and the Blackhole LLK's phase increment show that HiFi2
executes phases 0 and 1. These are not two symmetric BF16 products:

| Operand | Phase 0 fraction bits | Phase 1 fraction bits | Effective coverage |
|---|---|---|---|
| SrcA | first 4, plus implicit leading bit | next 5 | 9 fraction bits |
| SrcB | first 6, plus implicit leading bit | same first 6 | 6 fraction bits |

For the normal finite inputs relevant here, the product is approximately
`A_9 * B_6`, with the usual partial-product/accumulator behavior. This is a
quantization model, not an exact software simulation of all FPU rounding.

The SDPA operand order places **K and V in SrcA**, and **Q and P in SrcB**.
BF16 K/V's seven fraction bits fit in SrcA's coverage. Q loses its last BF16
fraction bit; P loses all bits below its sixth fraction bit. FP32 destination
does not restore those missing products. A Float32 CB also normally enters
through TF32 unless explicitly configured for full FP32 unpack-to-destination.

Primary documentation: [Matrix Unit phase table](https://github.com/tenstorrent/tt-isa-documentation/blob/5287a62727350bcef35f7b411d1b8a706172ec4c/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md).
The Blackhole documentation points to the shared matrix-unit description;
the phase behavior was cross-checked against this checkout's Blackhole LLK
and a local simulator implementation. The documentation checkout is pinned to
`5287a62727350bcef35f7b411d1b8a706172ec4c`.

### Q preprocessing

Simply rounding BF16 Q to six fraction bits with ties-to-even is not optimal:
half of BF16 values lie exactly on the coarser grid's midpoint. Instead use

```
Q_device = RNE_6(c * Q_original)
attention_scale = 1 / (sqrt(D) * c)
c = 1.0027
```

The slight scale shift breaks these ties. For finite normal BF16 and
`1 < c < 1 + 1/256`, the rounded device value is implemented by the especially
simple bit operation below; signed zero is preserved:

```
Q_device_bits16 = (Q_original_bits16 + 1) & ~1
```

Odd low fraction bits round away from zero, even ones remain unchanged. The
attention-scale compensation shifts this effective quantization grid back
towards the original values. This is not ordinary unscaled round-away-from-zero.
The repro exhaustively checks equivalence for all finite normal BF16 bit patterns
and signed zero at four scales, including 1.0027. The shortcut is intentionally
limited to the repro's normal/non-subnormal inputs; it is not a general NaN,
subnormal or overflow-handling API.

On the CPU quantization model, Q's relative error falls from about 0.404% with
plain RNE6 to 0.293% with the shifted grid. At 4K, including P rounding to six
bits and BF16 output, the model predicts about 0.474% SDPA L2. This leaves little
room for additional hardware error. The CPU model uses ideal matmuls and is not
a cycle- or bit-accurate Tensix simulator.

The float32 implementation took 749 ms for full H=4/S=256K Q. The equivalent
BF16 integer implementation took 42–43 ms on this host. A future fused producer
could avoid the separate pass; that performance has not been measured.

### P rounding and exp

Round each exponential weight to six fraction bits **before both the denominator
sum and PV**. Otherwise the denominator sums precision which HiFi2 PV discards,
creating a systematic numerator/denominator mismatch.

The original fast-exp construction is piecewise linear within an exponent
interval. Adjusting only its slope/bias does not remove that curvature error.
The candidate keeps the fast macro but uses a 10-bit grid, then a cubic
approximation to `2^(m-1)` for normalized mantissa `m` in `[1,2]`:

```
p(m) = 0.07901999*m^3 - 0.01293332*m^2 + 0.48564498*m + 0.44808037
```

The fitted cubic's maximum relative error on the fitting interval is about
0.0188%. A common 1.0002 coefficient factor prevents `setexp` from wrapping a
mantissa slightly below 1; the common scale cancels in softmax normalization.
The exponent bias is restored after refinement, then weights are rounded to six
fraction bits. This is still an approximate logit exponential.

Important range limitation: fitting the 10-bit grid in the macro's signed-magnitude
INT16 encoding zeros scaled logits below about -21.45 via packer ReLU. This was
adequate for the measured normal cases, but requires more range/adversarial
validation before production use. It is not a general-purpose exp replacement.

## Changes beyond the previous hybrid

The previous [hybrid report](../optimization/REPORT.md) describes the FP32 L1
state, compensated numerator update, accurate *small max-change correction*,
and refined reciprocal retained here. This turn adds:

1. Refined fast logit exp and six-fraction-bit P rounding, gated to Blackhole,
   BF16 Q/K/V, FP32 destination, HiFi2 and enabled approximation flags.
2. The optional BF16 Q preprocessing and compensated attention scale in the repro.
3. **HiFi4 for the one final denominator reduction**, not QK or PV. Its SrcB
   sums otherwise lose precision again under HiFi2. In an accurate-exp ablation,
   this alone lowered 256K L2 from 0.724% to 0.581% with plain Q RNE6.
4. SFPU final normalization, using full FP32 reciprocal unpack and ordinary TF32
   numerator ingress. This is once per query chunk, not once per K chunk.

“HiFi2” here refers to the two large matmuls. Small correction operations and
the final denominator reduction use higher precision. This is **not a claim
that every instruction uses HiFi2**, nor that every intermediate is full FP32.
BF16-destination streaming is outside this new path.

## Measurements

### Full causal 256K, H=4, D=128

| Candidate / seed | Tail-row L2 % | Device ms | CPU Q preprocessing ms |
|---|---:|---:|---:|
| Main / 1234, previous matched baseline | 8.7205 | 1583.97 | none |
| Previous hybrid / 1234 | 1.9527 | 1585.25 | none |
| Refined exp + shifted Q + SFPU final / 1234 | 0.49296 | 1637.83 | 748.81, old float implementation |
| Same / 1235, bitwise Q | 0.49504 | 1637.92 | 43.30 |
| Rebuilt scoped candidate / 1236, bitwise Q | 0.48987 | 1637.82 | 42.44 |
| Omit SFPU final / 1234 | 0.49736 | 1636.60 | 42.04 |
| Omit SFPU final / 1235 | 0.49937 | 1636.63 | 42.09 |

The smaller variant passes these two aggregate checks, but keeping SFPU final
normalization costs only about 1.2 ms (0.07% of operation time) and provides
useful margin. It is retained in the candidate. This is evidence for a small
sufficient change set, not proof of global minimality.

Raw files: `fast10-full-causal.jsonl`, `fast10-full-heldout.jsonl`,
`fast10-full-fpu-final.jsonl`, `fast10-full-fpu-heldout.jsonl`,
`final-full-seed1236.jsonl`.
The earlier baseline and hybrid are in `../optimization/base-causal.jsonl`
and `../optimization/final-causal.jsonl`.

A third seed with rows spread across the full context gives row-median L2
0.4870% and row-p95 0.5596%. Its aggregate L2 is only 0.0421%, because early
causal rows have much larger output norms, including the exact one-key first
row. **Do not treat this artificially favorable aggregate as the 256K result.**
Use the tail-row measurements above. Source: `fast10-full-spread.jsonl`.

### Diagnostic ablations, short Q=128

These are separate experiments; the changes listed must not be conflated with
a single fully matched sweep.

| Configuration, seed 1234, K=256K | L2 % | Device ms |
|---|---:|---:|
| Accurate logit exp, P6, plain Q RNE6, old denominator reduction | 0.72445 | 52.02 |
| Same, denominator HiFi4 | 0.58148 | 51.89 |
| Same, shifted Q c=1.0027 | 0.49759 | 51.87 |
| Fast 8-bit grid + quadratic, shifted Q c=1.0027 | 0.52375 | 27.53 |
| Fast 9-bit grid + cubic, shifted Q c=1.0027 | 0.50732 | 29.09 |
| Fast 10-bit grid + cubic, c=1.002, FPU final | 0.50044 | 28.28 |
| Same, explicit QK rounding to TF32 before pack | 0.51038 | 34.48 |
| Direct cubic exp without the fast macro, c=1.002 | 0.50398 | 35.00 |
| Retained fast10 + SFPU final, c=1.0027 | 0.50009 | 28.89 |

A matched Q-preprocessing comparison with the final arithmetic gives 0.67090%
without preprocessing, 0.57714% with plain RNE6, and 0.50009% with shifted RNE6.
All retain P6 and the same large HiFi2 matmuls. Sources:
`final-no-q-preprocess.jsonl`, `final-plain-q-rne6.jsonl`,
`fast10-sfpu-q10027.jsonl`. This isolates the benefit of the input-grid shift
from the other changes.

Explicit QK rounding and direct cubic exp were rejected: neither offered a
useful accuracy/performance tradeoff. Plain RNE6 Q can actually worsen error
until the denominator truncation is fixed. This is why only looking at PCC,
or treating HiFi2 as symmetric BF16 arithmetic, is misleading.

Short-query runtime is about 1.9x main's 15.19 ms, unlike the +3.4% full-prefill
result. The extra SFPU work is much more visible in this diagnostic workload.
Do not advertise the full-prefill overhead as shape-independent.

### Outliers and remaining limitations

With final arithmetic and c=1.0027, short-query outlier L2 is 0.7756%, 0.4919%,
and 0.5443% at 4K, 32K and 256K. At 256K, row median/p95 are 1.61%/2.83%.
The earlier hybrid's aggregate errors were 1.8548%, 0.8676% and 1.1096%.
Source: `fast10-outliers.jsonl`. Larger logit sensitivity makes Q quantization
more consequential; range truncation and state corrections also need isolated
outlier ablations before assigning all remaining error to one source.

Additional limitations: synthetic inputs, sampled reference rows rather than
all 256K rows, one card, D=128/K chunk=512 emphasized, no model-level quality
test, no Galaxy validation, and no complete feature-regression suite.

## Reproduce and verify

From the allocated checkout:

```bash
export TT_METAL_HOME=$PWD
export ARCH_NAME=blackhole
export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --kv-lens 262144 --causal --heads 4 --variants fp32_hifi2 \
  --q-round-bits 6 --q-prescale 1.0027 --q-bitceil \
  --seed 1235 --benchmark-iters 7 --max-l2-pct 0.5 \
  --label refined-hifi2 --output experiments/sdpa-l2/hifi2-rounding/repro.jsonl
```

The repro self-checks its FP64 online reference against dense softmax, the
PCC-versus-scale-error metric, and exhaustive normal-BF16 preprocessing
equivalence. It checks finite results, trace equality, and the requested L2
threshold. CPU quantization-only analysis: `cpu_quantization.py`.

Host build passed with
`CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install`;
see `final-build.log`. Kernel JIT and on-device cases were also exercised.
Formatting used Black 23.10.1 and git-clang-format/clang-format 19.1.4.
The final source snapshot is `final-approx-hifi2.patch` (tracked C++ changes
relative to the base commit); the repro and CPU model are separate files.
The post-build full seed-1236 run passed `--max-l2-pct 0.5`.
Post-build constant-V checks at 4K/32K/256K returned ones (L2 below 2e-13%,
the FP64 reference's numerical residual). BF16-destination streaming smoke
checks also passed execution/finite-output checks at 4K/32K; their L2 remains
2.5081%/2.8364%, not a claim of meeting the FP32 path's new target. Sources:
`final-constant-v.jsonl`, `final-streaming-smoke.jsonl`.

Invalid exploratory runs: `fast-refine-q6offset.jsonl` and
`fast-refine-init.jsonl` had an SFPU address-modifier error;
`fast9-cubic.jsonl` had polynomial mantissa wrapping. They are retained as raw
history but excluded from all accuracy claims. Failed import/compile logs also
remain as history; later successful runs replaced their empty result files.
