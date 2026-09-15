# Whole-KV-block permutation diagnostic

## Initial hardware result

[block-permutation-v1.jsonl](block-permutation-v1.jsonl) contains 24 completed,
accuracy-only cases: N4096/32768, H2/D128, 22 requested cores, Q256/K512,
seed1240; two distributions, two variants and three orders. Full-compensated
LoFi BF16 and LoFi FP32-state both use Q7 and RNE5 K/V in native BFP8 storage,
with native approximate score exp. Their existing input-slot layouts and
kernels are unchanged. No performance measurement was performed.

Every permutation jointly moves entire 512-token K/V blocks. This preserves
token pairing, intra-block reduction order and native shared-exponent groups.
Q is unchanged. `k_rms_descending` sorts block RMS over both heads and all
tokens/features; it is **not** query-dependent score-max sorting. The second
distribution multiplies successive K blocks by repeating [0.5, 1, 2, 4],
leaving V unchanged.

All 24 rows pass preprocessing, original/ordered CPU input and original device
input immutability, all-output finiteness, two bitwise combined trace replays,
and unchanged-source gates; the ledger records 82 principal source pins.
Preprocessing checks require exact represented nonzero values and permit only
signed-zero equivalence. There are 1,063,488 recorded signed-zero disagreements
across checks, so this must not be called bit-exact preprocessing. No nonzero
mismatch is accepted. Original-input FP64 references agree across permutation
to `rtol=1e-11, atol=1e-12`; the largest observed absolute difference is
1.9984e-14.

### Accuracy and order sensitivity

All cells are generated from the JSON ledger, which remains authoritative.
L2 against FP64 samples 128 explicit Q rows per head and includes all KV.
**Inter-order L2 instead checks every device output element**, relative to the
same variant's identity-order device output. These two metrics have different
references and scopes and must not be subtracted as an error decomposition.

| N | Input | State | Identity L2 % | Reverse L2 % | RMS-sort L2 % | Reverse vs identity % | RMS-sort vs identity % |
|---:|---|---|---:|---:|---:|---:|---:|
| 4,096 | normal | FP32 | 2.879 | 2.877 | 2.887 | 2.370 | 2.182 |
| 4,096 | normal | BF16 compensated | 3.067 | 3.085 | 3.077 | 2.605 | 2.404 |
| 4,096 | block-scaled K | FP32 | 4.137 | 4.137 | 4.137 | 0.857 | 0.046 |
| 4,096 | block-scaled K | BF16 compensated | 4.731 | 4.685 | 4.728 | 1.029 | 0.097 |
| 32,768 | normal | FP32 | 2.820 | 2.804 | 2.819 | 2.551 | 2.162 |
| 32,768 | normal | BF16 compensated | 3.156 | 3.138 | 3.145 | 2.811 | 2.392 |
| 32,768 | block-scaled K | FP32 | 4.512 | 4.507 | 4.526 | 1.225 | 0.915 |
| 32,768 | block-scaled K | BF16 compensated | 5.323 | 5.257 | 5.239 | 1.456 | 1.117 |

Normal identity/reverse PCC respectively: FP32 0.999587/0.999587 at 4K and
0.999602/0.999607 at 32K; compensated BF16 0.999532/0.999527 at 4K and
0.999537/0.999541 at 32K. Nearly unchanged aggregate L2/PCC can conceal
substantial changes to the actual output vector.

This is a real operator invariance stress, not an expectation of bitwise
associativity from floating-point arithmetic. It does not establish that the
observed magnitude affects any particular model's quality. Scaled blocks have
higher original-reference error but **lower** order sensitivity here; neither
metric substitutes for the other, and sorting RMS is not a general repair.

## Source-level candidate mechanism: exp is not translation-consistent

The private [streaming header](streaming/compute_streaming.hpp) folds the
previous row maximum into the new maximum before score exp. It computes
current weights from score minus that running maximum. Its
`sub_exp_first_col_blocks` computes previous-minus-current maximum and calls
`calculate_sdpa_exp_correction` for old numerator/denominator rescaling.
That helper calls `_sfpu_exp_fp32_accurate_` with the full FP32 attention scale.
The compensated BF16 frozen helper selects the same accurate correction
entrypoint, although its storage/arithmetic roundings differ.

By contrast, [exp_native.hpp](exp_native.hpp) keeps the native score-exp
LOADMACRO and omits the mantissa refiner. The public Blackhole exp
implementation uses `A=256*log2(e)*scale`, `B=32500.818359375`, integer
conversion and shift 15. It is a piecewise-linear approximation in the encoded
floating-point exponent/mantissa, not an exact exponential. The
[LUT helper](exp_lut.hpp) refines this same grid; it is therefore a useful
controlled intervention.

Let scores and maxima below already include the attention scale. Write the
score exponential as `F(x)=exp(x)*(1+epsilon(x))` and the rescale helper as
`G(x)=exp(x)*(1+gamma(x))`. A token inserted with maximum m1 and subsequently
rescaled to m2 receives weight proportional to:

```text
F(s-m1) * G(m1-m2),
```

whereas inserting it directly at m2 gives `F(s-m2)`. Their ratio is:

```text
[1+epsilon(s-m1)] * [1+gamma(m1-m2)] / [1+epsilon(s-m2)].
```

Even exact G and exact accumulation do not make this ratio one unless the
relative error of F is translation-independent for the relevant shifts.
A constant common exp bias would cancel in normalized attention; the
phase-dependent ripple is the issue. Represented-P rounding and the BF16
denominator/PV precision mismatch can add further violations, but are not
necessary for this mathematical failure of invariance.

### Scale of the native approximation, not a device-error bound

Ignoring integer-grid, score subtraction, P quantization, underflow and FPU
rounding, put `c=(32512-B)/256=0.04367828369140625` and
`t=x/log(2)-c`. The native linear float encoding gives:

```text
F(x) = 2^floor(t) * (1+u),  u=frac(t)
F(x)/exp(x) = 2^(-c) * (1+u) / 2^u.
```

The continuous ratio ranges from 0.970178 to 1.029821; its maximum occurs at
`u=1/log(2)-1`. Consequently two different maximum phases can change an
unnormalized token weight by a ratio as large as 1.061476 (about 6.15%),
even if rescale exp is exact. Integer log-grid spacing 1/256 and subsequent
7-bit P consumption add quantization not included in this envelope.

This is **not** a 6.15% attention-L2 bound. With one order's normalized weights
p and output O, perturbing each unnormalized weight by `1+e_i`,
`|e_i|<=eta<1`, gives the conditional absolute-error bound:

```text
||O_new-O|| <= eta/(1-eta) * sum_i p_i * ||V_i-O||.
```

Relative output error additionally divides by `||O||`, which can be small
through cancellation. Neither the unnormalized ripple nor a scalar exp
accuracy claim determines a universal operator-relative-L2 guarantee.

With exact G, multiple max-update factors telescope to `exp(m_insert-m_final)`;
the token retains one insertion-phase epsilon rather than multiplying one
native-F error per later block. Thus order sensitivity need not grow linearly
with context. Approximate G, BF16 spills and recurrence roundings may add
max-update-dependent effects. This explains why stable error bands at 4K/32K
do not rule out max-history dependence, without proving its device contribution.

## What distinguishes this from accumulation rounding

Whole-block permutation changes the order of cross-block accumulation and
running maxima but leaves each block's K/V quantization and internal dot
products intact. FP32 state does not make FPU product alignment or summation
exact, so its substantial order sensitivity alone is not causal proof against
accumulation. Conversely, FP32 accumulation cannot restore exp translation
consistency; a hardware accumulation explanation is not required by the math.

The controlled prediction was that refining F on the **same native grid**,
with the same LoFi QK/PV and FP32 recurrence, should materially reduce
inter-order L2 if exp phase ripple is a major contributor. Native versus
macro-LUT is the strongest comparison because it retains the base
grid and correction path. A cubic control is also informative, but its wider
grid/underflow behavior must be disclosed rather than calling it a
polynomial-only change. Similar residual sensitivity after refinement would
shift attention toward score/max subtraction, P consumption, rescaling and
FPU accumulation. A K-RMS ordering is not equivalent to fixing the actual
query-dependent final maximum.

## Completed exp-order ablation

[block-permutation-exp-v1.jsonl](block-permutation-exp-v1.jsonl) contains
18 completed cases: normal inputs only, both lengths, three orders, and three
exp implementations. All use LoFi QK/PV, Q7 and identical represented K/V5 in
BFP8, FP32 numerator/denominator/P storage, BF16 row maxima and BF16 output.
Q256/K512, one KV slot, original inputs and the accurate online-rescale helper
are retained. This is an accuracy-only experiment, not a new speed result.

Native and macro-LUT use the **same builder**, differing only by the two LUT
enable defines. The LUT refines the same native grid. The cubic control uses
the existing native-storage builder with `native_exp=false`: it changes the
exp family, including a wider log2 grid (8→10 fraction bits), polynomial,
effective-P bias treatment and underflow range. It is **not** a
polynomial-degree-only intervention. Native “8-bit grid” describes its log2
fraction resolution (nine significant output bits before later consumption),
not an FP8 or BFP8 datatype claim.

### Independent evidence audit

- The ledger contains all 18 expected result keys and a complete marker, with
  81 principal source hashes checked unchanged during the run. The prior native
  evidence file's SHA256 matches the recorded hash, and the completion marker
  confirms that evidence remained unchanged.
- All original/ordered input hashes, exact represented preparation hashes,
  block permutations and sampled reference rows match the original native
  experiment for the corresponding length/order. All six native outputs
  reproduce the old native driver's **exact output hashes**. All four recorded
  cross-driver checks pass: inputs, prepared bits, CB bytes and outputs.
- Every row passes exact nonzero preparation, all-output finiteness, two
  bitwise combined trace replays and before/after CPU/device immutability.
  Signed-zero equivalence remains explicit: 797,616 zero-sign disagreements
  across preparation checks, no nonzero mismatches.
- Native/LUT per-CB audit records match. Independently evaluating the two
  builders' selected CB specification expressions gives identical formats,
  page sizes and capacities for cubic as well: **1,212,416 bytes/core** and
  one KV slot. Cubic records total CB bytes rather than a per-CB audit; the
  finer comparison is source-verified, not an invented logged gate.
- FP64 permutation-reference agreement passes in all rows; maximum absolute
  discrepancy is 6.2450e-16. Accuracy still samples 128 Q rows/head, whereas
  inter-order output difference covers every element. No timing was performed.

The cubic builder spells the matched-denominator flag `SDPA_MATCH_HIFI2`,
where native/LUT use `SDPA_LOFI_DENOM`; the selected streaming header maps
both to the same LoFi denominator branch. This is not a denominator-fidelity
change despite the flag names.

### Measured results

| N | Exp family | Identity L2 % | Reverse L2 % | RMS-sort L2 % | Reverse vs identity % | RMS-sort vs identity % | Identity PCC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4,096 | Native | 2.879 | 2.877 | 2.887 | 2.370 | 2.182 | 0.999587 |
| 4,096 | Native + macro-LUT | 2.227 | 2.227 | 2.224 | 0.791 | 0.731 | 0.999752 |
| 4,096 | Wider-grid cubic | 2.170 | 2.174 | 2.172 | 0.484 | 0.454 | 0.999765 |
| 32,768 | Native | 2.820 | 2.804 | 2.819 | 2.551 | 2.162 | 0.999602 |
| 32,768 | Native + macro-LUT | 2.198 | 2.196 | 2.202 | 0.866 | 0.745 | 0.999758 |
| 32,768 | Wider-grid cubic | 2.142 | 2.142 | 2.145 | 0.522 | 0.457 | 0.999771 |

The same-grid LUT reduces reverse-order output difference by **66.6% at 4K
and 66.0% at 32K**. The broader cubic-exp change reduces it by about 79.6%
at both lengths. Original-reference L2 also improves, but less dramatically:
32K identity 2.820→2.198→2.142%. These percentages compare observed norms;
they are not an orthogonal decomposition of error or a statement that exactly
66% of baseline error was caused by one primitive.

This controlled result establishes that the exp-refinement choice materially
affects permutation sensitivity and strongly supports the max-history/
nonmultiplicativity mechanism above. It does not prove the analytic continuous
ripple is the sole contributor: the changed P values also change downstream
rounding. The result is stronger than inferring a cause merely from the
presence of order sensitivity in FP32 state.

The remaining 0.48–0.52% cubic reverse-order discrepancy must **not** be
labeled pure FP32 accumulation error. Quantized P, BF16 maxima, score/max
subtraction, rescale evaluation, FPU product alignment and final BF16 output
rounding remain. This suite covers one normal seed, not common modes,
block-scaled K, broader distributions or real-model quality.

Engineering implication: permutation sensitivity is a useful additional
qualification metric, and cheap-exp selection should consider online
translation consistency as well as standalone exp MSE and aggregate L2/PCC.
The accuracy/performance cost of refinement must use the separate qualified
timing records; this experiment does not authorize a dispatch change.
Numerical work is frozen. No additional kernels, device jobs or heavy CPU
model were created for this report.

## Independent idealized CPU mechanism check

The subsequently authorized [36-case CPU study](EXP_ORDER_CPU_MECHANISM.md)
isolates the continuous native exp shape with otherwise FP64 scores, state,
PV and exact exp rescaling. At 32K its running-max reverse-order discrepancy
is 2.576%, while a fixed-global-max schedule reduces it to 4.99e-14%.
Exact and constant-bias exp controls remain order-invariant to FP64 roundoff.
Fixed maxima do not remove exp approximation error: original-reference L2
still measures 1.827% in that global-max surrogate.

This establishes the proposed mechanism **within the isolated model** and
complements the hardware LUT intervention. H1/64-query CPU inputs differ from
the H2 device suite; no Q7/BFP quantization, native integer grid, BF16 rowmax
spill or FPU-alignment model is included. The CPU percentages are not a
quantitative attribution of device error, and the global maximum requires an
oracle/prepass whose implementation cost is not free or measured. No existing
numerical producer was changed.

A further [same-input CPU lattice-max study](EXP_ORDER_LATTICE_CPU.md) removes
the continuous surrogate's order dependence without a global prepass, but
retains about 1.8% original-reference L2 and introduces 2.6–3.5% output changes
under mathematically irrelevant common score offsets at 32K. BF16 maximum
spills also reintroduce order dependence. This is a numerical design tradeoff,
not a qualified kernel improvement.
