# Log2-lattice maxima: order consistency and its tradeoff

## Scope

[Standalone CPU driver](exp_order_lattice_cpu.py) extends the isolated
[continuous-exp mechanism study](EXP_ORDER_CPU_MECHANISM.md), without editing
that measured producer. The [84-case ledger](exp-order-lattice-cpu-v1.jsonl)
confirms that an ideal lattice removes order dependence, but does not make
the approximation accurate or invariant to arbitrary common score offsets.
There is no device implementation, throughput claim, or SageAttention execution.

## Measured results

Tables are generated from the JSON; raw values remain authoritative.
These are CPU FP64 outputs, except the explicitly labeled BF16-output column.
Original-reference and inter-order metrics cover the same 64 sampled Q rows
and all KV, unlike the separate H2 device experiment's input/metric scope.

Continuous native surrogate:

| N | Maximum schedule | Identity original L2 % | Reverse original L2 % | Reverse vs identity % | RMS-sort vs identity % | Reverse vs identity, BF16 output % |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | Ordinary running | 1.907 | 1.898 | 2.318 | 2.050 | 2.331 |
| 4,096 | Fixed global oracle | 1.851 | 1.851 | <1e-12 | <1e-12 | 0 |
| 4,096 | Ideal lattice running | 1.762 | 1.762 | <1e-12 | <1e-12 | 0 |
| 4,096 | Lattice + BF16 max spill | 1.761 | 1.761 | 0.081 | 0.076 | 0.161 |
| 32,768 | Ordinary running | 1.889 | 1.867 | 2.576 | 2.082 | 2.588 |
| 32,768 | Fixed global oracle | 1.827 | 1.827 | <1e-12 | <1e-12 | 0 |
| 32,768 | Ideal lattice running | 1.809 | 1.809 | <1e-12 | <1e-12 | 0 |
| 32,768 | Lattice + BF16 max spill | 1.810 | 1.809 | 0.113 | 0.090 | 0.197 |

Ideal-lattice reverse discrepancies are 3.96e-14% and 6.36e-14% at 4K/32K;
the three orders have identical BF16 output hashes at each length. Its final
BF16 output original-reference L2 is 1.768%/1.815%. Thus even perfect order
consistency retains roughly 1.8% numerical error on this one normal seed.

Adding a constant to every score leaves the original exact-attention reference
unchanged. Identity-order ideal-lattice results expose the separate tradeoff:

| N | Common score offset | Original L2 % | Output change versus unshifted % | Output change, BF16 output % |
|---:|---|---:|---:|---:|
| 4,096 | ln(2)/4 | 1.751 | 2.538 | 2.549 |
| 4,096 | ln(2)/2 | 1.742 | 3.395 | 3.406 |
| 32,768 | ln(2)/4 | 1.794 | 2.628 | 2.638 |
| 32,768 | ln(2)/2 | 1.805 | 3.494 | 3.498 |

Ordinary running/global native schedules remain common-shift invariant to
FP64 roundoff (maximum 6.26e-14% output change). All exact-exp schedules also
match original attention and retain order/shift invariance to FP64 roundoff.
The lattice's similar aggregate original L2 across offsets does not mean the
outputs are equally or consistently correct: their errors change direction.

**Decision:** this is a useful mechanism/representation experiment, not a
drop-in accuracy repair. Before considering a kernel, weigh the remaining
exp error and lost score-shift invariance against its order consistency, and
compare against the already measured exp refiners. No throughput benefit
has been demonstrated.

## Why an online lattice can remove exp-history dependence

For the continuous native surrogate, write
`F(x)=2^floor(t)*(1+frac(t))`, where `t=x/ln(2)-c` and c is the pinned
native-exp bias constant. For any integer n,

```text
F(x - n*ln(2)) = 2^-n * F(x).
```

Select each online maximum by rounding upward to a fixed absolute lattice:
`m = ceil(max_score/ln(2))*ln(2)`. Unlike the fixed-global-max oracle,
this needs only the current block maximum and previous state, not a prior
full-score maximum. The final lattice maximum M and every insertion maximum
m_i differ by an integer number of octaves. Consequently, with exact rescale,

```text
F(s_i-m_i) * exp(m_i-M) = F(s_i-M).
```

Thus the approximate weights are history-independent, apart from finite
arithmetic. This is not an accuracy theorem: their relative weights still
contain the nonconstant exp-approximation distortion. In ideal arithmetic,
normalized lattice attention is equivalent to normalizing `F(s_i)` with a
fixed absolute phase; the scale introduced by the final lattice maximum
cancels.

## Lost invariance: arbitrary common score offsets

Exact softmax is unchanged by adding the same offset a to all scores in a
row. Ordinary running/global maxima also translate by a, preserving their
score differences in ideal arithmetic. A fixed absolute lattice does not
generally translate by a unless a is an integer multiple of ln(2).
Therefore `F(s_i+a)` can have different relative distortion from `F(s_i)`.
Removing history dependence can trade it for sensitivity to an arbitrary
rowwise score offset, including offsets induced by mathematically removable
common K components. It is not a free accuracy improvement.

Anchoring the lattice to the first block's maximum would restore common-offset
equivariance, but make the lattice phase depend on which block arrives first.
That alternative is not implemented or measured here.

## Why naive BF16 snapping is a separate question

The BF16-spill control rounds each ideal snapped real maximum to BF16 before
using it. Those stored numbers generally no longer differ by exact multiples
of ln(2), so the translation identity is no longer guaranteed. Even the upward
bound on the true score maximum can be lost near a boundary after rounding;
the driver records any positive score-exp argument.

This control isolates only maximum storage. It does not emulate native integer
exp-grid rounding, FPU subtraction/alignment, quantized inputs or probabilities,
SFPU rescale approximation, or the actual kernel's scaled/unscaled maximum
representation. A viable kernel would need to preserve the intended integer
octave coordinate and exp-code phase through its real pipeline. Storing
`ceil(m/ln(2))*ln(2)` in an existing BF16 CB is not by itself proof of that.

The observed BF16-spill lattice phase deviation reaches 0.01195/0.02009
octaves at 4K/32K. A score-exp argument becomes as large as +0.00554 in the
BF16-spill controls: rounding can indeed destroy the intended upper bound.
See [Blackhole feasibility audit](LATTICE_MAX_FEASIBILITY.md) for the actual
CB, subtraction, exp-code and rescale constraints. Its proposed integer-code
approach is unimplemented and does not remove the common-offset tradeoff.

## Independent evidence audit

The driver uses the exact v1 BF16 normal inputs, H1/D128, seed1240,
N4096/32768, 64 sampled Q rows and all KV. All unmodified ordinary/global
controls match v1 input, score, original FP64 reference, permutation and output
hashes. Every case passes deterministic replay, original-input/source
immutability, finite output, positive denominator and independent telescoped
dense-weight checks. All ideal lattice cases match a fixed-snapped-global
dense calculation and are order-invariant to FP64 roundoff.

The 48 order cases cover exact/native exp, ordinary/global/ideal-lattice/
BF16-lattice-spill schedules and identity/reverse/block-RMS orders at both
lengths. Another 36 identity-order controls add offsets 0, ln(2)/4 and ln(2)/2
to every score while keeping the original unshifted FP64 reference. These
are not the full Cartesian product of offsets and order permutations.

All 84 expected unique cases and the successful completion marker were
independently checked. All seven source/evidence hashes match current local
files, including the measured v1 ledger; the headers are formula provenance,
not executed CPU dependencies or a complete Torch/BLAS manifest. Scalar
translation checks pass for 1/2/7-octave shifts. Maximum absolute telescoped
algebra discrepancy is 7.08e-16. Exact-exp original L2 remains below
6.01e-13%, and maximum exact-exp inter-order L2 below 1.18e-13%.
The run used four CPU threads and one interop thread, completing in 4.4 seconds
with zero device jobs. No existing producer was changed.

BF16 final-output casting is reported separately. CPU runtime is not device
performance. The original-reference metric is never shifted to excuse error.
