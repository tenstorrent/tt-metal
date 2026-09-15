# Exp-order mechanism in an isolated FP64 CPU model

## Conclusion and scope

**Input-dependent exp approximation alone can create substantial online
attention order dependence, without low-precision accumulation.** This is
established within the deliberately simplified model below, not as a
quantitative attribution of Blackhole device error.

[Driver](exp_order_cpu_mechanism.py) and
[36-case ledger](exp-order-cpu-mechanism-v1.jsonl). All cases use original
normal BF16 Q/K/V, H1/D128, seed1240, N4096/32768, 64 explicit sampled Q
rows and every KV token. Score matmul, state, PV, normalization and online
rescaling use CPU FP64. The exp rescale is always `exp(m_old-m_new)`.
Final BF16 output casting is measured separately.

This is **not bit-exact device emulation**. It omits Q7/KV5/BFP quantization,
integer exp-grid rounding, BF16 rowmax/score spills, FPU product alignment,
native packer behavior and device synchronization. H1 inputs use this driver's
own generator, not the H2 device experiment's input tensors. A similar
inter-order percentage on CPU and device is not a matched-input decomposition
or evidence that all device error has been explained.

## Controlled axes

Score-exp families:

- Exact: `F(x)=exp(x)` in FP64.
- Constant bias: `F(x)=c0*exp(x)`, `c0=0.9701782328162476`.
- Continuous native surrogate: with `c=(32512-32500.818359375)/256`,
  `t=x/log(2)-c`, use `F(x)=2^floor(t)*(1+frac(t))`.

The surrogate retains the source approximation's linear mantissa shape and B
constant but uses an ideal log2 factor, no integer-grid rounding, and no native
clamping. The model also deliberately omits its scale-rounding instructions.

Each family is tested with a running maximum or a fixed, precomputed global
maximum, and identity, reverse or descending K-block-RMS order. Joint whole
512-token K/V blocks move together. RMS sorting is not actual query-max
sorting. **The global maximum is an oracle diagnostic**: its prior full-score
computation is not free, and no feasible/performance-qualified device algorithm
is claimed.

## Numerical results

Tables are generated from the JSON; raw values remain authoritative. Both
original-reference and inter-order metrics cover the same 64 sampled Q rows
in this CPU study. Inter-order L2 uses the same family's/schedule's
identity-order output as reference, not original exact attention.

Continuous-native surrogate, FP64 outputs unless labeled otherwise:

| N | Maximum schedule | Identity L2 % | Reverse L2 % | RMS-sort L2 % | Reverse vs identity % | RMS-sort vs identity % | Reverse BF16-output L2 % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4,096 | Running | 1.907 | 1.898 | 1.917 | 2.318 | 2.050 | 1.907 |
| 4,096 | Fixed global | 1.851 | 1.851 | 1.851 | <1e-12 | <1e-12 | 1.860 |
| 32,768 | Running | 1.889 | 1.867 | 1.854 | 2.576 | 2.082 | 1.875 |
| 32,768 | Fixed global | 1.827 | 1.827 | 1.827 | <1e-12 | <1e-12 | 1.832 |

With the global maximum fixed, reverse-order FP64 discrepancy is
1.90e-14%/4.99e-14% at 4K/32K; all three orders produce identical BF16 output
hashes. With running maxima, reverse-order BF16-output discrepancies remain
2.331%/2.588%, so final BF16 casting does not explain the phenomenon.

Exact and constant-bias controls, maxima over all orders and both schedules:

| N | Maximum original-reference FP64 L2 % | Maximum inter-order FP64 L2 % | BF16-output rounding-floor L2 % |
|---:|---:|---:|---:|
| 4,096 | 2.30e-13 | 1.00e-13 | 0.165 |
| 32,768 | 6.01e-13 | 1.18e-13 | 0.168 |

Every exact/constant-bias control at a given length has the same BF16 output
hash and reaches the exact reference's BF16 rounding floor. A constant shared
score-exp bias therefore cancels as expected when numerator and denominator
use the same represented weights.

## Why fixed maxima remove order dependence but not exp error

Let M be the final maximum and m_i the running maximum when token i enters.
Write `F(x)=exp(x)*r(x)`. With exact rescaling, that token's final
unnormalized weight is:

```text
F(s_i-m_i) * exp(m_i-M) = exp(s_i-M) * r(s_i-m_i).
```

The model's independent dense calculation of these telescoped weights agrees
with its blockwise online recurrence. If r is constant, it cancels during
normalization. If r depends on the mantissa phase, changing insertion maxima
changes weights. Fixing m_i=M for every token removes this history dependence,
but still leaves the nonconstant factors `r(s_i-M)`. Hence global-max native
error remains around 1.83–1.85% even as inter-order discrepancy falls to FP64
roundoff. **Order invariance is not equivalent to accurate attention.**

A scalar example in the ledger makes the same mechanism explicit:
`F(-0.3)*exp(-0.4)/F(-0.7)=1.049345` for the continuous surrogate, versus
one for exact/constant-bias exp. Shifting by an exact log(2) octave instead
preserves the surrogate ratio, consistent with its periodic mantissa shape.

Together with the same-grid device LUT ablation in
[BLOCK_PERMUTATION.md](BLOCK_PERMUTATION.md), this supplies complementary
evidence: a controlled hardware intervention reduces order sensitivity, and
an isolated high-precision model reproduces the proposed mathematical
mechanism. It does not subtract CPU error from device error or label the
remaining device discrepancy pure FP32 accumulation noise.

## Independent audit

All 36 expected unique cases are present, with a successful completion marker
and zero device jobs. The five recorded source hashes match the formatted
local driver and cited source files at this audit. Those headers are pinned
formula provenance, not executed CPU dependencies or a complete Torch/BLAS
implementation manifest.

For each length, all cases share original input, sampled score, exact reference
and Q-row hashes. Valid whole-block permutations and inverse original-bit
recovery are asserted by the driver. Every result records finite sampled
outputs, positive denominator, final maximum equal to the global maximum,
one bitwise-deterministic CPU replay, unchanged inputs/sources, and passed
FP64 permutation/reference and telescoped-weight algebra checks. Exact and
constant-bias reference checks and fixed-global order checks pass throughout.
Maximum absolute algebra discrepancy is 6.80e-16; maximum reference-permutation
discrepancy is 7.22e-16 (`rtol=1e-11, atol=1e-12`).

The run used Torch 2.11.0+cpu, four CPU threads and one interop thread,
finishing in 1.2 seconds. This is model wall time, not kernel performance.
No existing numerical producer was changed.

## Follow-up test coverage

[test_exp_order_cpu_mechanism.py](test_exp_order_cpu_mechanism.py) adds small
CPU unit controls for the scalar exp identity, a deterministic two-block
order-dependence example, and constant-V preservation. These complement the
recorded full-model gates. All three tests passed separately on the remote
Python 3.10 CPU environment after formatting; local execution without Torch
explicitly skips them.

The subsequent [84-case lattice study](EXP_ORDER_LATTICE_CPU.md) checks whether
online maxima on a log2 lattice can retain translation consistency without
a global-max prepass. It confirms order invariance in the ideal model, but
exposes arbitrary common-score-offset sensitivity and the loss of exact
lattice structure when maxima spill to BF16. It is not a free accuracy fix.
