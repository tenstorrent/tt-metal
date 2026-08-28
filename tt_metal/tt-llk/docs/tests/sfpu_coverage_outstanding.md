<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# SFPU input-value coverage: what is still outstanding

Companion to [`sfpu_edge_coverage_plan.md`](sfpu_edge_coverage_plan.md), which holds the two
items that still have a written plan (W1 and W9). This file holds everything else the coverage
ledger shows as open, and it exists because the ledger surfaced work the original plan could
not have listed — it was written by reading code, and three of the items below are things it
had no way to see.

Numbering continues the plan's series, so the two files share one namespace and a reference to
"W14" is unambiguous. W2–W8, W10, W11 and W13 are closed.

**Regenerate the numbers before trusting them:**

```bash
cd tests/python_tests && python -m helpers.sfpu_domains --report
```

Every count in this document came from that command. It needs no hardware.

---

## Summary

| # | Gap | Effort | Value | Blocked by |
|---|---|---|---|---|
| [W1](sfpu_edge_coverage_plan.md#w1--signed-zero-at-a-registered-pole) | `-0.0` never reaches a registered pole — the whole of cat G | S | High | — |
| [W12](#w12--the-unary-family-has-no-cat-b-verdict-table) | 29 float ops have no cat-B verdict: not enrolled, no reason recorded | M | High | — |
| [W14](#w14--integer-extremes-record-the-exclusions) | 13 int ops are not driven at the format extremes and nothing says why | S | Medium | — |
| [W15](#w15--eighteen-float-ops-have-no-deliberate-edge-value-at-all) | 18 float ops are driven only by the random sweep | M | Medium | W12 settles most of it |
| [W16](#w16--sfpulogsigmoid-needs-a-derived-operand-probe-to-join-cat-b) | `SfpuLogsigmoid` cannot join cat B through a product of independent lists | M | Low | — |
| [W17](#w17--the-bfp8_b-lattice-path-still-has-no-caller) | `_bfp_block_aware_compare` is never reached on a Bfp8_b output | S | Low | — |
| [W18](#w18--cat-fs-remaining-tranches) | 85 float ops are outside `EXTREMES_READY_OPS` | L | Low | — |
| [W9](sfpu_edge_coverage_plan.md#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | M | Medium | needs a kernel-contract ruling |

Suggested order: **W1 → W12 → W14** (each small or already understood), then
**W15 → W16 → W17**, then **W18** and **W9** last — W18 for diminishing returns, W9 because it
cannot start until someone rules on what the trig kernels promise.

W13 is closed and its section is gone. It drove nothing new; it stopped the ledger overstating
the gap, so the counts below are now the real ones — cat B's backlog is 29 ops rather than the
50 it read before, cat C's is 13 rather than 16, and cat F's 85 rather than 106.

---

## The ledger as it stands

```
A singularities          covered  20  n/a 109  unrecorded   0
B ieee_specials          covered  76  n/a  24  unrecorded  29
C integer_extremes       covered   8  n/a 108  unrecorded  13
D knees                  covered  45  n/a  84  unrecorded   0
E operand_parameters     covered   5  n/a 124  unrecorded   0
F magnitude_extremes     covered  23  n/a  21  unrecorded  85
G signed_zero_at_a_pole  covered   0  n/a 115  unrecorded  14
```

`unrecorded` is a distinct state from `n/a` on purpose: it means nothing records whether the
class applies, which is a different problem from a class that does not apply. Most of the work
below is turning `unrecorded` into one or the other.

---

## W12 — The unary family has no cat-B verdict table

### Problem

The binary and ternary families each partition their ops across a `*_SPECIALS_READY_OPS` and a
`_*_SPECIALS_NOT_READY` dict, and `test_sfpu_domains` asserts the partition is total — so an op
outside cat B has a *recorded reason* for being outside it. The unary family has only the first
half. `SPECIALS_READY_OPS` holds 67 unary ops; the other 28, plus `SfpuAddTopRow`, are outside
cat B with nothing saying whether that is a decision or an omission:

```
CastFp32ToFp16a, Digamma, Erf, Erfc, Erfinv, Expm1Cw, Frac, Gelu, GeluDerivative,
Heaviside, I1, Lgamma, Log, LogWithBase, Polygamma, Rdiv, ReciprocalCompat, ReluMin,
Rpow, RsqrtCompat, SfpuAddTopRow, Sigmoid, SigmoidAppx, Sign, SqrtCustom, Tanh,
TanhDerivative, TanhDerivativeLut, UnaryPower
```

Some of those reasons are already known and written down — just as prose in a section comment
rather than in a table anything can read:

- **`Log`** — the section comment above `_dest_acc_flag` records it: the kernel clamps a
  non-finite input to the format maximum and takes the log of that, so `log(+inf)` comes back
  as 88.5 rather than `+inf`. Written up as "*kernel* behaviour with no ISA ruling", needing an
  owner.
- **`Sign` and `Heaviside`** — `_BINARY_SPECIALS_NOT_READY`'s `SfpuMask` entry names them:
  compare-against-zero on an operand that may be a NaN, through `SFPSETCC`, whose contract is
  conditioned "provided that VC is neither negative zero nor any kind of NaN".

That is three of the 29 already answered and unrecorded, which is the shape of the whole item.

### Steps

1. **Add `_UNARY_SPECIALS_NOT_READY: Dict[MathOperation, str]`** next to `SPECIALS_READY_OPS`,
   with the same "an op cannot be in both" assertion the other two families carry.

2. **Seed it from what is already known**, moving the prose above into entries: `Log`,
   `Sign`, `Heaviside`. Do not paraphrase — the existing wording is the measured result.

3. **Make the partition total, and let the test enforce it.** Extend
   `test_every_float_binary_op_is_classified_for_cat_b`'s pattern to the unary family, so a
   newly registered unary op fails at collection until someone decides.

4. **Measure the rest in one tranche, as the third unary tranche was measured.** Drive the
   remaining ~26 over the full specials set on every Blackhole-reachable triple, enrol the ones
   that agree, and record a reason for the ones that do not. Convention 3 applies: a reason
   string written to make a variant green becomes a permanent claim about the hardware.

### Expect

Several to be one cause, not many. `Sigmoid` / `SigmoidAppx` / `Tanh` / `TanhDerivative` /
`TanhDerivativeLut` are LUT compositions and `SFPLUTFP32` documents no NaN/inf handling — the
same section 5.6 Q1 question that holds six binary ops and two ternary operands out. The gamma
family is already excluded at its poles for a domain reason that likely extends to cat B.

### Pin it

The totality test in step 3 is the pin. It is what converts "nobody got to it" into "here is
why not", and it is why this item is worth more than the 29 cells suggest.

**Cost:** one dict, one test, one measurement tranche.

---

## W14 — Integer extremes: record the exclusions

### Problem

Thirteen ops are unrecorded for cat C:

```
SfpuDivInt32, SfpuDivInt32Floor, SfpuFmodInt32, SfpuGcd, SfpuLcm, SfpuMaxInt32,
SfpuMaxUint32, SfpuMinInt32, SfpuMinUint32, SfpuMulInt32, SfpuRemainderInt32,
SfpuRemainderUint32, SfpuRsubInt32
```

Almost every one of them has a *documented sub-range* in `_INT_BINARY_STIMULI`'s comments that
the format extremes fall outside of — div and fmod below `2**24` for an exact int→fp32
reciprocal, lcm assuming `|a|, |b| < 2**15`, mul below ~46340 so the product stays under
`2**31`, max/min non-negative so signed and unsigned agree. Driving those at `INT32_MAX` would
produce failures that are documented limitations rather than findings, which is the trap
`test_eltwise_binary_sfpu_int_extremes`' own scope comment warns about.

So this is not a stimulus gap. It is that nothing records the exclusion, so a reader cannot tell
it from an omission — the same problem W3 solved for the zero divisor.

### Steps

1. **Add `_INT_EXTREMES_OUT_OF_RANGE: Dict[MathOperation, str]`** in the binary suite, next to
   `_INT_ZERO_UNDEFINED_DIVISOR` and in the same shape, with the sub-range each op documents.
   Take the reasons from `_INT_BINARY_STIMULI`'s existing comments rather than restating them.

2. **Check `SfpuRsubInt32` separately.** It is the one op in the list with no documented
   sub-range — `out = in1 - in0` is exact integer subtraction — so it may be a genuine cat-C
   candidate rather than an exclusion. Measure before deciding, and note the subtraction can
   overflow int32, which is its own question.

3. **Assert totality**, as W3's zero probe does: every op in `_INT_BINARY_STIMULI` is either
   driven at the extremes or has an entry here.

**Cost:** one table, one test, one measurement for rsub.

---

## W15 — Eighteen float ops have no deliberate edge value at all

### Problem

The ledger's most direct output: these ops have no `COVERED` cell in any of the seven classes,
so nothing but the random sweep has ever driven them at a value chosen on purpose.

```
CastFp32ToFp16a, Digamma, Erf, Erfc, Expm1Cw, Gelu, GeluDerivative, I1, Lgamma,
Polygamma, Rpow, SfpuAddTopRow, Sigmoid, SigmoidAppx, Tanh, TanhDerivative,
TanhDerivativeLut, UnaryPower
```

Nothing here is known to be *wrong*. The point is that each one is covered by a uniform draw
over a registered domain and by nothing else — no pole, no knee, no special, no extreme.

### Steps

1. **Do W12 first.** It settles cat B for every op on this list, which is the cheapest class to
   give them and may be all several of them need.

2. **Then ask, per op, whether it has a knee worth registering.** `Erf` and `Erfc` saturate;
   `Sigmoid` and `Tanh` saturate at both ends; `GeluDerivative` and `TanhDerivative` have a
   maximum. A `_OP_EDGE_POINTS` entry is all it takes to enrol one, because the edge sweep
   derives its ops.

3. **Leave the gamma family alone**, and record that it is deliberate — the plan's "What stays
   uncovered" section already explains why a probe at their poles tests a value the kernel never
   promised.

4. **Treat `SfpuAddTopRow` separately.** It is not element-wise — it returns before
   `BinarySFPUGolden`'s Dest modelling and cannot report a generated-NaN mask — so it needs its
   own decision rather than a shared one.

**Cost:** mostly W12's. Each knee after that is a table entry.

---

## W16 — `SfpuLogsigmoid` needs a derived-operand probe to join cat B

### Problem

`_BINARY_SPECIALS_NOT_READY` records why it is out, and the reason is structural rather than a
missing measurement: the kernel's contract is `in1 == exp(-in0)`, so the two operands are not
independent — but cat B here is driven by `edge_pair_values()`, a cartesian *product* of two
independently-chosen lists. A NaN placed in B against a finite A is not `exp(-A)`, so the pair
is not a stimulus the kernel has any contract about and asserts nothing whichever way it comes
out.

### Steps

1. **Add a paired cat-B builder** alongside `edge_pair_values()`: one that takes A's value list
   and a function producing B from it. `_logsigmoid_stimuli_specs()` already builds both
   operands from one ramp; this is the same idea for a fixed list rather than a linspace.

2. **Derive B at the specials.** `exp(-inf) = 0`, `exp(inf) = inf`, `exp(-NaN) = NaN` — all
   coherent, so the probe is well defined.

3. **Then move the op's entry** from `_BINARY_SPECIALS_NOT_READY` to
   `BINARY_SPECIALS_READY_OPS` with a measured reason, or rewrite it again if it diverges.

### Expect

The `x > 4` branch returns `-in1` exactly (measured: max `|hw - (-exp(-x))|` of 0 over 256
lanes), so a non-finite in B should pass straight through on that branch and be ignored on the
other two. The interesting case is `x` itself being non-finite.

**Cost:** one builder, one variant. Low value — it closes one cell.

---

## W17 — The Bfp8_b lattice path still has no caller

### Problem

`passed_test` falls through to `_bfp_block_aware_compare` only when the tolerance check has
rejected something, and on a Bfp8_b output nothing rejects. Measured over `Exp`, `Gelu`, `Silu`
and `Sqrt` at block spreads of `2**-4`, `2**-12` and `2**-24`: `torch.isclose` accepts all 4096
elements every time, even with 2816 of them flushed to zero by the shared exponent.

W10 expected its block spread to be what made that path earn its keep, and it is not. The
stimulus is not the problem — the block really does span the exponent — it is that golden and
hardware agree closely once both have been through the same output quantization.

The path is exercised for Bfp4_b and Bfp2_b outputs, where there is no tolerance pre-check and
the lattice is the only verdict. So this is about Bfp8_b alone.

### Steps

1. **Decide whether the fallback is still needed.** It was added for a reason; if no current
   suite can reach it on Bfp8_b, either a suite that could has gone away or the tolerance has
   since widened. Read the history before writing a test to justify it.

2. **If it is needed, find the op that reaches it.** It will be one whose *approximation* error
   is large relative to its block, not one whose stimulus is wider — that is the axis this
   depends on. Do not engineer a stimulus to force it.

3. **If it is not, say so** where the fallback is defined, rather than leaving a branch that
   reads as load-bearing.

**Cost:** an afternoon of reading, and possibly no code at all.

---

## W18 — Cat F's remaining tranches

### Problem

`EXTREMES_READY_OPS` holds 14 ops and the saturation sweep covers nine more, so 23 of 129 are
covered for cat F. The rest are unrecorded — 21 of them integer-only and therefore W13's
problem, leaving ~85 float ops that have never been driven at their format's ceiling, its
neighbour, its smallest normal or a subnormal.

This is last on the list on purpose. Cat F is opt-in per op precisely because driving an
unenrolled op at an extreme produces a wall of failures with one root cause, and the first
tranche was chosen as the ops whose behaviour there is uncontroversial. Every op after that is
its own measurement, and the yield falls off: the first tranche found one real fact
(`subnormal_delivered`), and it found it with four ops.

### Steps

1. **Enrol in tranches, cheapest first**, as the first one was: ops whose golden is plain
   arithmetic and whose behaviour at an extreme is not in question.

2. **Raise `_COVERAGE_FLOORS["F"]`** with each tranche. That is the ratchet working as intended
   and is the natural last step of enrolling an op.

3. **Stop when a tranche finds nothing.** The measurement is the product, not the count.

**Cost:** open-ended. Budget one tranche at a time and re-read the report between them.

---

## What will not close

Unchanged from the plan's own section, and repeated here only as a pointer:
`INT32_MIN` through any int32 kernel, the gamma family at their poles, a zero's sign in the
*result* (which bounds what W1 can achieve — it fixes the input side only), and a generated
NaN's sign on Wormhole. See
[`sfpu_edge_coverage_plan.md`](sfpu_edge_coverage_plan.md#what-stays-uncovered-and-why).
