<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# SFPU input-value coverage: what is still outstanding

Companion to [`sfpu_edge_coverage_plan.md`](sfpu_edge_coverage_plan.md), which holds W9 — the
one item there that is still open. This file holds everything else the coverage
ledger shows as open. It exists because the ledger surfaced work the original plan could not
have listed: that plan was written by reading code, and several of the items below are things
no amount of reading would have found.

Numbering continues the plan's series, so the two files share one namespace and a reference to
"W19" is unambiguous. W1–W8 and W10–W18 are closed.

**Regenerate the numbers before trusting them:**

```bash
cd tests/python_tests && python -m helpers.sfpu_domains --report
```

Every count in this document came from that command. It needs no hardware.

---

## Summary

| # | Gap | Effort | Value | Blocked by |
|---|---|---|---|---|
| [W19](#w19--cat-f-has-no-sweep-for-the-binary-and-ternary-families) | 25 binary and ternary ops have no cat-F sweep to be enrolled into | M | Medium | — |
| [W9](sfpu_edge_coverage_plan.md#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | M | Medium | needs a kernel-contract ruling |

Two items left. **W19** is the one gap with real work in it; **W9** cannot start until someone
rules on what the trig kernels promise outside ±π — a ruling that now blocks two coverage
classes rather than one, since W18 found `sin`/`cos`/`tan` returning ±inf at 3.39e38 against a
bounded golden.

**Every class now reports nothing unrecorded.** Each of the 143 ops has a verdict for all seven
classes: covered, not applicable, or a recorded reason. `test_no_class_has_anything_unrecorded`
is what keeps it that way — a new op or a new sweep can put cells back, and the fix is a verdict
rather than a floor bump.

Four ops still have no *covered* class at all, and all four are explained rather than
outstanding: `Digamma`, `Lgamma` and `Polygamma` have poles at zero and registered domains that
start above it, so every probe the suite could place would be a value the kernel never promised;
`SfpuAddTopRow` is not element-wise, so the sweeps' whole shape misses it.

---

## The ledger as it stands

```
143 ops
A singularities          covered  23  n/a 120  unrecorded   0
B ieee_specials          covered  88  n/a  55  unrecorded   0
C integer_extremes       covered  24  n/a 119  unrecorded   0
D knees                  covered  59  n/a  84  unrecorded   0
E operand_parameters     covered   5  n/a 138  unrecorded   0
F magnitude_extremes     covered  78  n/a  65  unrecorded   0
G signed_zero_at_a_pole  covered  17  n/a 126  unrecorded   0
```

`unrecorded` is a distinct state from `n/a` on purpose: it means nothing records whether the
class applies, which is a different problem from a class that does not apply. Most of the work
below is turning `unrecorded` into one or the other.

---

## W19 — Cat F has no sweep for the binary and ternary families

### Problem

W18 closed cat F for the unary family and, in doing so, showed that the class was never
*reachable* for the other two. `test_eltwise_unary_sfpu_extremes` is unary-only, and the
saturation sweep covers `SfpuElwmul` and `SfpuElwadd`. That leaves 25 ops with nothing to be
enrolled into:

```
SfpuAddTopRow  SfpuAddcdiv  SfpuAddcmul  SfpuAtan2  SfpuBinaryFmod  SfpuBinaryMax
SfpuBinaryMin  SfpuBinaryRemainder  SfpuElwEq  SfpuElwGe  SfpuElwGt  SfpuElwLe
SfpuElwLt  SfpuElwNe  SfpuElwdiv  SfpuElwpow  SfpuElwrsub  SfpuElwsub  SfpuIsclose
SfpuLerp  SfpuLogsigmoid  SfpuMask  SfpuSnakeBeta  SfpuWhere  SfpuXlogy
```

The ledger reports these as *not applicable* with a reason naming the missing sweep, rather
than as unrecorded — a missing sweep and an undecided op are different problems, and reading
25 of the first as 25 of the second would invent a backlog of decisions nobody owes.

Nothing here is known to be wrong. A magnitude extreme has simply never been driven into a
binary or ternary operand.

### Steps

1. **Binary first, and one operand at a time.** `extreme_values()` already returns the cat-F
   list for a pipeline; the shape to copy is `test_sfpu_ternary_operand_edges`' operand axis —
   the probed operand takes the extremes, the other keeps its random domain, so one variant
   asks one question. A product of two extreme lists would pair index-wise rather than crossing,
   which is the trap recorded there.

2. **Expect the divergences W18 already found, in binary form.** `div` and `xlogy` compose a
   reciprocal and a log, so the input-FTZ group's subnormal behaviour should reappear;
   `SfpuElwpow` cannot reach its ceiling inside its registered domain, as the saturation sweep
   already records. Enrol per op against a measurement, not against the unary verdict.

3. **Ternary after binary.** `addcdiv` and `snake_beta` divide by operand C, so an extreme
   there interacts with the pole already registered on it — drive them as separate classes, the
   way `test_sfpu_ternary_operand_edges` separates `pole` from `specials_in`, or one xfail will
   cover two causes.

4. **Raise `_COVERAGE_FLOORS["F"]`** as each family lands, and drop the ops from
   `SuiteCoverage.extremes_sweepable`'s complement — the ledger stops calling the class
   inapplicable for them the moment a sweep can reach them.

**Cost:** one variant per family, plus a measurement pass each.

---

## What will not close

Unchanged from the plan's own section, and repeated here only as a pointer:
`INT32_MIN` through any int32 kernel, the gamma family at their poles, a zero's sign in the
*result* — the signed-zero work drives a `-0.0` *into* a pole and cannot check the sign of a
zero that comes back out — and a generated NaN's sign on Wormhole. See
[`sfpu_edge_coverage_plan.md`](sfpu_edge_coverage_plan.md#what-stays-uncovered-and-why).
