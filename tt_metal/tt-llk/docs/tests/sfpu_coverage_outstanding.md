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
"W18" is unambiguous. W1–W8 and W10–W17 are closed.

**Regenerate the numbers before trusting them:**

```bash
cd tests/python_tests && python -m helpers.sfpu_domains --report
```

Every count in this document came from that command. It needs no hardware.

---

## Summary

| # | Gap | Effort | Value | Blocked by |
|---|---|---|---|---|
| [W18](#w18--cat-fs-remaining-tranches) | 99 float ops are outside `EXTREMES_READY_OPS` | L | Low | — |
| [W9](sfpu_edge_coverage_plan.md#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | M | Medium | needs a kernel-contract ruling |

Two items left. **W18** is open-ended with falling yield; **W9** cannot start until someone
rules on what the trig kernels promise outside ±π.

Six of the seven classes have nothing unrecorded. Cat F is the exception, and it is W18.

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
F magnitude_extremes     covered  23  n/a  21  unrecorded  99
G signed_zero_at_a_pole  covered  17  n/a 126  unrecorded   0
```

`unrecorded` is a distinct state from `n/a` on purpose: it means nothing records whether the
class applies, which is a different problem from a class that does not apply. Most of the work
below is turning `unrecorded` into one or the other.

---

## W18 — Cat F's remaining tranches

### Problem

`EXTREMES_READY_OPS` holds 14 ops and the saturation sweep covers nine more, so 23 of 143 are
covered for cat F. The other 21 integer-only ops read "not applicable" — no subnormal band, no
float ceiling — which leaves 99 float ops that have never been driven at their format's
ceiling, its neighbour, its smallest normal or a subnormal.

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
*result* — the signed-zero work drives a `-0.0` *into* a pole and cannot check the sign of a
zero that comes back out — and a generated NaN's sign on Wormhole. See
[`sfpu_edge_coverage_plan.md`](sfpu_edge_coverage_plan.md#what-stays-uncovered-and-why).
