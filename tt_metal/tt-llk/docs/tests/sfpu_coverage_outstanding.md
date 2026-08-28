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
"W16" is unambiguous. W1–W8 and W10–W15 are closed.

**Regenerate the numbers before trusting them:**

```bash
cd tests/python_tests && python -m helpers.sfpu_domains --report
```

Every count in this document came from that command. It needs no hardware.

---

## Summary

| # | Gap | Effort | Value | Blocked by |
|---|---|---|---|---|
| [W16](#w16--sfpulogsigmoid-needs-a-derived-operand-probe-to-join-cat-b) | `SfpuLogsigmoid` cannot join cat B through a product of independent lists | M | Low | — |
| [W17](#w17--the-bfp8_b-lattice-path-still-has-no-caller) | `_bfp_block_aware_compare` is never reached on a Bfp8_b output | S | Low | — |
| [W18](#w18--cat-fs-remaining-tranches) | 85 float ops are outside `EXTREMES_READY_OPS` | L | Low | — |
| [W9](sfpu_edge_coverage_plan.md#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | M | Medium | needs a kernel-contract ruling |

Suggested order: **W16 → W17**, then **W18** and **W9** last — W18 for diminishing returns,
W9 because it cannot start until someone rules on what the trig kernels promise.

Six of the seven classes have nothing unrecorded. Cat F is the exception, and it is W18.

Four ops still have no *covered* class at all, and all four are explained rather than
outstanding: `Digamma`, `Lgamma` and `Polygamma` have poles at zero and registered domains that
start above it, so every probe the suite could place would be a value the kernel never promised;
`SfpuAddTopRow` is not element-wise, so the sweeps' whole shape misses it.

---

## The ledger as it stands

```
A singularities          covered  20  n/a 109  unrecorded   0
B ieee_specials          covered  79  n/a  50  unrecorded   0
C integer_extremes       covered  20  n/a 109  unrecorded   0
D knees                  covered  59  n/a  70  unrecorded   0
E operand_parameters     covered   5  n/a 124  unrecorded   0
F magnitude_extremes     covered  23  n/a  21  unrecorded  85
G signed_zero_at_a_pole  covered  14  n/a 115  unrecorded   0
```

`unrecorded` is a distinct state from `n/a` on purpose: it means nothing records whether the
class applies, which is a different problem from a class that does not apply. Most of the work
below is turning `unrecorded` into one or the other.

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
covered for cat F. The other 21 integer-only ops read "not applicable" — no subnormal band, no
float ceiling — which leaves 85 float ops that have never been driven at their format's
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
