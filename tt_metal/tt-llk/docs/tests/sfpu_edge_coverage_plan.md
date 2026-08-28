<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# SFPU input-value coverage: what is left, and how to add it

Scope: the *input values* driven into the SFPU element-wise suites —
`tests/python_tests/test_eltwise_unary_sfpu.py`, `test_eltwise_binary_sfpu.py`,
`test_sfpu_ternary.py` and the shared metadata in
`tests/python_tests/helpers/sfpu_domains.py`. The first two are referred to below by their
short names, `test_sfpu_unary.py` and `test_sfpu_binary.py`.

This is not a review of the edge framework. The cat A/B/C/D machinery
(singularities straddled by a format- *and* `dest_acc`-aware ULP step, IEEE specials
behind a measured delivery matrix, integer extremes through raw overrides, knees sourced
from the same dispatch constants the goldens read) is sound and self-extending. What
follows are the regions of the input space that framework does not currently reach, and
the concrete steps to reach them.

---

## Summary

| # | Gap | Family | Effort | Value | Blocked by |
|---|---|---|---|---|---|
| [W9](#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | unary | M | Medium | needs a kernel-contract ruling |

One item left here. **W9** needs a kernel-contract ruling before any of it can be written;
everything else this document once held is closed. The remaining coverage work that never had a
section here is in
[`sfpu_coverage_outstanding.md`](sfpu_coverage_outstanding.md).

`python -m helpers.sfpu_domains --report` prints the coverage ledger. W9 is cat A.

The numbering is not contiguous: W1–W8, W10 and W11 are closed and their sections have been
removed. W9 keeps its original number so that references from commit messages and reviews still
resolve.

---

## Conventions this plan follows

These are the codebase's own rules, restated so every work item below can be read against
them. Breaking one of them is how a gap gets re-introduced.

1. **Derive enrolment, never list it.** An op joins a sweep by gaining a registry entry
   (`_OP_SINGULARITIES`, `_OP_EDGE_POINTS`, `SPECIALS_READY_OPS`), not by being appended
   to a list in a test file. Every new work item below adds a *table*, and the sweep
   intersects that table with the ops it can drive.
2. **Two independent gates, and both must pass.** A `*_READY_OPS` entry says the
   *golden* defines an answer; a `*_safe()` / `*_delivered()` predicate says the
   *pipeline* delivers the stimulus intact. Neither implies the other. New value classes
   need both halves.
3. **Measure before enrolling.** A reason string written to make a variant green becomes
   a permanent, plausible-looking claim about the hardware. Drive it on silicon first;
   record what happened.
4. **Divergences are non-strict `xfail`s with a reason, enumerated per
   `(input, output, dest_acc)`** — so the case still executes and reports XPASS when
   behaviour changes, and so a combination drifting in or out shows up as a diff.
5. **Partition by failure class before driving.** One tensor holding several unrelated
   edge classes means one `xfail` covers two causes and hides the second. See
   `_EDGE_CLASSES` in `test_sfpu_binary.py`.
6. **Pin new metadata in `test_sfpu_domains.py`.** It runs without hardware, so the
   invariant is checked at collection time in every CI lane.

**Verification loop.** Every work item lists two commands:

```bash
# host-only: the metadata is self-consistent (no device needed)
pytest test_sfpu_domains.py -q

# on silicon: the new values actually pass, or produce a triageable diff
CHIP_ARCH=blackhole pytest <target> -q
```

Getting an interpreter that can run the device-side half on a dev host is its own
problem (`ttexalens` / `tt_umd` version skew); see the team's env notes. `pytest-xdist`
is mandatory even for serial runs because `pytest.ini` puts `--maxschedchunk` in
`addopts`.

---

## W9 — `Tan` has no registered pole; `sin`/`cos` never exceed π

### Problem

Two related holes in the trig family.

**`Tan`.** Its registered domain is `uniform(-1.3, 1.3)` with the comment "stay inside the
poles at ±π/2 (~1.5708)". It has no `_OP_SINGULARITIES` entry and no
`_SFPU_UNDEFINED_RANGES` entry, so `boundary_probes()` produces nothing and the poles are
never approached. Contrast `Lgamma`, `Digamma` and `Polygamma`, whose poles are also
unprobed but *deliberately*, with a recorded reason ("their kernels are polynomial/LUT fits
that only claim accuracy well inside a positive domain … a probe at their boundary tests a
value the kernel never promised"). Tan is in the first category by accident, not by
decision.

**`sin` / `cos`.** Capped at exactly `[-π, π]` in every consumer: the main sweep
(`_OP_DOMAIN_REGISTRY`), `accuracy/test_sfpu_accuracy.py`, and
`quasar/test_eltwise_unary_sfpu_quasar.py`. Argument reduction for `|x| > π` is therefore
completely unexercised. If the kernel performs reduction, nothing tests it; if it does
not, nothing records that limitation.

### Steps

1. **Settle the contract before writing a probe.** For each of the three ops, read the
   kernel and answer: what range does it claim? For `tan`, is `±π/2` inside the promise or
   outside it? For `sin`/`cos`, is there a reduction step at all, and over what range is it
   accurate?

2. **Then take one of two branches per op — never neither.**

   - *Inside the promise:* register it. For tan,
     `MathOperation.Tan: {Operand.A: ((-math.pi/2, _BOTH), (math.pi/2, _BOTH))}` in
     `_OP_SINGULARITIES` is all it takes; the edge sweep picks the op up automatically
     because enrolment is derived. Note `edge_values()` does not clip to the registered
     random domain, so the probe will be driven even though `±1.5708` is outside
     `uniform(-1.3, 1.3)` — which is correct and is the point.
   - *Outside the promise:* record it, in the same shape as the gamma family's registry
     comment, so the next reader sees a decision rather than an omission.

3. **For `sin`/`cos`, add a dedicated range-reduction test rather than widening the
   registry domain.** Widening the registry entry would change the main sweep's tolerance
   profile and the accuracy plots, for a property that deserves its own assertion:

   ```python
   @pytest.mark.nightly
   @parametrize(
       formats=input_output_formats([DataFormat.Float32], same=True),
       mathop=[MathOperation.Sin, MathOperation.Cos],
       decade=[1, 2, 3],          # |x| up to 10, 100, 1000
   )
   def test_trig_argument_reduction(formats, mathop, decade):
       """sin/cos outside [-pi, pi]. Every probe is exactly representable in the input
       format, and the multiples of pi/2 are approached from both sides, so a reduction
       that loses low bits shows up as a phase error rather than as noise."""
   ```

   Expect accuracy to degrade with the decade — that is the finding, and the tolerance
   should be per-decade rather than one loose number that hides the trend.

**Cost:** 6 new ELFs for the trig test; the tan change is metadata only.

---

## What stays uncovered, and why

Not everything above closes. These are the cases where the honest outcome is a recorded
limitation, not a test:

- **`INT32_MIN` through any int32 kernel.** Dst stores integers sign-magnitude and reads
  `0x80000000` as "negative zero", so the value cannot round-trip. Covered by a dedicated
  `xfail` (`test_sfpu_binary_int_shift_int32_min_unsupported`) and by `integer_specials()`'s
  docstring. `INT32_MIN + 1` and `2**31 + 1` stand in for it on the signed and unsigned sides.
- **`Lgamma` / `Digamma` / `Polygamma` at their poles.** The kernels are polynomial and LUT
  fits that claim accuracy only well inside a positive domain; a probe at the boundary
  tests a value the kernel never promised. Already recorded above `_OP_SINGULARITIES`.
- **A zero's sign in the *result*.** `passed_test()` judges by `torch.isclose`, a both-NaN
  clause and PCC, and `-0.0 == +0.0` under all three. Asserting a result's zero sign needs
  a bitwise comparator, which is a suite-wide change and out of scope here. This is what bounds
  W1's signed-zero work: it drives a `-0.0` *into* a pole, and cannot check the sign of a zero
  that comes back out.
- **A generated NaN's sign on Wormhole.** `SFPMAD.md` says it "might or might not be set";
  `generated_nan_sign_is_asserted()` already gates it per lane. Not a gap.
