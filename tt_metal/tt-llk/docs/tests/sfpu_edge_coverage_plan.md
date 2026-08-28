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
| [W1](#w1--signed-zero-at-a-registered-pole) | `-0.0` never reaches a pole operand (`div(x, -0.0)`, `atan2(y, -0.0)`) | binary, ternary | S | High | — |
| [W9](#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | unary | M | Medium | needs a kernel-contract ruling |

Two items left. **W1** is small, unblocked and independently mergeable. **W9** needs a
kernel-contract ruling before any of it can be written.

`python -m helpers.sfpu_domains --report` prints the coverage ledger, which is the fastest way
to see where each moves the needle: W1 is cat G, which stands at 0 covered of the 14 ops that
have a zero pole to deliver one to, and W9 is cat A.

The numbering is not contiguous: W2–W8, W10 and W11 are closed and their sections have been
removed. What remains keeps its original number so that references from commit messages and
reviews still resolve.

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

## W1 — Signed zero at a registered pole

### Problem

`boundary_probes()` emits only `+0.0` at a singularity registered at `0.0`. The other
zero arrives only through `format_specials()`, which is gated behind
`BINARY_SPECIALS_READY_OPS` — and **every op with a zero pole is in
`_BINARY_SPECIALS_NOT_READY`**: `SfpuElwdiv`, `SfpuBinaryFmod`, `SfpuBinaryRemainder`,
`SfpuXlogy`, `SfpuElwpow`, `SfpuAtan2`. The result is that `div(x, -0.0)` — which must be
`∓inf`, the opposite sign from `div(x, +0.0)` — is never driven anywhere in the suite.
Same for `atan2(+0, -0.0)` (= π, against `0` for `+0.0`) and for `addcdiv` / `snake_beta`
on operand C.

This is cleanly separable from the cat-B question: `-0.0` is *finite*, so the
`specials_safe()` gate (which is about non-finites surviving unpack) does not apply. The
right gate already exists — `negative_zero_delivered(input_format, dest_acc)` — and
`edge_values()` already uses it for the cat-D zero knees.

### Evidence

```python
>>> edge_values(MathOperation.SfpuElwdiv, DataFormat.Float32, DataFormat.Float32,
...             operand=Operand.B, dest_acc=DestAccumulation.Yes)
[-2.384185791015625e-07, 0.0, 2.384185791015625e-07]      # no -0.0
```

The two straddling probes are one fp32 ULP either side of the pole; they were an order of
magnitude wider when this was first written, before the step became `dest_acc`-aware. Neither
of them is the missing value — the gap is that the zero in the middle only ever has one sign.

### Steps

1. **`helpers/sfpu_domains.py` — `boundary_probes()`.** In the `_OP_SINGULARITIES`
   branch, when the point is a zero, append the other zero alongside it:

   ```python
   for point, side in singularities:
       probes.append(point)
       if point == 0.0:
           # 1/+0 = +inf against 1/-0 = -inf, so a zero pole is two probes, not one.
           # _dedup_representable() already keys zeros by sign rather than by value, so
           # both survive. Emitted unconditionally here; edge_values() drops the negative
           # one on the pipelines that flatten it.
           probes.append(math.copysign(0.0, -math.copysign(1.0, point)))
       ...
   ```

   `math.copysign` rather than a `-0.0` literal because the registered point is only
   *conventionally* `+0.0`; deriving the opposite sign keeps this correct if an entry is
   ever written as `-0.0`.

2. **`helpers/sfpu_domains.py` — `edge_values()`.** The negative-zero filter is currently
   applied twice, to `edge_points` (cat D) and to `injected` (cat B), and *not* to the
   cat-A probes. Consolidate it into one pass over the combined list so a new source
   cannot miss it:

   ```python
   vals = list(boundary_probes(...)) + list(op_edge_points(op, operand))
   if specials:
       vals += list(format_specials(range_fmt))
   if not range_fmt.is_integer() and not negative_zero_delivered(input_format, dest_acc):
       # The datacopy path hands the kernel +0.0, so a -0.0 probe would blame the kernel
       # for a datum it never received. Applies to every source, not just cat B and cat D.
       vals = [v for v in vals if not _is_negative_zero(v)]
   return _dedup_representable(clip_to_format(vals, range_fmt), range_fmt)
   ```

   Note this consolidation is a behaviour change for cat A only; cat B and cat D end up
   filtered exactly as before. `edge_values()` has since grown a cat-F branch as well
   (`extremes=`, gated by `subnormal_delivered()`); fold that into the same single pass rather
   than leaving a third filter beside it, which is the shape this step exists to prevent.

3. **No change needed in `test_sfpu_binary.py`.** `_classify_edge_pair()` already handles
   the new pairs correctly: `(0.0, -0.0)` lands in `_EDGE_CLASS_BOTH_ZERO` (because
   `-0.0 == 0.0`), and `(2.0, -0.0)` lands in `_EDGE_CLASS_ORDINARY` (finite operands, a
   `-inf` answer). The pair count grows by roughly the size of the counterpart spread.

4. **No change needed in `test_sfpu_ternary.py`** either — `test_sfpu_ternary_operand_edges`
   reads `edge_values(..., operand=Operand.C)`, so it picks the new probe up for `addcdiv` and
   `snake_beta` automatically. It lands in the `pole` class, `-0.0` being finite.

### Pin it

Add to `test_sfpu_domains.py`, next to the existing `test_zero_pole_probes_are_not_loosened`:

```python
@pytest.mark.parametrize("op,operand", [
    (MathOperation.SfpuElwdiv,   Operand.B),
    (MathOperation.SfpuAtan2,    Operand.B),
    (MathOperation.Reciprocal,   Operand.A),
    (MathOperation.SfpuAddcdiv,  Operand.C),
])
def test_zero_poles_are_probed_with_both_signs(op, operand):
    """A zero pole is two probes, not one: 1/+0 and 1/-0 differ in the result's sign."""
    vals = edge_values(op, DataFormat.Float32, DataFormat.Float32, operand=operand,
                       dest_acc=DestAccumulation.Yes)
    signs = {math.copysign(1.0, v) for v in vals if v == 0.0}
    assert signs == {1.0, -1.0}, f"{op.name} operand {operand.name} probes {vals}"


def test_negative_zero_pole_probe_is_dropped_where_it_cannot_be_delivered():
    """Not sent on the datacopy path — the LREG holds +0.0 there, so the probe is vacuous."""
    vals = edge_values(MathOperation.SfpuElwdiv, DataFormat.Float16_b, DataFormat.Float32,
                       operand=Operand.B, dest_acc=DestAccumulation.Yes)
    assert not any(_is_negative_zero(v) for v in vals)
```

### Verify

```bash
pytest test_sfpu_domains.py -q
CHIP_ARCH=blackhole pytest test_sfpu_binary.py -q -m nightly -k "edges and (div or fmod or remainder or atan2 or xlogy or pow)"
CHIP_ARCH=blackhole pytest test_sfpu_ternary.py -q -m nightly -k edges
```

### Expect

New failures are the point. Triage each against `tt-isa-documentation`:
`SFPMAD.md`'s "flushed to positive zero" (Wormhole) vs "sign-preserved zero" (Blackhole)
already explains the existing `_EDGE_CLASS_NEGATIVE_ZERO` xfails, and a `-0.0` *input* at
a pole is likely to land in the same bucket. Record whatever happens per
`(input, output, dest_acc)` in `_BINARY_EDGE_COMBINATIONS` with a reason, per convention 4.
If Blackhole agrees and Wormhole does not, add the class to
`_WORMHOLE_ONLY_EDGE_CLASSES` so Blackhole *asserts* it.

**Cost:** no new ELFs (the pair list is a runtime axis). A few more elements per override
tensor.

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
  a bitwise comparator, which is a suite-wide change and out of scope here. Note this bounds
  W1: it fixes the *input* side of signed zero, not the output side.
- **A generated NaN's sign on Wormhole.** `SFPMAD.md` says it "might or might not be set";
  `generated_nan_sign_is_asserted()` already gates it per lane. Not a gap.
