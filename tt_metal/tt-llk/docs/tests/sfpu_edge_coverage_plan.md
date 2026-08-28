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
| [W7](#w7--the-overflow-producing-ops-are-not-driven-at-their-ceiling) | Nothing asserts that a result too large for the output format saturates rather than wraps | unary, binary | M | High | — |
| [W9](#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | unary | M | Medium | needs a kernel-contract ruling |
| [W10](#w10--block-float-inputs-never-see-a-mixed-magnitude-block) | Bfp8_b/Bfp4_b blocks always uniform-magnitude | all | M | Medium | — |
| [W11](#w11--a-coverage-ledger-so-the-next-gap-is-visible) | No machine-checked record of which value classes each op has seen | infra | M | High | the items above |

Suggested order: **W1** (small, unblocked, independently mergeable), then **W11**, then
**W7 → W9 → W10** (each needs a measurement pass, and W9 a kernel-contract ruling).

**Already landed**, in the commit this document arrives with: the whole of the original W5
(IEEE specials for the ternary family, including the `TernarySFPUGolden` and `WhereGolden`
Dest/pack modelling that blocked it), and all of the original W7 except its overflow half —
`format_extremes()`, `extremes_safe()`, `subnormal_delivered()`, the `extremes=` axis on
`edge_values()`/`edge_spec()`, `EXTREMES_READY_OPS` with its first tranche enrolled, and one
saturation test. Also the whole of W2 — `StimuliSpec.cycle`, honoured by `CustomStrategy`,
on by default in `edge_spec()` — the whole of W4, whose cat-B half the ternary specials work
had already closed, the whole of W6, and the whole of W8 — which also retired the "effectively
unary" justification that kept `SfpuLogsigmoid` out of cat B, replacing it with the structural
one, and the whole of W3 — including the negatives it expected to be blocked, which
`twos_complement=True` turns out to deliver. W7 below is what is left of it. The numbering is
unchanged so
that references from commit messages and reviews still resolve; the closed items are simply
gone.

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
[-0.015625, 0.0, 0.015625]      # no -0.0
```

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

## W7 — The overflow-producing ops are not driven at their ceiling

### Problem

The cat-F machinery now exists and its first tranche is enrolled, but that tranche is
deliberately the ops that *cannot* overflow: `Abs`, `Neg`, `Sign`, `Signbit`, the rounding
family, `Identity`, `Fill` and the four sweep-reachable comparisons all return their input, a
bounded constant, or a magnitude no larger than the input. So `test_eltwise_unary_sfpu_extremes`
asserts that the pipeline *delivers* a magnitude extreme, and says nothing about what happens
when a result leaves the format.

That second half is the one with a silent failure mode. The convert from the SFPU's fp32 to a
narrower output must saturate to ±inf. If it ever wrapped instead, ±inf would still come out
right for a non-finite *input* — so every cat-B probe would keep passing — while every large
finite input silently returned a tiny wrong value, and no random sweep would reach it, because
the widest registered domain in `_OP_DOMAIN_REGISTRY` is `Square` at ±1000.

`test_eltwise_unary_sfpu_square_saturation` is the one op that has this today, and it is the
template. Still unwritten: `Exp`, `Exp2`, `ExpWithBase`, `Expm1`, `Sinh`, `Cosh`, and on the
binary side `SfpuElwpow`, `SfpuElwmul` and `SfpuElwadd`.

### What is already there to build on

Landed with this document, so none of it needs rebuilding:

- `format_extremes(fmt)` — the ceiling, the largest step below it, the smallest normal and one
  subnormal, both signs, each rounded onto the format's own grid so nothing quantizes on the way
  in. `_FORMAT_MIN_NORMAL` alongside it, sourced from the same `torch.finfo` call
  `golden_generators._FTZ_THRESHOLD` uses.
- `extremes_safe(input, output, dest_acc)` and the `extremes=` axis on `edge_values()` /
  `edge_spec()`, kept separate from `specials` because the delivery rules and the failure
  classes differ.
- `extreme_values(input, output, dest_acc)` — cat F alone, for a sweep that wants one failure
  class per variant.
- `subnormal_delivered(input, dest_acc)` — measured: the datacopy path hands the kernel `+0.0`,
  so the subnormal probe is only sent on a 32-bit input at `dest_acc=Yes`.
- `EXTREMES_READY_OPS`, opt-in per op, and `test_eltwise_unary_sfpu_extremes` reading it.

### Steps

1. **One op per PR, and measure before enrolling.** Driving an op at its ceiling on a golden
   that does not model saturation produces a wall of failures with one root cause, which is
   indistinguishable from no measurement at all. `UnarySFPUGolden._square` is the shape that
   works: a `isfinite(result)` test routed through `handle_infinite_numbers()`, which already
   knows that a B-exponent format gets ±inf and Float16 gets NaN. Check the op's golden has an
   equivalent before writing the probe list.

2. **Choose probes that are exactly representable, and unambiguous across the format axis.**
   `_SQUARE_SATURATION_MAGNITUDES` is powers of two for this reason: a decimal written near a
   threshold is pinned to a value other than the one it names (`88.7` is `88.5` in bfloat16).
   The second trap is subtler — `_FORMAT_MAX_MAGNITUDE` falls back to bfloat16's ceiling for
   every format at least that wide, so a probe landing between bfloat16's maximum and fp32's is
   finite on a Float32 output and infinite on a Float16_b one, and the variant then measures the
   output format rather than the kernel. Pick probes clear of that band on both sides.

   For the exp family the natural grid is the exponent itself: `exp2(127)` is finite in both,
   `exp2(129)` overflows both. For `Sinh`/`Cosh` the bfloat16 grid near the threshold is spaced
   0.5, so `89.0` (finite) and `90.0` (overflows both) are exact and unambiguous.

3. **Assert the straddle host-side.** `_assert_square_probes_straddle_the_ceiling()` is the
   guard to copy: without it a probe list stays plausible while the ceiling it was chosen to
   straddle moves, and the variant passes either way — asserting ordinary arithmetic if every
   probe went finite, or saturation with no control if every probe went infinite.

4. **The binary ops need the pair, not the value.** `SfpuElwmul` and `SfpuElwadd` overflow as a
   function of *both* operands, so their probe is a pair list through `edge_pair_values()` and
   an `_EDGE_CLASSES` member of its own rather than a `spec_A`. Do these after two unary ones,
   not before.

5. **Generalise last, not first.** Once two or three ops have their own saturation test the
   shared shape will be obvious; derived before then it fits the op it was written against and
   nothing else. This is the step that closes W7.

### Pin it

Each op's straddle assertion runs at collection, as `_assert_square_probes_straddle_the_ceiling`
does, so a format-table change fails in every lane with no hardware.

### Verify

```bash
pytest test_sfpu_domains.py -q
CHIP_ARCH=blackhole pytest test_eltwise_unary_sfpu.py -q -m nightly -k saturation
```

**Cost:** small per op, and the number of ops is the whole cost. Budget one PR each.

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

## W10 — Block-float inputs never see a mixed-magnitude block

### Problem

Bfp8_b, Bfp4_b and Bfp2_b share one exponent across each 16-element block, so the
stimulus that actually exercises the format is a block holding one large element and
fifteen small ones — where the small values quantize hard, or to zero. Every current
stimulus is a narrow-range uniform or gaussian, so the shared exponent never bites, and
`_bfp_block_aware_compare`'s lattice fallback in `helpers/utils.py` is never stressed.

The edge sweeps cannot reach this at all: their format axis is `Float16_b` / `Float32`,
and block-float inputs are excluded from cat B on the (correct) grounds that
`quantize_input_to_unpack_format()` destroys a NaN.

### Steps

1. **Add a distribution.** A `DistributionKind.BLOCK_SPREAD` (or a face callable, if a new
   enum member is too heavy) writing, per 16-element block, one element at
   `high` and fifteen log-spaced down to `high * 2**-k`. Parameterise `k` so the test can
   walk from "all values keep full mantissa" to "the small ones flush".

2. **Drive it on the ops already swept at Bfp8_b** — the `BROAD_SWEEP_OPS` list in
   `test_sfpu_unary.py` and `SfpuAddcmul` on the ternary side — as a separate nightly
   variant, not as a replacement for the existing uniform stimuli.

3. **Check the golden first.** The BFP helpers in `golden_generators.py` no longer FTZ
   internally and funnel through the centralised `_apply_ftz`; confirm the host-side
   quantization models a mixed block the same way the unpacker does before treating a
   mismatch as a kernel finding.

4. **Expect the verdict to come from the lattice check, not the atol.** `passed_test`
   already has a block-aware path for exactly this; this work item is what makes it earn
   its keep.

**Cost:** one distribution + one nightly variant per family.

---

## W11 — A coverage ledger, so the next gap is visible

### Problem

Every gap in this document was found by reading code and running the registries by hand.
Nothing in the suite states, per op, *which classes of input value it has actually seen* —
so an op can sit for a release with a positive-only uniform and look fully covered.

The suite already has the ingredients: `SPECIALS_READY_OPS` and
`_BINARY_SPECIALS_NOT_READY` are exactly this kind of ledger for one class, and the
`_assert_domain_sets_consistent()` / `_classify_stimuli_source()` pair is exactly this
kind of totality check for another. The missing piece is generalising them across classes.

### Steps

1. **Name the classes.** The four the framework already has, plus the ones this plan adds:

   | Class | Meaning | Source |
   |---|---|---|
   | A | domain singularities, straddled | `_OP_SINGULARITIES` |
   | B | IEEE specials (±inf, NaN, ±0) | `format_specials()` + `specials_safe()` |
   | C | integer extremes | `integer_specials()` |
   | D | knees, thresholds, rounding ties | `_OP_EDGE_POINTS` |
   | E | operand-as-parameter (shift amounts, scalars) | `SHIFT_EDGE_AMOUNTS`, W6 |
   | F | finite magnitude extremes and subnormals | `format_extremes()` + `extremes_safe()` |
   | G | signed zero at a pole | W1 |

2. **Add `_OP_COVERAGE_LEDGER: Dict[MathOperation, Dict[EdgeClass, str]]`** to
   `sfpu_domains.py`, where the value is either `COVERED` (derived, not asserted — see
   step 3) or a reason string explaining why the class does not apply or is not yet
   reachable.

3. **Derive the `COVERED` half rather than declaring it.** For each op and class, ask the
   existing machinery: does `edge_values(op, fmt, ..., operand=o)` actually emit a value of
   that class? That way the ledger cannot claim coverage the sweep does not deliver — which
   is the failure mode a hand-maintained matrix has.

4. **Add one `test_sfpu_domains.py` test asserting totality:** every op in
   `sfpu_unary_ops() | _SFPU_BINARY_OPS | _SFPU_TERNARY_OPS` has an entry for every class,
   either derived-covered or explained. A new op with no entry fails at collection, in
   every lane, with no hardware.

5. **Emit it as a report.** A `--sfpu-coverage-report` flag (or a plain
   `python -m helpers.sfpu_domains --report`) printing the matrix makes the remaining gaps
   a one-line query instead of a code-reading exercise.

**Cost:** medium, and it is the item that stops this document needing to be written again.

---

## What stays uncovered, and why

Not everything above closes. These are the cases where the honest outcome is a recorded
limitation, not a test:

- **`INT32_MIN` through any int32 kernel.** Dst stores integers sign-magnitude and reads
  `0x80000000` as "negative zero", so the value cannot round-trip. Covered by a dedicated
  `xfail` (`test_sfpu_binary_int_shift_int32_min_unsupported`) and by `integer_specials()`'s
  docstring. `INT32_MIN + 1` and `2**31 + 1` stand in for it on the signed and unsigned sides.
- **Negative integer operands generally** — *retracted*. This entry expected the
  sign-magnitude Dst to block them, and `twos_complement=True` turns out to deliver them
  intact: `test_eltwise_binary_sfpu_int_signed_division` drives both signs on both operands and
  passes, which is what separates truncating from flooring division. Only the single
  `0x80000000` pattern above is genuinely blocked.
- **`Lgamma` / `Digamma` / `Polygamma` at their poles.** The kernels are polynomial and LUT
  fits that claim accuracy only well inside a positive domain; a probe at the boundary
  tests a value the kernel never promised. Already recorded above `_OP_SINGULARITIES`.
- **A zero's sign in the *result*.** `passed_test()` judges by `torch.isclose`, a both-NaN
  clause and PCC, and `-0.0 == +0.0` under all three. Asserting a result's zero sign needs
  a bitwise comparator, which is a suite-wide change and out of scope here. Note this bounds
  W1: it fixes the *input* side of signed zero, not the output side.
- **A generated NaN's sign on Wormhole.** `SFPMAD.md` says it "might or might not be set";
  `generated_nan_sign_is_asserted()` already gates it per lane. Not a gap.
