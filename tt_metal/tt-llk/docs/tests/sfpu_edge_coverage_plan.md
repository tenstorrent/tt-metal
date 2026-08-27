<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# SFPU input-value coverage: what is left, and how to add it

Scope: the *input values* driven into the SFPU element-wise suites —
`tests/python_tests/test_sfpu_unary.py`, `test_sfpu_binary.py`, `test_sfpu_ternary.py`
and the shared metadata in `tests/python_tests/helpers/sfpu_domains.py`.

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
| [W2](#w2--edge-probes-occupy-ten-lanes-out-of-a-tile) | Unary edge probes fill ~4–10 of 256 elements per face; rest is `0.0` | unary | S | High | — |
| [W3](#w3--integer-binary-ops-never-see-zero-negatives-or-the-uint32-upper-half) | Int binary ops: no `0`, no negatives, uint32 capped at 1e6 | binary | M | High | partly HW (sign-magnitude Dst) |
| [W4](#w4--ttnn_wheremixed-is-not-mixed) | `test_ttnn_where[mixed]` is all-true on Float32 | ternary | S | High | — |
| [W5](#w5--no-ieee-specials-reach-any-ternary-op) | Ternary family has no cat B at all | ternary | M | High | golden work |
| [W6](#w6--the-ternary-scalar-is-hardcoded-to-20) | `SFPU_TERNARY_SCALAR` never varies | ternary | S | Medium | — |
| [W7](#w7--nothing-drives-format-extremes-or-subnormals) | No probe at ±format-max, min-normal, or in the subnormal band | all | L | High | golden work |
| [W8](#w8--logsigmoids-x--4-branch-is-never-driven) | `SfpuLogsigmoid` never executes the only branch that reads operand B | binary | S | Medium | — |
| [W9](#w9--tan-has-no-registered-pole-sincos-never-exceed-π) | `Tan` has no pole entry; `sin`/`cos` capped at ±π | unary | M | Medium | needs a kernel-contract ruling |
| [W10](#w10--block-float-inputs-never-see-a-mixed-magnitude-block) | Bfp8_b/Bfp4_b blocks always uniform-magnitude | all | M | Medium | — |
| [W11](#w11--a-coverage-ledger-so-the-next-gap-is-visible) | No machine-checked record of which value classes each op has seen | infra | M | High | W1–W10 |

Suggested order: **W1 → W2 → W4 → W6 → W8** (small, unblocked, each independently
mergeable), then **W3 → W5 → W11**, then **W7 → W9 → W10** (each needs a measurement
pass and probably a golden change).

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
   filtered exactly as before.

3. **No change needed in `test_sfpu_binary.py`.** `_classify_edge_pair()` already handles
   the new pairs correctly: `(0.0, -0.0)` lands in `_EDGE_CLASS_BOTH_ZERO` (because
   `-0.0 == 0.0`), and `(2.0, -0.0)` lands in `_EDGE_CLASS_ORDINARY` (finite operands, a
   `-inf` answer). The pair count grows by roughly the size of the counterpart spread.

4. **No change needed in `test_sfpu_ternary.py`** either — `edge_spec(..., operand=Operand.C)`
   picks the new probe up for `addcdiv` and `snake_beta` automatically.

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

## W2 — Edge probes occupy ten lanes out of a tile

### Problem

`edge_spec()` returns `StimuliSpec.custom(values=vals)`, and `CustomStrategy.generate_face`
writes those values at the head of each 256-element face and **zero-fills the remaining
~250**. Measured across the 95 swept unary ops, the median edge list is **4 values** and
the longest is 10 (`Round`).

Two consequences:

- Probe values only ever occupy lanes 0–9 of the first vector operation in each face. A
  lane-position-dependent defect at an edge value is invisible.
- The tolerance verdict is computed over a tensor that is ~96% `0.0`. PCC and any
  aggregate statistic are dominated by a value that is not the probe.

There is also an accidental side effect worth deciding about deliberately: for ops whose
domain excludes zero (`Acosh`, `Log`, `Rsqrt`), the zero fill is silently driving an
out-of-domain input on every edge variant, and nothing records that it is being tested.

The binary suite already solved this. `_build_paired_tile_override()` in
`test_sfpu_binary.py` **cycles** the pair list to fill a whole tile, "so the override
divides evenly into whatever buffer the driver picks and every element is a pair the
caller meant to drive". The unary and ternary sides should match.

### Steps

1. **`helpers/stimuli_generator/spec.py`.** Add a field to `StimuliSpec`:

   ```python
   cycle: bool = False   # "custom" only: repeat *values* to fill the face instead of
                         # zero-filling the remainder.
   ```

   Document it under the `"custom"` entry of the class docstring, next to the existing
   "Values are not repeated." sentence — which becomes "…unless `cycle=True`."

2. **`helpers/stimuli_generator/strategies/structured.py` — `CustomStrategy.generate_face`.**
   Honour it, and drop the `len(values) > size` error only in the cycling case (cycling a
   long list is well defined):

   ```python
   if spec.cycle:
       reps = -(-size // len(vals))            # ceil-div
       tensor = torch.tensor((vals * reps)[:size], dtype=dtype)
   else:
       tensor = torch.zeros(size, dtype=dtype)
       tensor[: len(vals)] = torch.tensor(vals, dtype=dtype)
   return tensor
   ```

3. **`helpers/sfpu_domains.py` — `edge_spec()`.** Pass `cycle=True`, and update the
   docstring paragraph that currently justifies the zero fill ("a face is far larger than
   these lists, and 0.0 is itself a useful probe"). The replacement rationale: `0.0` is
   already a registered pole or knee wherever it is meaningful, and cycling both spreads
   the probes across every lane and stops the verdict being computed mostly over a
   filler value.

4. **Re-baseline.** Ops whose domain excludes zero will stop receiving the accidental
   `0.0`. If any of them *loses* coverage you care about, that value belongs in
   `_OP_SINGULARITIES` or `_OP_EDGE_POINTS` explicitly — which is the right place for it.

### Pin it

```python
def test_edge_spec_cycles_probes_across_the_whole_face():
    spec = edge_spec(MathOperation.Reciprocal, DataFormat.Float32, DataFormat.Float32,
                     dest_acc=DestAccumulation.Yes)
    assert spec.cycle, "edge probes must fill the face; a zero-filled tail makes the " \
                       "verdict a statement about 0.0, not about the probe"
```

and a strategy-level test that a 4-element list produces no zeros in a 256-element face
unless `0.0` is one of the four.

### Verify

```bash
pytest test_sfpu_domains.py -q
CHIP_ARCH=blackhole pytest test_sfpu_unary.py -q -m nightly -k edges
```

### Expect

Some variants that passed on the strength of a mostly-zero tensor may now fail. That is
the gap closing, not a regression — triage per convention 3.

**Cost:** zero. Same tensor size, same ELFs.

---

## W3 — Integer binary ops never see zero, negatives, or the uint32 upper half

### Problem

`_INT_BINARY_STIMULI` (`test_sfpu_binary.py:1126`) gives every integer binary op a single
positive uniform range, all of them `low >= 1.0` except max/min at `0.0`, and all far
below the format ceiling. `test_sfpu_binary_int_extremes` covers only the bitwise ops,
`eq`/`ne`, and the four ordered comparisons — so the *arithmetic* int ops get nothing else.

Three distinct holes:

- **`SfpuDivInt32` and `SfpuDivInt32Floor` are indistinguishable as tested.** Truncating
  and flooring division differ *only* on negative operands. The comment in the table even
  says so ("trunc == floor"). The entire reason the second op exists is unexercised.
- **Zero operands are never driven** for `gcd`, `lcm`, `div`, `remainder`, `fmod`, `mul`.
  `gcd(0, x) = x` and `lcm(0, x) = 0` are the identities those kernels most plausibly get
  wrong, and `x / 0` / `x % 0` are undriven poles.
- **`SfpuMaxUint32`, `SfpuMinUint32`, `SfpuRemainderUint32` are capped at 1e6**, so no
  operand ever lands in `[2**31, 2**32)` — the only region where an unsigned op differs
  from its signed twin. The generator's own UInt32 default is `uniform(0, 2**32 - 2)`
  (`helpers/stimuli_generator/generator.py:253`); the override throws that away.

Part of this is genuinely blocked: Dst stores int32 sign-magnitude, so a negative operand
does not round-trip, which is why the table is positive-only. That blocks the negatives.
It does **not** block zero, and it does not block the uint32 upper half.

### Steps

1. **Split the table's two jobs.** `_INT_BINARY_STIMULI` currently conflates "the range
   this kernel is documented to be valid on" with "the values we happen to drive".
   Restructure the value side as an explicit list per op so zero can be included where the
   kernel defines it:

   ```python
   # (low, high) for the random bulk, plus the discrete values that must appear.
   _INT_BINARY_STIMULI = {
       MathOperation.SfpuGcd: _IntStimuli(low=1.0, high=100_000.0, must_include=[0, 1]),
       MathOperation.SfpuLcm: _IntStimuli(low=1.0, high=20_000.0,  must_include=[0, 1]),
       ...
   }
   ```

   Deliver `must_include` the same way `_int_unary_stimuli_spec` does — a
   `StimuliSpec.custom(values=straddle + spread, cycle=True)` (see W2), not a second
   tensor.

2. **Add `test_sfpu_binary_int_zero_operands`** (nightly), driving the cartesian product
   of `{0, 1, 2, small}` against itself for the ops whose kernels define an answer at
   zero, via `_build_paired_tile_override`. For the *divisors* — `div`, `divfloor`,
   `remainder`, `fmod` — a zero divisor needs a decision first: read the kernel, decide
   whether it is UB or a defined answer, and either register it or record it as excluded
   with the reason. Do not drive it blind.

3. **Add `test_sfpu_binary_uint32_high_range`** (nightly), for the three uint32 ops, with
   `spec = StimuliSpec.uniform(intervals=[(0.0, 1e6), (2.0**31, 2.0**32 - 2)])` and
   `twos_complement=True`. This is the variant that can tell `MaxUint32` from `MaxInt32`;
   until it exists, nothing can.

4. **Add `test_sfpu_binary_int_signed_division`** (nightly, `xfail` where appropriate) for
   `SfpuDivInt32` vs `SfpuDivInt32Floor` over `{-7, -1, 1, 7} × {-3, -1, 1, 3}` with
   `twos_complement=True`. Expect the sign-magnitude Dst limitation to bite; the point is
   to have it recorded as a *specific* `xfail` with the ISA reference — the same shape as
   `test_sfpu_binary_int_shift_int32_min_unsupported` — rather than as an absence.

### Pin it

In `test_sfpu_domains.py`, assert every op in `_INT_BINARY_STIMULI` either includes `0`
in its driven values or appears in a new `_INT_ZERO_UNDEFINED: Dict[MathOperation, str]`
with a reason. That converts "nobody got to it" into "here is why not".

### Verify

```bash
CHIP_ARCH=blackhole pytest test_sfpu_binary.py -q -k "int_zero or uint32_high or int_signed"
```

**Cost:** three new nightly tests, ~30 new ELFs.

---

## W4 — `test_ttnn_where[mixed]` is not mixed

### Problem

`test_sfpu_ternary.py:318` draws the condition tensor from `StimuliSpec.uniform(0.0, 1.0)`,
and `WhereGolden` selects on `cond != 0.0`. Measured over 4096 elements at seed 0:

| format | exact zeros in the condition |
|---|---|
| Float32 | **0** |
| Float16_b | 20 (0.5%, and only because bf16 rounds small draws to zero) |
| Int32 | 2090 (genuinely ~50/50 — the integer path narrows `uniform(0,1)` to `randint(0,2)`) |

So on Float32 the `mixed` case is bit-for-bit identical in coverage to `all_ones`: the
false branch is never taken. `test_ttnn_where_mcw`'s alternating pattern rescues the
family's overall coverage, but the variant whose entire purpose is the mixed case is not
testing it.

Separately, the condition operand never sees `-0.0`, NaN, or ±inf. This is exactly the
shape of bug the unary suite already documents for `Sign` and `Heaviside`: `SFPSETCC` is
specified only "provided that VC is neither negative zero nor any kind of NaN", and
`where` is a compare-to-zero on an arbitrary operand. `cond != 0.0` is `False` for `-0.0`
and `True` for NaN — both are answers the kernel is not promised to give.

### Steps

1. **Fix `mixed`.** Replace the condition spec with one that is mixed by construction on
   every format, not by accident of quantization:

   ```python
   _WHERE_MIXED_COND = StimuliSpec.uniform(intervals=[(0.0, 0.0), (0.5, 1.0)], seed=0)
   ```

   or, clearer, a face callable alternating `0.0` and a ramp — the same shape
   `_eq_ne_stimuli_specs()` uses on the binary side. Keep the true/false *value* tensors
   on their existing spec; only the condition changes.

2. **Assert the mix.** The failure mode this fixes is silent, so make it loud:

   ```python
   frac_true = float((src_A.flatten().to(torch.float32) != 0.0).float().mean())
   assert 0.2 < frac_true < 0.8, (
       f"the 'mixed' condition is {frac_true:.1%} true — this variant is a duplicate of "
       "all_ones/all_zeros and asserts nothing about the select"
   )
   ```

   Guard it on `test_case == "mixed"`.

3. **Add `test_ttnn_where_condition_specials`** (nightly). Condition tensor built from
   `[+0.0, -0.0, nan, inf, -inf, 1.0, -1.0, tiny]` cycled across the tile
   (`StimuliSpec.custom(..., cycle=True)` from W2); true/false tensors on the ordinary
   spec so every lane's expected answer is distinguishable. Gate on
   `specials_safe(input, output, dest_acc)` for the non-finites and on
   `negative_zero_delivered(input, dest_acc)` for `-0.0`, exactly as the unary edge sweep
   does — a probe that cannot be delivered must not be sent.

4. **Expect `-0.0` and NaN to diverge**, and record them as non-strict `xfail`s citing the
   same `SFPSETCC` clause as `_EDGE_DIVERGENCE_REASON[MathOperation.Sign]`. Cross-check
   against `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h` first — if the kernel
   loads raw LO16 and tests bits rather than going through `SFPSETCC`, the answer may be
   correct and the note should say so.

### Verify

```bash
CHIP_ARCH=blackhole pytest test_sfpu_ternary.py -q -k where
```

**Cost:** one new nightly test; `mixed` costs nothing extra.

---

## W5 — No IEEE specials reach any ternary op

### Problem

`_SFPU_TERNARY_OPS` is `frozenset()` (`sfpu_domains.py:1126`), no ternary op appears in
`SPECIALS_READY_OPS` or `BINARY_SPECIALS_READY_OPS`, and
`test_sfpu_ternary_edges` calls `edge_spec()` without a `specials=` argument — so it
defaults to `False`. The unary family had three enrolment tranches and the binary family
one; the ternary family has had none.

Combined with the fact that only operand C is probed at all, the entire deliberate-edge
coverage for the ternary family is:

```
SfpuAddcdiv    C: [-0.015625, 0.0, 0.015625]      A: []  B: []
SfpuLerp       C: [-1.0, 0.0, 1.0, 2.0]           A: []  B: []
SfpuSnakeBeta  C: [-0.015625, 0.0, 0.015625]      A: []  B: []
SfpuAddcmul    (nothing — the variant skips)
```

Three or four values on one operand, for a family of four ops.

### Steps

1. **Populate `_SFPU_TERNARY_OPS`** in `sfpu_domains.py` with the four ops plus
   `TTNNWhere`. It is currently an empty placeholder, so nothing that keys off "is this a
   ternary op" can work.

2. **Add `TERNARY_SPECIALS_READY_OPS: Dict[MathOperation, str]`**, mirroring
   `BINARY_SPECIALS_READY_OPS` exactly — same shape, same per-op reason string, and a
   companion `_TERNARY_SPECIALS_NOT_READY`, plus the assertion that the two sets are
   disjoint:

   ```python
   assert not (set(TERNARY_SPECIALS_READY_OPS) & set(_TERNARY_SPECIALS_NOT_READY)), \
       "an op cannot be both enrolled in cat B and recorded as not ready for it"
   ```

3. **Run the host-side tranche first, with no hardware.** For each of the four ops and
   each of the 5×5×5 special triples, call `TernarySFPUGolden` directly and check it
   *answers* rather than raising. The unary tranche found three goldens that raised on a
   non-finite (`math.acos`/`asin`/`tan`) and three whose finite-input guard misrouted a
   NaN; `TernarySFPUGolden` is plain torch arithmetic so it will probably survive, but
   confirm before spending device time.

4. **Extend `_run_sfpu_ternary` to take per-operand edge specs.** It already accepts
   `spec_A`/`spec_B`/`spec_C`; the missing piece is a sweep that varies *which* operand
   carries the edge:

   ```python
   @parametrize(
       formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
       dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
       mathop=_TERNARY_EDGE_OPS,
       operand=runtime([Operand.A, Operand.B, Operand.C]),
   )
   def test_sfpu_ternary_operand_edges(request, formats, dest_acc, mathop, operand):
       ...
   ```

   `operand` is a `runtime()` axis — it changes the tensor, not the kernel — so all three
   share one ELF.

5. **Pass `specials=` through.** Same two gates as everywhere else:

   ```python
   specials = mathop in TERNARY_SPECIALS_READY_OPS and specials_safe(
       formats.input_format, formats.output_format, dest_acc
   )
   spec = edge_spec(mathop, formats.input_format, formats.output_format,
                    operand=operand, specials=specials, dest_acc=dest_acc)
   ```

   Where `edge_spec` returns `None` for the probed operand (A and B for every ternary op
   today, until specials are on), fall back to `edge_counterparts()` rather than skipping
   — the point of probing A while C is at its pole is that the *pair* is the case.

6. **Give `addcmul` something.** It is the only ternary op allowed on Bfp8_b and it
   currently has no edge coverage at all. `a + value*b*c` has no pole, so its edges are
   cat B (specials), cat F (magnitude extremes — see W7), and cancellation: drive
   `a = -value*b*c` exactly, where the result must be zero and the sign of that zero is a
   real question given the `SFPMAD` flush-to-zero note. Add the cancellation triple as a
   `_OP_OPERAND_EDGE_POINTS` entry or as a purpose-built variant.

### Pin it

Extend the existing `test_every_float_binary_op_is_classified_for_cat_b` pattern to the
ternary family: every op in `_SFPU_TERNARY_OPS` must appear in exactly one of
`TERNARY_SPECIALS_READY_OPS` / `_TERNARY_SPECIALS_NOT_READY`.

### Verify

```bash
pytest test_sfpu_domains.py -q
CHIP_ARCH=blackhole pytest test_sfpu_ternary.py -q -m nightly
```

**Cost:** one new nightly test sharing ELFs with the existing edge sweep (the operand axis
is `runtime()`), plus one host-side tranche run.

---

## W6 — The ternary scalar is hardcoded to 2.0

### Problem

`_SCALAR_VALUE = 2.0` (`test_sfpu_ternary.py:42`) is the only multiplier `addcmul` and
`addcdiv` are ever given, in the main sweep, the edge sweep and the perf test. It reaches
the kernel as `constexpr std::uint32_t SFPU_TERNARY_SCALAR` — a compile-time constant, so
varying it is a compile-time axis, and it is currently not an axis at all.

`value = 0.0` makes both ops reduce to identity in `a`, which is a strong and very cheap
check that neither kernel is reading the wrong Dst tile. `value = 1.0` removes the
multiply. A negative value flips a sign the golden and kernel must agree on.

### Steps

1. **Leave the main sweep at 2.0.** Widening the scalar on the existing sweep multiplies
   its ELF count; the value is in a separate, narrow variant.

2. **Add `test_sfpu_ternary_scalar`** (nightly), scalar as a compile-time axis, format
   axis reduced to one column to keep the ELF count honest:

   ```python
   _SCALAR_PROBES = [0.0, 1.0, -2.0]   # identity-in-a, no-op multiply, sign flip

   @pytest.mark.nightly
   @parametrize(
       formats=input_output_formats([DataFormat.Float32], same=True),
       dest_acc=[DestAccumulation.Yes],
       mathop=[MathOperation.SfpuAddcmul, MathOperation.SfpuAddcdiv],
       scalar=_SCALAR_PROBES,
   )
   def test_sfpu_ternary_scalar(formats, dest_acc, mathop, scalar):
       ...
   ```

3. **Parameterise `_run_sfpu_ternary`** with a `scalar_bits` argument defaulting to
   `_SCALAR_VALUE_BITS`, so both the templates list and the golden call read the same
   value. They already both read the module constant; the change is to thread one
   argument through instead.

4. **Assert the identity explicitly for `scalar = 0.0`.** With `value = 0`, `addcmul` and
   `addcdiv` must both return `a` exactly, bit for bit, for *any* `b` and `c` — including
   `c = 0`, where `0 * (b/0)` is `0 * inf = NaN` in IEEE but the kernel may short-circuit.
   Decide which answer is intended before writing the assertion; the golden currently says
   NaN (`a + 0.0 * (b/0.0)`).

**Cost:** 6 new ELFs.

---

## W7 — Nothing drives format extremes or subnormals

### Problem

Dumping all 113 entries of `_OP_DOMAIN_REGISTRY`: the widest domain is `Square` at ±1000
and `CastFp32ToFp16a` at ±1e5. Everything else is ≤ ±100. `FLOAT_SPECIALS` is
`(inf, -inf, nan, 0.0, -0.0)`. So the input space jumps from ~10 straight to infinity,
and nothing occupies the region in between or immediately above zero:

- **No probe at ±format-max** (3.39e38 bf16, 65504 fp16, 448 e4m3) or one ULP below it.
- **No probe near min-normal, and none in the subnormal band.** The goldens model
  flush-to-zero carefully (`golden_generators.py:89`, `_FTZ_THRESHOLD` / `_apply_ftz`) but
  no test ever drives an input that triggers it. `_format_elem_min_magnitude()`
  (`stimuli_generator/utils.py:672`) exists specifically to keep random stimuli *away*
  from denormals, and no edge probe puts them back.
- **No probe where the *result* overflows the output format.**
  `test_exp_overflow_saturation.py` was hand-written for exactly this case, and its
  docstring says plainly: "If that convert ever wrapped instead, +inf would still come out
  right … while every large finite input silently returned a tiny wrong value — and no
  other test would notice." The same reasoning applies unchanged to `Square`, `Exp2`,
  `ExpWithBase`, `Expm1`, `Sinh`, `Cosh`, `SfpuElwpow`, `SfpuElwmul` and `SfpuElwadd`.

This is the largest gap on the list and the one most likely to need golden changes, which
is why it is scheduled late despite its value.

### Steps

1. **Add a magnitude table to `sfpu_domains.py`**, next to `_FORMAT_MAX_MAGNITUDE`:

   ```python
   # Smallest positive *normal* of each format, and whether it keeps subnormals below that.
   # fp32 and bf16 flush; fp16 keeps them — the same split _FTZ_THRESHOLD records, sourced
   # once so the probe and the golden cannot disagree about where the cliff is.
   _FORMAT_MIN_NORMAL: Dict[DataFormat, float] = {...}
   ```

   Source it from `torch.finfo(...).tiny` / `MX_FORMAT_MIN_MAGNITUDE` rather than
   literals, and cross-check it against `_FTZ_THRESHOLD` in a `test_sfpu_domains.py` test
   — a probe placed on the wrong side of the FTZ cliff tests nothing and looks like it
   tests everything.

2. **Add `format_extremes(fmt) -> Tuple[float, ...]`**, the cat-F counterpart to
   `format_specials()`:

   ```python
   def format_extremes(fmt: DataFormat) -> Tuple[float, ...]:
       """Finite magnitude extremes of *fmt*: the ceiling, the largest step below it, the
       smallest normal, and one subnormal (or the FTZ cliff for a flushing format)."""
   ```

3. **Add an `extremes: bool = False` parameter to `edge_values()` / `edge_spec()`**,
   parallel to `specials`, and a `_extremes_safe(...)` predicate. Do *not* fold it into
   `specials` — the delivery rules differ (a subnormal survives an unpack that destroys a
   NaN) and the failure classes are unrelated.

4. **Add `EXTREMES_READY_OPS: Dict[MathOperation, str]`**, opt-in per op, for the same
   reason `SPECIALS_READY_OPS` is opt-in: driving `exp(3.39e38)` on an op whose golden
   does not model saturation produces a wall of failures with one root cause.

5. **Enrol in tranches, cheapest first.** Start with the ops whose behaviour at an extreme
   is uncontroversial and whose golden is plain arithmetic: `Abs`, `Neg`, `Sign`,
   `Signbit`, `Floor`, `Ceil`, `Trunc`, `Round`, `Identity`, `Fill`, the comparison
   family. Those also settle a second question for free — at magnitudes above `2**mantissa`
   every float is an integer, so `floor`/`ceil`/`round`/`trunc` must be the identity there,
   and nothing currently checks that.

6. **Then the overflow-producing ops**, one at a time, each with the
   `test_exp_overflow_saturation.py` treatment: a hand-chosen probe list where every value
   is exactly representable in the input format (a literal like `88.7` becomes `88.5` in
   bf16, and a probe written next to a threshold gets pinned to a value other than the one
   it names), and an assertion that the result saturates rather than wraps.

7. **Generalise `test_exp_overflow_saturation.py` last, not first.** Once two or three ops
   have their own saturation test, the shared shape will be obvious; deriving it before
   then will fit exp and nothing else.

### Pin it

```python
@pytest.mark.parametrize("fmt", [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16])
def test_format_extremes_straddle_the_ftz_cliff(fmt):
    """The subnormal probe must be on the flushed side and the min-normal probe above it,
    or the pair asserts nothing about FTZ."""
```

**Cost:** the largest item here. Budget it as: table + helper + one enrolled tranche
(step 5) as a first PR, then one op per follow-up.

---

## W8 — `SfpuLogsigmoid`'s `x > 4` branch is never driven

### Problem

`_logsigmoid_stimuli_spec()` (`test_sfpu_binary.py:484`) pins `x` to
`linspace(-8.0, 3.9)`, and its own comment explains why: "in1 (`exp(-x)`) is only read in
the `x > 4` branch, so restrict x to `[-8, 3.9]` (never uses in1)".

So the only code path in the kernel that reads operand B is never executed. The same fact
is then used to excuse the op from cat B — `_BINARY_SPECIALS_NOT_READY` records it as
"effectively unary — operand B is read only on the `x > 4` branch and the golden ignores
it, so a cat-B probe in B asserts nothing". That reasoning is correct *given* the stimulus
restriction, but the restriction is the thing to remove.

### Steps

1. **Widen the sweep to cross the branch:**

   ```python
   def dist(size, dtype, generator):
       return torch.linspace(-8.0, 12.0, size).to(dtype)
   ```

   `12.0` puts roughly a third of each face above the threshold, which is enough for the
   branch to be non-vacuous without dominating the tolerance verdict.

2. **Check what the golden does above 4.** `BinarySFPUGolden`'s logsigmoid currently
   ignores operand B outright. If the kernel reads `in1 = exp(-x)` there, the golden has
   to model the same composition — otherwise widening the sweep just produces a failure
   that blames the kernel for the golden's simplification. Read
   `ckernel_sfpu_logsigmoid.h` (or wherever the op lives for the target arch) and make the
   golden match the branch structure before widening.

3. **Confirm operand B is actually fed the right thing.** The driver supplies B from the
   op's stimuli spec; if the kernel expects `in1 == exp(-in0)` specifically, then B is not
   a free operand and the test needs a *paired* spec computing it from A — the shape
   `_isclose_stimuli_specs()` and `_comparison_stimuli_specs()` already use.

4. **Then revisit cat B.** Once B is genuinely read, the
   `_BINARY_SPECIALS_NOT_READY[SfpuLogsigmoid]` reason stops being true and the op becomes
   a cat-B candidate. Move it or rewrite the reason; leaving a stale justification is worse
   than the gap.

### Verify

```bash
CHIP_ARCH=blackhole pytest test_sfpu_binary.py -q -k logsigmoid
```

**Cost:** no new variants; one golden change.

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
   | F | finite magnitude extremes and subnormals | W7 |
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
  `0x80000000` as "negative zero", so the value cannot round-trip. Already covered by a
  dedicated `xfail` (`test_sfpu_binary_int_shift_int32_min_unsupported`) and by
  `integer_specials()`'s docstring. W3 should extend that `xfail`'s scope, not try to
  defeat it.
- **Negative integer operands generally**, for the same reason — which is why W3's signed
  division item is scoped to *recording* the limitation rather than passing.
- **`Lgamma` / `Digamma` / `Polygamma` at their poles.** The kernels are polynomial and LUT
  fits that claim accuracy only well inside a positive domain; a probe at the boundary
  tests a value the kernel never promised. Already recorded above `_OP_SINGULARITIES`.
- **A zero's sign in the *result*.** `passed_test()` judges by `torch.isclose`, a both-NaN
  clause and PCC, and `-0.0 == +0.0` under all three. Asserting a result's zero sign needs
  a bitwise comparator, which is a suite-wide change and out of scope here. Note this bounds
  W1: it fixes the *input* side of signed zero, not the output side.
- **A generated NaN's sign on Wormhole.** `SFPMAD.md` says it "might or might not be set";
  `generated_nan_sign_is_asserted()` already gates it per lane. Not a gap.
