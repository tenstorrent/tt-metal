# SFPU accuracy: the ULP sweep, and how to enrol an op in it

Every SFPU op declares how closely its output must match its golden, in
`python_tests/helpers/sfpu_accuracy_budget.yaml`. For most ops that declaration is a
tolerance. For the ops enrolled here it is a **step budget**: "every element is within
N representable values of the reference". Most unary budgets come from the exhaustive
sweep, which checks an op against *every distinct finite value of a 16-bit input format*
-- `±inf` and NaN are never fed, and `-0.0` is the same value as `+0.0` -- or, for
`Float32`, a strided sample of 65,280 across the whole range (`ulp_sweep.is_exhaustive`
tells the two apart). The exceptions are the `Signbit`, isinf/isnan and threshold-family
rows, measured on their functional drivers' hand-built stimuli (`MEASURED_ON_SWEEP`, after
the format table below). The isinf/isnan sweep feeds `±inf` and NaN on purpose.

This document is how you add an op to that second group.

## Quick start

```bash
cd tests/python_tests

# What the hardware measures for your op, over the whole format (a stride of Float32).
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --ulp-emit \
  --compile-producer -n 8
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --ulp-emit \
  --compile-consumer

# ...which rewrites your op's block in the table. Read the diff, then gate on it.
git diff helpers/sfpu_accuracy_budget.yaml
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --compile-consumer
```

`--op` matches the op name exactly. `-k MyOp` is a substring match: `-k Exp` also runs
`Exp2`, `ExpWithBase`, `Expm1` and `Expm1Cw`, and rewrites their blocks too.

`--ulp-emit` **writes the checked-in table**, once, at the end of the session; under
`-n` the controller merges every worker's measurements first. It refuses to write unless
you are on Wormhole, the session ran to the end (an interrupt, `pytest.exit()` or an
internal error writes nothing, however many grids were complete), no test in it failed
and every touched `(in, out)` grid is complete — but it is still a deliberate act, so
read the diff before committing it. Off Wormhole it refuses at startup, before
anything compiles. The producer half only compiles, and `--collect-only` runs nothing,
so both write nothing and say so. An op whose block it cannot regenerate (a floor row
or an op-wide `atol`/`rtol` row on a measured cell) is kept verbatim and named in the
summary; the session still passes, so a whole-table emit can end green.

## Why a step budget rather than a tolerance

`atol=0.05` is about 6 bfloat16 steps at 1.0 and about 0.01 at 512. A tolerance loose
enough to pass an op's tail is blind in the middle of its range, and PCC stays above
0.99 through error levels no consumer would accept. A step count is the same strictness
everywhere.

And the sweep measures a different number than a functional driver does. The drivers
sample a few thousand points from an op's *safe* domain; a budget measured that way
describes the sample, and cannot see a tail the sample never reaches. The sweep's number
is the format's.

## What the sweep feeds

| format | as input | as output | how |
|---|---|---|---|
| `Float16_b` | yes | yes | all 65,279 finite values |
| `Float16` | yes | yes | all 63,487 finite values |
| `Bfp8_b` | yes | yes | swept in bfloat16, packed on the way in |
| `Bfp4_b` | yes | **no** | swept in bfloat16, packed on the way in |
| `Float32` | yes | yes | **strided**, not exhaustive — see below |

`Bfp4_b` is input-only: it keeps 2 fractional bits, so a bfloat16 step count would read
every legal quantization of a `Bfp4_b` *output* as a 32-step error.

Binary, ternary and scalar rows are measured over those drivers' own sweeps, and those
drivers gate on the whole contract, `Float32` included. So do the unary signbit, isinf/isnan
and threshold sweeps, whose hand-built stimuli are what the predicates' rows were
measured on (`MEASURED_ON_SWEEP` in `test_sfpu_accuracy_budget.py`). The isinf/isnan
and threshold predicates have no registered domain, so those sweeps are their only gate;
`Signbit`, `ReluMin` and `ReluMax` are driven by the exhaustive sweep as well.

`Float32` has 2^32 values and one device run holds 2^16, so it cannot be enumerated. The
sweep strides the format's total order instead. Every binade holds the same number of
representable values, so each gets an equal share: one run reaches 261 binades from 0 to
3.4e38. Ask `ulp_sweep.is_exhaustive(input_format)` if you need to know which you got.

On Wormhole and Blackhole an exponent-B input (`Float16_b`, `Bfp8_b`, `Bfp4_b`) packed to
`Float16` needs a 32-bit Dest, so the sweep runs those cells with `dest_acc=Yes` only:
asked for `No`, the harness would run the `Yes` kernel anyway.

The domain is deliberately **not** clipped to the op's safe range. Undefined inputs are
swept and then masked out of the statistics, so they still reach hardware.

## Lanes the sweep does not count

`measurable_mask` drops four kinds, none of them a budget question:

- **either side NaN** — an op undefined at an input lands here on its own;
- **the two sides disagreeing about being non-finite** — a reciprocal overflowing where
  the golden is still finite; one such lane ranks at ~48,000 steps;
- **inputs the unpack does not deliver as fed** — below the smallest normal of the input
  format *and* of the format the unpacker writes, or past the latter's largest value
  where its exponent range is narrower: a Float32 input into a `Float16` output at
  `dest_acc=No` lands in a `Float16` Dest, which flushes below 2**-14 and saturates past
  65504 (about 44% of the strided lanes). Subnormal inputs are the first case: the
  hardware flushes them and the golden does not, so
  `ceil(5.69e-39)` is 1 in the model and 0 on silicon: 16,129 bfloat16 steps. Measured,
  that class alone was the whole of `Ceil`'s, `Floor`'s and `Sqrt`'s apparent error.
  Judged on the input as generated *and* as the block-float quantizer hands it to the
  golden: the sweep's one `-0.0` shares a `Bfp8_b` block with the bfloat16 subnormals
  beside it and quantizes to `-2**-127`, so `floor` read -1 there against silicon's 0;
- **the sweep's own zero padding** — the tensor is 65,536 lanes and bfloat16 has 65,279
  finite values, so the last 257 are padding rather than data.

The second kind is a *failure*, not a non-question, so `nonfinite_failures` reports it
separately — over everywhere the op claims an answer: the whole format, less the
undefined side of each singularity `sfpu_domains._OP_SINGULARITIES` registers (`Log`
below zero, `Reciprocal` at zero) and a per-op argument-reduction limit (`Sin` and `Cos`
past pi, on the formats that reach `sin(2.6e28)`: bfloat16 and Float32, not float16,
which ends at 65504 and is reduced correctly throughout). Not the functional driver's
sampling window, which is where points are drawn rather than where an op stops being
defined, and not `_SFPU_UNDEFINED_RANGES`, whose holes are guard bands around those
points rather than the points themselves. A golden past the output format's range is
excused only where the store saturated -- NaN, an infinity of the golden's sign, or on a
`Float16` output the pack's clamp to ±65504 of that sign (an out-of-range value from a
wider Dest packs to the format's largest magnitude). Any other finite answer to an
infinite golden is a failure.

One more exclusion is a defect already on the books rather than the sweep's doing.
`ulp_sweep._KNOWN_NONFINITE_LANES` names, per op and per cell, the inputs on which the
hardware is known to answer on the wrong side of infinity, each entry pointing at its
issue (`Celu` returns `inf` for `x` in 65408..65504 on a 16-bit `Float16` Dest, #58607).
Without it one such lane parked the whole ~64,000-lane cell as `not measurable`, which
the gate skips outright. Only the non-finite disagreement is excused: a named lane that
agrees is ranked like any other, and the gate fails the day no named lane of a cell
disagrees any more, so an entry cannot outlive its fix.

Subnormal *outputs* are ranked with the band flushed, on `Float16` as well: the golden
keeps IEEE fp16 subnormals that the pack path does not reproduce, and an exact op read
512 steps on `Float16_b -> Float16` from that band alone.

## Enrolling an op, step by step

### 1. Check it can be gated at all

A step count needs a per-element ULP, which means a float output. `ulp.has_ulp_gate(fmt)`
is the authority. Integer formats want bit equality; `Bfp4_b`, `Bfp2_b` and the MX
formats keep their block-aware lattice compares.

Your op also has to be in `sfpu_domains.sfpu_unary_ops()` and have a registered domain,
or the sweep has nothing to feed it.

### 2. Give it a block in the table

The emitter keeps an op's key line -- its name and any header comment -- and only adds
or replaces the `measured by:` clause on it; it never writes a key line itself. So a
*new* op needs one by hand first. Add the name and one placeholder row (a key with no
rows fails to load):

```yaml
MyOp:
  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # placeholder, replaced by the emit run
```

Key the placeholder on a swept `in`/`out` pair, or the emit run keeps it instead of
replacing it.

### 3. Measure it

Run the two commands under **Quick start**. The emitter rewrites your op's block with
what the hardware reported, and preserves anything it did not measure — a sampled row
for a cell the sweep does not drive, or an `arch:`-keyed entry. It writes only whole `(in, out)` grids: a
run narrowed inside one (`-k` on a single approx mode, `--maxfail`, an interrupt)
writes nothing.

### 4. Read what it wrote

```yaml
MyOp:  # measured by: exhaustive Float16_b/Float16/Bfp8_b/Bfp4_b + strided Float32 sweep, wormhole, 2026-09-23, except where a row says otherwise
  - {in: Float16_b, out: Float16_b, max_ulp: 2}  # max 1 ULP
  - {in: Float16, out: Float16_b, metric: tolerance}  # max 14337 ULP, budget would be 15771 > 6-step ceiling
  - {in: Float16_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized, so tolerance
```

Four verdicts:

- **`max_ulp: N`** — enrolled. `N` is the measurement plus 1.1x headroom, capped at the
  output's usable ceiling when the measurement itself fits under it. A measured 0 stays
  0 on a 16-bit input the sweep enumerates, and on an op exact by construction; on the
  strided `Float32` input it is written as 1, since a sample cannot assert exactness.
- **`metric: tolerance`, "budget N > ceiling C"** — the measurement itself is past
  `usable_budget_ceiling`, so a step budget would no longer be *tighter* than the
  tolerance it replaces; `N` is the budget it would have needed. The op keeps tolerance + PCC on that cell and the number is
  recorded so nobody re-derives it.
- **`metric: tolerance`, "block-quantized"** — a block float output. The sweep
  enumerates a format in value order, so sixteen adjacent values share a `Bfp8_b` block
  and the exponent fits all of them: the best case for quantization, not a
  representative one. `Abs` reads **15,616 steps** there from random mixed-magnitude
  blocks (the table's `Bfp8_b` note) and **393** from the sorted sweep of a bfloat16
  input, so neither number is enrollable.
- **`metric: tolerance`, "not measurable: …"** — the cell had no lane a step count could
  describe, or disagreed with the golden about being finite where the op claims an
  answer (inf/NaN against a finite golden, or a finite answer to an infinite one). No budget buys that;
  the row says why instead of leaving a hole the next emit would paper over, and it still
  records `max N ULP` over the lanes that *were* measurable, so the rest of the cell is
  not thrown away with the demotion. On a
  gateable output such a row must also be acknowledged, with its cause, in
  `test_sfpu_accuracy_budget._UNMEASURABLE_CELLS_ACKNOWLEDGED` — and if the cause is a
  tracked defect on a handful of inputs, it belongs in `_KNOWN_NONFINITE_LANES` instead,
  so the rest of the cell keeps its gate.

### 5. Gate on it

Re-run without `--ulp-emit`. Green means the budgets in the table hold over every value
of a 16-bit input and over the `Float32` stride. Then run the host guards, which check things the sweep cannot:

```bash
pytest test_sfpu_accuracy_budget.py test_ulp_sweep.py -q
```

## The rules the table enforces

- **No number is a guess.** The trailing comment on a row is the measurement it came
  from. A budget may only be *raised* by re-measuring and updating that comment in the
  same change. From #57527, `llk-sfpu-ulp-budget-guard` fails a pull request that raises
  one without it; the `ulp-budget-raise-approved` label is the override, and it still
  reports which rows it admitted without a fresh measurement.
- **Most specific key wins.** A row's key fields are `in`, `out`, `approx`, `dest` and
  `arch`, all optional. Two rows matching one variant equally specifically are an
  authoring error, not a tie-break, and the loader refuses them.
- **A row holds for the configuration it was measured in.** Every exhaustive row was
  measured with `FAST_MODE(No)` and `CLAMP_NEGATIVE(True)` compiled in, and the key has
  no axis for either. The unary functional driver builds `Sqrt`/`Rsqrt` at
  `FastMode.Yes` and takes only a row's tolerance arm, which those flags do not move.
- **An exhaustive budget is the emitter's number.** `test_no_step_budget_exceeds_the_measurement_it_records`
  holds every row the sweep wrote to exactly `_verdict`'s budget for the measurement
  beside it, so widening one by hand has to falsify its comment. A sampled row may sit up
  to 2x its measurement.
- **A gated cell is not parked quietly.** A `not measurable` verdict on a gateable output
  has to be acknowledged in `_UNMEASURABLE_CELLS_ACKNOWLEDGED` with its cause, or the
  lanes behind it named in `_KNOWN_NONFINITE_LANES` with their issue.
- **Budgets do not transfer between architectures.** Every unkeyed number was measured
  on Wormhole. Off it, an op falls back to tolerance unless a row names that `arch`
  itself. Re-measure before trusting any of it on Blackhole.
- **An exact op may not carry a wide budget.** Sign-bit ops, copies, integer results,
  predicates and constant fills are the flakiness canaries: if one fails, the golden or
  the datapath moved. `test_an_exact_op_never_carries_a_wide_budget` holds their
  budgets, and `test_every_swept_cell_of_an_exact_op_is_gated_or_waived` holds the cells
  demoted to tolerance to an explicit list, each with its measurement.
- **`Bfp8_b` charges for block quantization.** A budget there is denominated in bfloat16
  steps, two to one `Bfp8_b` step, and does not forgive the shared exponent. No cell
  with a `Bfp8_b` output is gated by steps: the sweep records its measurement and the
  cell stays on tolerance, whose lattice compare is the stronger criterion there.

## Not enrolling an op

Two ways, and they mean different things:

- **No row in the table** — the op resolves to today's tolerance and the sweep skips it.
  Fine as an interim state; from #57520, `test_every_unary_op_is_enrolled_or_excused`
  fails until you pick one of these two deliberately.
- **`sfpu_domains._UNARY_OPS_NOT_SWEPT`** — the op cannot be swept at all. Say why in a
  comment next to it.

## A budget that is nearly right

If an op is exact over most of its range and catastrophic in a narrow band near zero,
the measurement will be dominated by the band and the cell will demote to tolerance.
`near_zero_atol` is the floor under the budget for exactly that: the lanes where the
reference crosses zero, judged on absolute error instead of steps. `Gelu` and most of
`Erfinv` are gated that way. The emitter cannot re-derive a floor, so an op with a floor
row on a cell the run measured keeps its whole block as it was: every other op is
written, and the summary names the ops it kept. You settle those by hand; the session
does not fail on them, because such a block is hand-maintained by design.

## Gotchas

- **Any host pytest run in `tests/` wipes `/tmp/tt-llk-build`.** Chain
  `--compile-producer` and `--compile-consumer` in one go; if you run a host suite in
  between, recompile.
- **The sweep is marked `accuracy`**, which every LLK workflow deselects. Run it by name
  or by `-m accuracy`. `nightly` would *not* have kept it out of `llk-e2e`.
- **Both runs take the same op set**: every unary op with a key line in the table,
  whatever its rows say. The key line is the enrolment, so an op on tolerance everywhere
  is measured on every run and held to the figures its rows record by the headroom
  report, and an op with no block is not swept at all (step 2 adds the block first).
  `--ulp-emit` changes what is written, not what is collected, so the producer and the
  consumer build the same set with or without it.
- **`CHIP_ARCH` must be set**, and `--ulp-emit` refuses to write on anything but
  Wormhole, because `_render` does not emit `arch` and the rows would be badged wrongly.

## Where things live

| file | what |
|---|---|
| `python_tests/test_unary_sfpu_ulp.py` | the harness: one device run per variant |
| `python_tests/helpers/ulp.py` | the metric: `ulp_distance`, `within_ulp`, the near-zero floor |
| `python_tests/helpers/ulp_sweep.py` | what the sweep feeds, what it masks, and the emitter |
| `python_tests/helpers/sfpu_accuracy_budget.yaml` | the table |
| `python_tests/helpers/sfpu_accuracy_budget.py` | how a row is resolved |
| `python_tests/helpers/ulp_budget_diff.py` | the CI comparison, and the headroom report (from #57527) |
