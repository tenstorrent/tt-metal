# SFPU accuracy: the ULP sweep, and how to enrol an op in it

Every SFPU op declares how closely its output must match its golden, in
`python_tests/helpers/sfpu_accuracy_budget.yaml`. For most ops that declaration is a
tolerance. For the ops enrolled here it is a **step budget**: "every element is within
N representable values of the reference". Most unary budgets come from the exhaustive
sweep, which checks an op against *every normal value of a 16-bit input format* and its
zero -- `±inf`, NaN and the subnormals are never fed (the unpack flushes a subnormal, so a
lane there would measure the flush rather than the op), and `-0.0` is the same value as
`+0.0` -- or, for `Float32`, a strided sample of 65,024 across the whole range
(`ulp_sweep.is_exhaustive` tells the two apart). The exceptions are the `Signbit`, isinf/isnan and threshold-family
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
and every touched `(input, output)` grid is complete — but it is still a deliberate act, so
read the diff before committing it. Off Wormhole it refuses at startup, before
anything compiles. The producer half only compiles, and `--collect-only` runs nothing,
so both write nothing and say so. Of a measured op it replaces every row of each
measured `(input, output)` pair, and keeps the rows no measurement re-derives: one that
names an `arch`, declares `atol`/`rtol`, or names a pair this run did not measure. Each
refusal is its own `EmitRefused` subclass, carrying the facts as attributes; a rewrite the loader would refuse, or that would resolve ambiguously for some
variant, is a `TableError` raised before anything is written, since `write_table_file`
reads its dump back and resolves it first. A successful
emit returns a `WriteReport` that the one-line summary is rendered from.

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
| `Float16_b` | yes | yes | all 65,025 values: every normal one, and zero |
| `Float16` | yes | yes | all 61,441 values: every normal one, and zero |
| `Float32` | yes | yes | **strided**, not exhaustive — see below |

Those are the formats the SFPU computes in. A block float (`Bfp8_b`, `Bfp4_b`) is not
swept and no row names one: its error is the unpacker's and packer's, not the op's, so
a variant with one keeps the op's declared tolerance.

Binary, ternary and scalar rows are measured over those drivers' own sweeps, and those
drivers gate on the whole contract, `Float32` included. So do the unary signbit, isinf/isnan
and threshold sweeps, whose hand-built stimuli are what the predicates' rows were
measured on (`MEASURED_ON_SWEEP` in `test_sfpu_accuracy_budget.py`). The isinf/isnan
and threshold predicates have no registered domain, so those sweeps are their only gate;
`Signbit`, `ReluMin` and `ReluMax` are driven by the exhaustive sweep as well.

On Wormhole and Blackhole no functional driver sweeps a `Float16` input or output, so
nothing in CI gates the `Float16` rows there: the binary ones, and the unary rows on
the tolerance metric. They stay in the table as the last measurement, and a stale one
fails no test; re-measure before relying on one.

`Float32` has 2^32 values and one device run holds 2^16, so it cannot be enumerated. The
sweep strides the format's total order instead. Every binade holds the same number of
representable values, so each gets an equal share: one run reaches all 254 normal
binades, from 1.2e-38 to 3.4e38, with no subnormal and no zero among its 65,024
samples. Ask `ulp_sweep.is_exhaustive(input_format)` if you need to know which you got.

On Wormhole and Blackhole an exponent-B input (`Float16_b`) packed to
`Float16` needs a 32-bit Dest, so the sweep runs those cells with `dest_acc=Yes` only:
asked for `No`, the harness would run the `Yes` kernel anyway.

The domain is deliberately **not** clipped to the op's safe range. Undefined inputs are
swept and then masked out of the statistics, so they still reach hardware.

## Lanes the sweep does not count

`measurable_mask` drops five kinds, none of them a budget question, and the driver then
keeps only the lanes the op *claims* (below):

- **either side NaN** — an op undefined at an input lands here on its own;
- **the two sides disagreeing about being non-finite** — a reciprocal overflowing where
  the golden is still finite; one such lane ranks at ~48,000 steps;
- **inputs the unpack does not deliver as fed** — below the smallest normal of the input
  format *and* of the format the unpacker writes, or past the latter's largest value
  where its exponent range is narrower: a Float32 input into a `Float16` output at
  `dest_acc=No` lands in a `Float16` Dest, which flushes below 2**-14 and saturates past
  65504 (about 44% of the strided lanes). The sweep feeds no subnormal of its own --
  the hardware flushes them and the golden does not, so `ceil(5.69e-39)` read 1 in the
  model and 0 on silicon, 16,129 bfloat16 steps, once the whole of `Ceil`'s, `Floor`'s
  and `Sqrt`'s apparent error;
- **answers the Dest lost** (`dest_holds`) — a `Float16` input runs on an fp16 Dest,
  which tops out at 65504 and flushes below 2^-14 whatever the output. Under a `Float32`
  or `Float16_b` output the hardware hands back 65536 (or NaN) where the answer was
  70000, while the golden holds the answer (or substitutes a finite 131008 for an fp16
  infinity): 8 million fp32 steps for the Dest's overflow, not the op's. Judged on the
  pair: an overflow is excused only where the answer is NaN or has the golden's sign in
  the Dest's top binade (2^15 or more -- past 65504 the golden is a stand-in, so an
  answer near the edge has nothing true to be ranked against: `I0`'s 65440 for 65772
  is its usual 0.5%), and a flush only across the 2^-14 boundary (one side 0, the other
  at most 2^-14). A wrong-signed or small finite answer is the op's -- `Expm1Cw`'s
  #54770 answers -1 or -131008 where the golden is 131008. Only where the Dest
  is narrower than the output; under an fp16 output the two overflow together and that
  is judged as an output overflow;
- **the sweep's own zero padding** — the tensor is 65,536 lanes and the bfloat16 sweep feeds
  65,025 values (every normal one, and zero), so the last 511 are padding rather than data.

The second kind is a *failure*, not a non-question: it is dropped from the ranking only
because `nonfinite_failures` reports it, over the same claimed lanes.
`ulp_sweep.ranked_mask` ANDs `claimed_lanes` into the mask, so the budget is measured over
everywhere the op claims an answer: the whole format, less the undefined side of each
singularity `sfpu_domains._OP_SINGULARITIES` registers (`Log` below zero, `Reciprocal`
at zero), every non-positive integer for `Digamma`, `Lgamma` and `Polygamma`
(`_NONPOSITIVE_INTEGER_POLES`: the poles are a lattice, and every fp32 value of
magnitude 2^23 or more is an integer) and a per-op argument-reduction limit (`Sin` and
`Cos` and `Tan` past 65504, where the reduction breaks, on the formats that reach it:
bfloat16 and Float32, not float16, which ends there). The same limit bounds the lanes the
sweep ranks, so a range-reduction regression below it fails the gate. Not the functional driver's
sampling window, which is where points are drawn rather than where an op stops being
defined, and not `_SFPU_UNDEFINED_RANGES`, whose holes are guard bands around those
points rather than the points themselves. In `nonfinite_failures`, a golden past the
output format's range is excused only where the store saturated -- NaN, an infinity of the golden's sign, or on a
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

A step count needs a format the SFPU computes in: `Float16_b`, `Float16` or `Float32`
(`ulp.ULP_FORMATS`), as input and output. Integer formats want bit equality; the block floats
and the MX formats keep their block-aware lattice compares.

Your op also has to be in `sfpu_domains.sfpu_unary_ops()` and have a registered domain,
or the sweep has nothing to feed it.

### 2. Give it a block in the table

The sweep collects the ops the table names, so a *new* op needs one row by hand first
(an op with no rows fails to load, and so does a ULP row without the `measured` figure
that is its gate):

```yaml
MyOp:
- {input: Float16_b, output: Float16_b, metric: TOLERANCE, note: placeholder}
```

Key the placeholder on a swept `input`/`output` pair, or the emit run keeps it instead
of replacing it.

### 3. Measure it

Run the two commands under **Quick start**. The emitter rewrites your op's block with
what the hardware reported, and preserves anything it did not measure — a sampled row
for a cell the sweep does not drive, or an `arch:`-keyed entry. It writes only whole `(input, output)` grids: a
run narrowed inside one (`-k` on a single approx mode, `--maxfail`, an interrupt)
writes nothing.

### 4. Read what it wrote

```yaml
MyOp:
- {input: Float16, output: Float16_b, dest: 'No', domains: [{low: -.inf, high: -4.0, ulp: 14337}, {low: -4.0, high: .inf, ulp: 1}]}
- {input: Float16_b, output: Float16_b, dest: 'No', domains: [{low: -.inf, high: .inf, ulp: 1}]}
```

Every row records what was measured: a `domains` list on every swept cell with a
per-element figure, a plain row with a `measured` figure on the rest (a cell the sweep
cannot measure, one where no entry would carry a figure, a cell it does not drive). A
figure *is* the gate: there is no headroom over it. A row's `note` is for people; no
code reads it. Five verdicts:

- **`domains: [{low: L, high: H, ulp: U}, …]`** — every cell with a per-element ULP,
  one list over the whole swept range ("Where an op holds its figure", below). An entry
  within the output's usable ceiling is gated at its figure: the sweep holds its lanes to
  exactly `U` steps, a reading one step worse fails it, and a re-emit that records the
  worse figure is a raised gate the PR guard reports. An entry past the ceiling is
  measured, not gated, and the headroom report holds it to its figure. An entry with no `ulp`
  has too few ranked lanes to draw an edge on, none of them worse than the
  row's largest figure. A one-domain op gets a one-entry list, so every
  such row reads the same way. A measured 0 stays 0, on the strided `Float32` input
  too: the stride is a fixed set of values, and the figure is a measurement of it.
- **`near_zero_atol: A`** beside the list — a floor: the step count is a difference of
  bit-pattern ranks, so a lane where the hardware answers 0 for a 2e-9 the golden still
  holds reads as 2^29 fp32 steps, and one such lane demoted the whole cell. Lanes under
  1 % of the cell's largest golden (and under `A/0.01`) are judged on absolute error
  instead; `A` is exactly the largest absolute error among the band lanes past the
  ceiling, written as measured, and every entry's figure is over the lanes outside it.
  The emitter grants a floor up to `ulp_sweep.EMIT_MAX_NEAR_ZERO_ATOL` (1e-3); a cell
  that would need more keeps its list with the floor capped there, and the entries show
  where it still falls past the ceiling.
- **`measured: N`**, no `metric` — a row a functional driver sampled (the binary,
  ternary and scalar ops, and the hand-built unary sweeps). The driver holds the cell to
  exactly `N` steps.
- **`metric: TOLERANCE, measured: N`** on a step-gateable output — a row the drivers
  sampled past `usable_budget_ceiling`, where a step budget would no longer be *tighter*
  than the tolerance it replaces. The op keeps tolerance + PCC on that cell, and
  `measured` is what the headroom report holds it to. A list whose largest figure is
  past the ceiling implies the same for its whole cell, while its entries within the
  ceiling stay gated.
- **`metric: TOLERANCE, measured: N, nonfinite: L`**, noted "not measurable: …" — `L` lanes
  disagreed with the golden about being finite where the op claims an answer (inf/NaN
  against a finite golden, or a finite answer to an infinite one), or the cell had no
  lane a step count could describe (no `measured`). No budget buys that; the row says
  why instead of leaving a hole the next emit would paper over, and `measured` is the
  worst of the lanes that *were* measurable (absent when none was), so the rest of the
  cell is not thrown away with the demotion. On a
  gateable output such a row must also be acknowledged, with its cause, in
  `test_sfpu_accuracy_budget._UNMEASURABLE_CELLS_ACKNOWLEDGED` — and if the cause is a
  tracked defect on a handful of inputs, it belongs in `_KNOWN_NONFINITE_LANES` instead,
  so the rest of the cell keeps its gate.

### 5. Gate on it

Re-run without `--ulp-emit`. Green means the budgets in the table hold over every value
of a 16-bit input and over the `Float32` stride. Then run the host guards, which check things the sweep cannot:

```bash
cd tests/python_tests
LLK_HARDWARE_FREE=1 CHIP_ARCH=wormhole pytest -c pytest.ini --rootdir=. --confcutdir=. \
  -p helpers.llk_pytest_plugin \
  test_ulp_budget_diff.py ../hardware_free/accuracy/test_ulp_table.py \
  ../hardware_free/accuracy/test_ulp_domains.py \
  ../hardware_free/accuracy/test_sfpu_accuracy_budget.py \
  ../hardware_free/accuracy/test_ulp_sweep.py -q
```

## The rules the table enforces

- **The file is typed data with one writer.** Every row is a `ulp_table.Row`: its keys
  are the dataclass's fields, an enum is written by its member name (`Float16_b`,
  `TOLERANCE`, `'Yes'`), and `ulp_table.dump_table` writes the whole file, rows in one
  order. A row the loader refuses raises `TableError`, whose `problem` is a `Problem`
  member saying why. A hand edit must leave the file as the writer would
  (`test_the_table_on_disk_is_exactly_what_the_serializer_writes`); to normalise one,
  `python -c "from helpers.ulp_table import *; write_table_file(load_table(), TABLE_PATH)"`.
- **No number is a guess, and the measurement is the number.** A ULP row's gate is its
  `measured` figure; the loader refuses one without it. There is no headroom: a worse
  measurement is a raised gate, recorded by a re-emit in a change of its own.
  `python -m helpers.ulp_budget_diff diff --base <old.yaml> --head <new.yaml>` lists
  every cell a change loosens.
- **Most specific key wins.** A row's key fields are `input`, `output`, `approx`, `dest`
  and `arch`, all optional. Two rows matching one variant equally specifically are an
  authoring error, not a tie-break, and the loader refuses them.
- **A row holds for the configuration it was measured in.** Every exhaustive row was
  measured with `FAST_MODE(No)` and `CLAMP_NEGATIVE(True)` compiled in, and the key has
  no axis for either. The unary functional driver builds `Sqrt`/`Rsqrt` at
  `FastMode.Yes` and takes only a row's tolerance arm, which those flags do not move.
- **A gate is a measurement, swept or sampled.** The drivers seed their stimuli, so a
  sampled row's figure is reproducible and gates like the sweep's; neither carries an
  envelope.
- **The list is the statement.** A row with `domains` carries no `measured` or `metric`
  beside it and pins the `input` and `output` it was measured on; the loader refuses anything
  else. An entry never widens or narrows what the op
  claims (`claimed_lanes`): it reports where the claim holds, and at what figure.
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

## Not enrolling an op

Two ways, and they mean different things:

- **No row in the table** — the op resolves to today's tolerance and the sweep does not
  collect it.
  Fine as an interim state; from #57520, `test_every_unary_op_is_enrolled_or_excused`
  fails until you pick one of these two deliberately.
- **`sfpu_domains._UNARY_OPS_NOT_SWEPT`** — the op cannot be swept at all. Say why in a
  comment next to it.

## A budget that is nearly right

If an op is exact over most of its range and catastrophic in a narrow band near zero,
the measurement will be dominated by the band and the cell will demote to tolerance.
`near_zero_atol` is the floor under the budget for exactly that: the lanes where the
reference crosses zero, judged on absolute error instead of steps. The emitter derives
it from the measurement (`ulp_sweep.near_zero_floor`) and writes it on the row, so a
re-emit regenerates a floor row like any other. The rows it keeps as written are the
ones no measurement re-derives (`ulp_sweep._kept`): a row naming an `arch`, a row
declaring `atol`/`rtol` (the op-wide anchors of `GeluAppx` and `SigmoidAppx`), and every
row for a pair the run did not measure, such as the `Abs` and `Identity` wildcard rows. A floor past `EMIT_MAX_NEAR_ZERO_ATOL` is a kernel's cut-off rather than its
rounding -- `Softplus` answers 0 below x = -5 where the answer is up to 0.0065, the
approximate `Gelu` is 0.024 off -- and stays a hand decision: the demoted row records
the floor it would have needed.

## Where an op holds its figure: domains

One line per format combination says how the op behaves over the whole swept range:

```yaml
Tanh:
- {input: Float16_b, output: Float16_b, approx: 'Yes', dest: 'No', near_zero_atol: 0.00042724609375, domains: [{low: -.inf, high: -1.0, ulp: 3}, {low: -1.0, high: -0.0078125, ulp: 10}, {low: -0.0078125, high: 0.0078125, ulp: 0}, {low: 0.0078125, high: 1.0, ulp: 10}, {low: 1.0, high: .inf, ulp: 3}]}
Log:
- {input: Float16_b, output: Float16_b, dest: 'No', domains: [{low: -.inf, high: 0.0}, {low: 0.0, high: .inf, ulp: 1}]}
Abs:
- {input: Float16_b, output: Float16_b, dest: 'No', domains: [{low: -.inf, high: .inf, ulp: 0}]}
```

Read: approximate `tanh` is 3 steps outside `[-1, 1)` and 10 on either side of the
origin, past bfloat16's ceiling of 6 — those two entries are why the cell as a whole
is not gated; the three others are, at 3, 0 and 3. The 0 is `[-2^-7, 2^-7)`, which
reads 0 steps outside the row's `near_zero_atol` floor (the lanes it rescues are
judged on absolute error instead). `log` answers nothing below 0 and is 1 step
from 0 up. `abs` is exact over the whole range.

- **An entry** `{low: L, high: H, ulp: U}` holds the lanes whose input, *as the kernel
  received it* (`ulp_sweep.received_inputs`: truncated to the 16-bit Dest for a
  `Float32` input at `dest: "No"`), satisfies
  `L <= x < H`. `U` is the worst ranked lane there, outside the row's near-zero floor
  if it has one; an entry with no `ulp` means too few lanes there are ranked to draw an edge on (fewer
  than `DOMAIN_MIN_BIN_LANES` in each of its binades): past the op's claim (`cos`
  beyond 65504), undefined (`log` below 0, every value below -128 for `lgamma`), or an
  answer the Dest cannot hold. The few ranked lanes an entry with no `ulp` can still hold are
  gated at the row's largest figure, so the emitter leaves `ulp` out only while they fit
  it; otherwise the entry whose strays are worst carries its own figure. A cell where
  no entry would carry a figure gets a plain row, not a list. Which lanes an op claims
  stays in code
  (`_CLAIM_LIMIT`, `_OP_SINGULARITIES`, `_NONPOSITIVE_INTEGER_POLES`); the list reports
  it.
- **Gated or not follows the figure.** An entry within `usable_budget_ceiling` of the
  output (6 for `Float16_b`, 51 for `Float16`, 419,430 for `Float32`) is gated at it;
  one past the ceiling is measured and held to its figure by the headroom report. The
  whole-cell contract `accuracy_contract` hands to callers that do not look at domains
  (the functional drivers) is derived: gated at the largest figure when every entry is
  within the ceiling, tolerance otherwise.
- **The list is contiguous and complete**: the first `low` is `-.inf`, the last `high`
  is `.inf`, each `high` is the next `low`, and every interior edge is `0` or a power
  of two (`0.00390625`, `65536.0`, `8.507059173023462e+37`).
- **The emitter draws it** from the sweep's per-lane distances (`ulp_domains`): the
  worst ranked lane per signed binade of the received input (`[2^k, 2^(k+1))`, or
  `[-2^(k+1), -2^k)`, so -1 opens `[-1, -0.5)` and no binade straddles an edge), split
  where it crosses the ceiling (`DOMAIN_CLASS_BITS` would split the gateable range into step classes as
  well; the binary rule is the question asked, and factor-2 classes gave eight times
  the entries for the pack path's 0-vs-1 flicker between binades), a binade with fewer
  than `DOMAIN_MIN_BIN_LANES` lanes folded into its neighbour, a gateable domain under
  `DOMAIN_MIN_LANES` ranked lanes folded into the neighbour at least as loose (a bad
  region is never hidden that way), an edge with no swept lane between it and zero
  written as `0`. A lane the row's floor covers counts as ranked at 0 steps when the
  list is drawn -- it is held, by the floor -- so the near-zero band draws no entry of
  its own, and an entry the floor covers entirely reads `0`. Every entry's figure is
  then measured over exactly its lanes outside the floor, so what is written is what
  the gate holds them to.
- **The sweep gates entry by entry.** Each gateable entry's lanes are held to its
  figure with the row's floor; the ranked lanes an entry with no `ulp` leaves (strays in
  otherwise unranked binades) are held to the row's largest figure, so no ranked lane
  goes unjudged. A failure names the entry. Where that figure is past the ceiling the
  sweep gates no fallback, and the headroom report holds those lanes to it instead.
- **The headroom report judges entry by entry too**: the `--ulp-measure` row carries
  the reading over each entry of the table's list (`domains`) and the per-binade profile
  it was drawn from (`bins`), and `python -m helpers.ulp_budget_diff domains --measured
  ulp.jsonl [--op Cos] [--table helpers/sfpu_accuracy_budget.yaml]` prints, from that
  artifact and no device, the list the rule would draw today beside the one the table
  holds.
- **The PR guard judges segment by segment**: over the union of both revisions' edges,
  an entry raised, a list replaced by a looser one, or a plain row turned into a list
  are each reported for exactly the lanes they change, with the range on the line, and
  a list redrawn with the same figures is no change at all.

The hand-written `atol`/`rtol` anchors (`GeluAppx`, `SigmoidAppx`) are op-wide rows the
emitter keeps beside the lists it writes for those ops' cells. A cell whose list is past
the ceiling has no numbers of its own, so the functional driver falls back to the
anchor's `atol`/`rtol`.

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
- **A plain gate run does not judge tolerance cells.** It fails a gated cell over its
  budget and *skips* a tolerance cell, with the measured maximum in the skip reason. To
  hold those cells to the figures their rows record, measure and compare:

  ```bash
  CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py -n auto --ulp-measure=ulp.jsonl
  python -m helpers.ulp_budget_diff headroom \
    --table helpers/sfpu_accuracy_budget.yaml --measured ulp.jsonl
  ```

  Each line of `ulp.jsonl` is one `ulp_table.Measurement`, read back by
  `read_measurements`; `judge` returns the typed findings the report is rendered from.
- **`--ulp-measure` can fail the sweep.** Elsewhere it only reports, but the sweep's
  records are required: the headroom report judges only the cells it reads, so a sweep
  cell whose record is not written fails (`RecordDropped`), and a path that cannot be
  prepared at session start is a usage error before anything runs.
- **`CHIP_ARCH` must be set**, and `--ulp-emit` refuses to run on anything but
  Wormhole, because the emitter writes no `arch` and the rows would be badged wrongly.

## Where things live

| file | what |
|---|---|
| `python_tests/test_unary_sfpu_ulp.py` | the harness: one device run per variant |
| `python_tests/helpers/ulp.py` | the metric: `ulp_distance`, `within_ulp`, the near-zero floor |
| `python_tests/helpers/ulp_sweep.py` | what the sweep feeds, what it masks, and the emitter |
| `python_tests/helpers/sfpu_accuracy_budget.yaml` | the table |
| `python_tests/helpers/ulp_table.py` | the table as typed rows, its one reader and writer, and the `--ulp-measure` records |
| `python_tests/helpers/sfpu_accuracy_budget.py` | how a row is resolved |
| `python_tests/helpers/ulp_domains.py` | the `domains` list: its meaning and the rule that draws it; `Problem` and `TableError` |
| `python_tests/helpers/ulp_budget_diff.py` | the CI comparison, and the headroom report (from #57527) |

## Notes on the table

What the rows of the table record, and why, by group and then by op. A row's own
`note` says what is particular to that cell.

### Enrolled first: the ops exact by construction

They are the flakiness canaries. A 0-step budget on Abs cannot be wrong about the
kernel -- if one fails, the golden or the datapath moved. Re-measure on Blackhole
before trusting any of it there.

Relu is absent because it is packer-applied via STACC_RELU and is not a SfpuType
member, so it will not compile through the unary driver. ReluMax/ReluMin take a
threshold operand; they are enrolled with the exact and predicate ops below.

Sign-bit and select: exact in fp32, one step in the 16-bit formats. No arithmetic to
round, so fp32 is bit-exact. The single step on the 16-bit outputs is the *pack* path,
not the op -- it shows up for Abs, Neg and Identity, mostly at dest_acc=Yes where it is
rounded at pack rather than truncated in Dest first, and a step budget is what makes it
visible at all (atol=0.05 is ~6 bf16 steps). These are exact by construction, so any
movement is real signal. The narrowing Float16->Float16_b rows measure 1.

### Exact and predicate unary ops

Fill, Threshold, the Is* predicates, LogicalNot, Signbit, UnaryEq/UnaryNe,
ReluMax/ReluMin, Frac, Sign and Heaviside, each in its alphabetical place below.

Measured on Wormhole over the standard unary sweep with --ulp-report.
These ops are exactly rounded by construction -- a predicate writing 1.0/0.0, a
constant fill, a clamp, or a subtraction the hardware does in one step -- so the budget
is 0 and any drift at all is a regression. The exception is a 16-bit output from a
wider input, where a selection or clamp that passes x through can measure 1 step: the
op is still exact, and the step is the output rounding on the way out of Dest, as the
rows there record, and every row there carries exactly that 1: the measured step is
the gate.

Every row names an output format that was actually measured; an unlisted format falls
through to the tolerance metric.

### Binary and ternary SFPU, and the remaining unary approximations

GeluTanh, Tanhshrink, SqrtCustom and the Sfpu* ops other than SfpuElwpow and
SfpuXlogy (which have their own note), each in its alphabetical place below.

Measured on Wormhole with --ulp-report over each driver's own sweep. The
`measured` figure each row records is the worst lane of that sweep, and it is the
gate: the drivers seed their stimuli, so the figure is reproducible. A row is
only written where it is *tighter* than the tolerance it replaces: the usable ceiling
is 419430 steps on Float32, 51 on Float16 and 6 on Float16_b, and an op past it is
recorded as tolerance with its measurement so nobody re-derives it.

### Derived from the accuracy sweep

The rows of an op the sweep drives were written by `pytest test_unary_sfpu_ulp.py
--ulp-emit` from that run's own measurements; the other rows were measured by a
functional driver's sample, and nothing regenerates them. Rows the emitter keeps
-- an op-wide `atol`/`rtol` -- are hand-maintained. Every key pins `input` as well as `output`: tanh fp32->fp32 at
dest_acc=Yes agrees to 22 mantissa bits, which a cell mixing all three input formats
reported as 8.

Regenerate a swept block with `--ulp-emit` rather than editing a row: `measured` is
what the budget was derived from, and a number changed without a run is a guess.

### Unary approximations mostly on tolerance

GeluTanh and Tanhshrink are past the ceiling on most Float16_b and Float32 columns, by
up to three orders of magnitude on Float32, where the tolerance is the tighter bound.
The cells that measure inside it -- mostly Float16 outputs -- carry step rows and gate.

### Binary selection and arithmetic

max/min are a bare SFPSWAP operand selection, so the answer is one of the inputs and
exact. The one step is the output pack on a cell whose output has fewer mantissa bits
than its input -- Float16 -> Float16_b and Float32 -> Float16 read 1, every other cell
0 (Float32 -> Float16_b's sample included) -- so every 16-bit cell is keyed
by input, and an exponent-B input into Float16, which only ever runs on the promoted
32-bit Dest, by no Dest at all.

### Binary comparison predicates: exact by construction

Each writes 1.0/0.0, so 0 is the only correct answer. The measurement sits on each op
rather than here, so every budget names the sweep that backs it: 0 ULP over 147k-246k
lanes on each of the six (SfpuElwEq, SfpuElwNe, SfpuElwGe, SfpuElwGt, SfpuElwLe,
SfpuElwLt).

### SfpuElwpow and SfpuXlogy: binary SFPU, still on the tolerance metric

These were BINARY_CUSTOM_TOLERANCES in test_eltwise_binary_sfpu. Only two binary ops
need an override: their error is a property of the op's own composition rather than of
the stimuli, so it grows with the operands however the domain is drawn. Both numbers
were measured over the domains registered in sfpu_domains -- re-measure before
widening either. Both are keyed per output format; an unlisted format falls through to
the per-format default rather than to the widest measured value.

No step budget for these two yet -- enrolling one means measuring the binary sweep the way the
unary ops were measured. The numbers now sit next to the op instead of in a dict in
a test file, which is the point of the table.
pow's error is relative and roughly flat, and does not scale with the output format's
precision, so one rtol suits every float column.

### Per op

#### Digamma

tracked: #51128 (saturates below the LUT range, under #52178), #58682 (no branch for x <= 0)

#### Erfc

tracked: #51137 (input clamp, under #52178)

#### Expm1Cw

tracked: #54770 (exponent wraps past overflow; closed, production feeds x < 0 only)

#### Fill

0 ULP, 115 variants / 2.09M lanes

#### Frac

Frac computes x - trunc(x), so the result carries none of the input's rounding, and
its rows gate each swept cell at what it measured (0 on most, 1 on the narrowing
cells). The sampled `{output: Float16}` wildcard's 384 steps have no
cause recorded; it answers only the cells the sweep does not drive.

#### GeluAppx

GeluAppx and SigmoidAppx, still on the tolerance metric, moved here from the test body:
CUSTOM_TOLERANCES in test_eltwise_unary_sfpu, a coarse 3-segment LUT carrying atol=0.13
so the sweep passes. The clearest argument for this mechanism -- that number makes the
test blind to a 10x regression anywhere else in the domain, and equally blind to the
improvement a retune would produce. The two carry the same op-wide row because they
share one cause: a retune that moves one should move the other in the same diff. The
per-cell rows record what the sweep measured; being numberless, they do not shadow that
atol.

#### Heaviside

See the note under Sign.

#### I0

tracked: #50465 (no large-|x| branch; closed unfixed, fix in #52126)

#### I1

The kernel rounds its result to bf16 unless the input is Float32 (calculate_i1,
`#ifndef INP_FLOAT32`), whatever dest_acc is: a 16-bit input's Float16 and Float32
rows measure that rounding against the fp32 golden, not kernel drift.
tracked: #52174 (input clamped at 88.5)

#### Identity

Keyed per format, not through a DEFAULT: Identity was never in BROAD_SWEEP_OPS, so it
has no sampled input-keyed row: every row is the sweep's but the two sampled
`output: Float32` / `output: Float16_b` wildcards, which answer only for cells no
input-keyed row names. Its steps are the pack path's, as for Abs and Neg.

#### Isfinite

The five is_* predicates write 1.0/0.0, so an exact result is the only correct one.
Measured on Float32 and Float16_b only, which is all the isinf_isnan sweep drives.
On a bf16 input at dest_acc=Yes the unpack hands the SFPU a NaN as the infinity of
its own sign (measured on Wormhole with both NaN signs driven: Identity
writes +inf on +NaN lanes and -inf on -NaN lanes; +inf and -inf arrive intact). So
is_nan, is_inf, is_pos_inf and is_neg_inf are not measured there; only is_finite,
0 for an infinity and a NaN alike, is gated on that cell.
0 ULP, 8 variants / 32768 lanes

#### Isinf

0 ULP, 6 variants / 24576 lanes

#### Isnan

0 ULP, 6 variants / 24576 lanes

#### Isneginf

0 ULP, 6 variants / 24576 lanes

#### Isposinf

0 ULP, 8 variants / 32768 lanes

#### Lgamma

tracked: #55356 (Stirling series with no argument shift)

#### LogicalNot

0 ULP, 8 variants / 32768 lanes

#### Polygamma

tracked: #52278 (wrong for x < 0)

#### ReluMax

ReluMax/ReluMin follow the Abs/Neg shape: exact on Float32, one step of output
rounding on the 16-bit formats.

#### Rpow

tracked: #52675 (no overflow guard, 16-bit Dest), #55129 (Veltkamp split overflows, 32-bit Dest)

#### SfpuAddcmul

SfpuAddcmul, SfpuAddcdiv and SfpuLerp are ternary. The ternary sweep
is small, so each of their sampled figures covers every variant that exists for that
format rather than a sample of a larger space; the counts say how many that is.

#### SfpuBinaryFmod

fmod and remainder share an implementation and measure identically.

#### SfpuBinaryRemainder

Shares SfpuBinaryFmod's implementation: see the note there.

#### SfpuElwGe

0 ULP, 147k-246k lanes; one of the six binary comparison predicates above.

#### SfpuElwGt

0 ULP, 147k-246k lanes; one of the six binary comparison predicates above.

#### SfpuElwLe

0 ULP, 147k-246k lanes; one of the six binary comparison predicates above.

#### SfpuElwLt

0 ULP, 147k-246k lanes; one of the six binary comparison predicates above.

#### SfpuElwNe

0 ULP, 147k-246k lanes; one of the six binary comparison predicates above.

#### SfpuElwadd

add/sub/rsub are single IEEE operations: one step on bf16 and fp16, and the Float32 65536
is the usual output-rounding figure. sub/rsub's fp16 cells read 140 steps when ranked
unflushed -- a near-cancelling a - b lands in the subnormal band the golden keeps and the
pack flushes -- and 1 ranked with that band flushed, as the gate ranks them.

#### SfpuElwdiv

Division is driven by two sweeps: the nightly test_eltwise_binary_sfpu_edges (its
registered poles, on Float16_b and Float32), where it is exact on both float columns,
and test_eltwise_binary_sfpu_div, whose own stimuli are what set these numbers. The
budget has to cover the harder of the two -- an earlier pass enrolled the 0 the edge
sweep supports, which the dedicated sweep then failed on hardware.

#### SfpuElwmul

Multiplication's op-wide row is tolerance: on the sample its Float32 worst
lane is 9.4e8 steps against a p99 of 65536, the near-zero spread a product produces.

#### SfpuIsclose

isclose writes 1.0/0.0 too. Mask does not -- its golden returns the data operand where
the mask is nonzero -- but it is exact for the same reason: it selects an input or zero
and never computes a new value.

#### SfpuXlogy

The second of the two tolerance-only binaries: see the note above SfpuElwpow.
xlogy's error is absolute and linear in x. It splits by nearly 5x between formats, so
applying Float16_b's atol to Float32 would accept five times what that format
produces.

#### SigmoidAppx

Carries GeluAppx's op-wide `atol`/`rtol`: see the note there.

#### Sign

Sign and Heaviside gate at 0 steps on every input format the sweep drives (Float16_b,
Float16, Float32), the strided Float32 one included: they are exact by construction, and the measured 0 is the gate there as
everywhere. The in-tree WH kernels
branch on `abs(v) == 0`, so -0.0 is answered as zero (fixed in #55306), and a
recurrence is a regression, not specified behaviour. The op-wide tolerance row answers
only what the sweep does not drive.

#### Signbit

0 ULP, 16 variants / 65536 lanes

#### Softplus

tracked: #51866 (returns 0 below -5, under #52178)

#### Square

One multiply, and one open question. x*x has rounding slack the sign-bit ops do not:
the golden rounds in float64, the hardware in the datapath, and ties can differ.
Its Float32 outputs measure 0 almost everywhere since UnarySFPUGolden stopped rounding
its reference to the input format (the dest_acc=Yes rows' 32768 steps, half a bf16
step in fp32 units, were that cast). What is left is dest_acc=No: a 16-bit Dest holds
the product at bf16 or fp16 precision, so a Float16 input reads the fp16 Dest's
subnormal flush there. The op-wide Float32 row keeps its sample only for the cells the
sweep does not drive.

#### Threshold

0 ULP, 116 variants / 2.10M lanes

#### UnaryEq

0 ULP, 8 variants / 32768 lanes

#### UnaryNe

0 ULP, 8 variants / 32768 lanes
