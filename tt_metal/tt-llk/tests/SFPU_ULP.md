# SFPU accuracy: the ULP sweep, and how to enrol an op in it

Every SFPU op declares how closely its output must match its golden, in
`python_tests/helpers/sfpu_accuracy_budget.yaml`. For most ops that is a tolerance; for
the ops enrolled here it is a **step budget**, "every element is within N representable
values of the reference", the same strictness at every magnitude where `atol=0.05` is 6
bfloat16 steps at 1.0 and 0.01 at 512. Most unary budgets come from the exhaustive
sweep, which feeds every normal value of a 16-bit input format and its zero, or a stride
of `Float32`; the `Signbit`, isinf/isnan and threshold-family rows are measured on their
functional drivers' hand-built stimuli (`MEASURED_ON_SWEEP` in
`test_sfpu_accuracy_budget.py`). This document is how you add an op to the first group.

## Quick start

```bash
cd tests/python_tests

# Measure the op over the whole format (a stride of Float32) and rewrite its block.
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --ulp-emit --compile-producer -n 8
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --ulp-emit --compile-consumer

# Read the diff, then gate on it.
git diff helpers/sfpu_accuracy_budget.yaml
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --compile-consumer
```

`--op` matches the op name exactly; `-k Exp` would also run and rewrite `Exp2`,
`ExpWithBase`, `Expm1` and `Expm1Cw`.

`--ulp-emit` **writes the checked-in table**, once, at the end of the session, merging
every xdist worker's measurements first. It raises `ulp_sweep.EmitRefused` and writes
nothing unless the session ran on Wormhole, ran to the end with no failure, and measured
every cell of each `(input, output)` grid it touched (the producer half and
`--collect-only` measure nothing and say so). Of a measured op it replaces every row of
each measured pair and keeps the rows no measurement re-derives: one naming an `arch`,
declaring `atol`/`rtol`, or naming a pair this run did not measure. Read the diff before
committing it.

## What the sweep feeds

| format | how |
|---|---|
| `Float16_b` | all 65,025 values: every normal one, and zero |
| `Float16` | all 61,441 values: every normal one, and zero |
| `Float32` | 65,024 samples, one per bfloat16 cell, each binade an equal share |

Those are the formats the SFPU computes in, as input and as output; a block float is not
swept and no row names one, since its error is the unpacker's and packer's. No subnormal,
infinity or NaN is fed: the unpack flushes a subnormal, so a lane there would measure the
flush rather than the op. The domain is not clipped to the op's safe range: undefined
inputs reach the hardware and are masked out afterwards. An exponent-B input packed to
`Float16` needs a 32-bit Dest on Wormhole and Blackhole, so those cells run at
`dest_acc=Yes` only (`ulp_sweep.sweep_cells`).

The binary, ternary and scalar rows, and the hand-built unary sweeps, are measured over
their own drivers' stimuli, and those drivers gate on the whole contract. No functional
driver sweeps a `Float16` input or output on Wormhole or Blackhole, so nothing in CI
gates the `Float16` binary rows: re-measure before relying on one.

## Lanes the sweep does not count

`ulp_sweep.measurable_mask` drops the lanes no step count can describe: either side NaN,
the two sides disagreeing about being non-finite, an input the unpack flushes or
saturates (a `Float32` input into a `Float16` Dest, below 2^-14 or past 65504), an
answer an fp16 Dest lost under a wider output (`dest_holds`: NaN or the golden's sign in
the Dest's top binade where the golden is past 65504, or a flush across 2^-14), and the
tensor's zero padding. `claimed_lanes` then keeps only where the op claims an answer: the
whole format less the undefined side of each singularity in
`sfpu_domains._OP_SINGULARITIES`, the non-positive integers for the gamma family, and
`Sin`/`Cos`/`Tan` past 65504, where the range reduction breaks.

A non-finite disagreement is a *failure*, not a non-question: `nonfinite_failures`
reports it over the claimed lanes, and the cell is written `not measurable` instead of
measured. Excused there: a NaN golden, a golden past the output's range answered by a
saturated store (NaN, an infinity of the golden's sign, or the fp16 pack's clamp to
±65504), an infinite golden on the singularity point itself, and the lanes a tracked
defect names in `ulp_sweep._KNOWN_NONFINITE_LANES` (`Celu` answers `inf` for x in
65408..65504 on a 16-bit `Float16` Dest, #58607). A named lane is excused only for its
non-finite disagreement, on its cell, and the gate fails the day none of an entry's
lanes disagrees any more, so an entry cannot outlive its fix.

Subnormal *outputs* are ranked with the band flushed, on `Float16` too: the golden keeps
fp16 subnormals the pack does not reproduce.

## Enrolling an op

1. **Check it can be gated.** The op needs a format the SFPU computes in
   (`ulp.ULP_FORMATS`) as input and output, a place in `sfpu_domains.sfpu_unary_ops()`,
   and a registered domain.
2. **Give it a block.** The sweep collects the ops the table names, and an op with no
   rows fails to load, so a new op needs one row by hand, keyed on a swept pair or the
   emit keeps it instead of replacing it:

   ```yaml
   MyOp:
   - {input: Float16_b, output: Float16_b, metric: TOLERANCE, note: placeholder}
   ```

3. **Measure it** with the two `--ulp-emit` commands above.
4. **Read what it wrote.** Every row records what was measured, and a figure *is* the
   gate, with no headroom over it; a row's `note` is for people. The shapes:

   - `domains: [{low: L, high: H, ulp: U}, …]` — every swept cell with a per-element
     figure (a cell the sweep cannot measure, or where no entry would carry a figure,
     gets one of the plain rows below): one list over the whole swept range ("Where an op holds its figure", below). An entry within
     the output's usable ceiling is gated at exactly `U` steps; one past it is measured,
     not gated, and the headroom report holds it to its figure; one with no `ulp` has too
     few ranked lanes to draw an edge on.
   - `near_zero_atol: A` beside the list — a floor. The step count is a difference of
     bit-pattern ranks, so a 0 answered for a 2e-9 golden reads as 2^29 fp32 steps, and
     one such lane would demote the cell. Lanes under 1 % of the cell's largest golden
     (and under `A/0.01`) are judged on absolute error instead; `A` is exactly the
     largest such error, derived by `ulp_sweep.near_zero_floor` and written as measured,
     and every entry's figure is over the lanes outside it. The emitter grants up to
     `EMIT_MAX_NEAR_ZERO_ATOL` (1e-3); more is a kernel's cut-off rather than its
     rounding (`Softplus` answers 0 below x = -5), and stays a hand decision.
   - `measured: N`, no `metric` — a row a functional driver sampled; it holds the cell
     to exactly `N` steps.
   - `metric: TOLERANCE, measured: N` — a cell past `usable_budget_ceiling`, where a
     step budget would no longer be tighter than the tolerance it replaces. The op keeps
     tolerance + PCC there, and the headroom report holds it to `N`.
   - `metric: TOLERANCE, measured: N, nonfinite: L`, noted "not measurable" — `L` lanes
     disagreed with the golden about being finite; `N` is the worst of the lanes that
     were measurable. On a gateable output the row must be acknowledged with its cause
     in `test_sfpu_accuracy_budget._UNMEASURABLE_CELLS_ACKNOWLEDGED`, unless the cause is
     a tracked defect on a handful of inputs, which belongs in `_KNOWN_NONFINITE_LANES`
     so the rest of the cell keeps its gate.

5. **Gate on it.** Re-run without `--ulp-emit`, then the host guards:

   ```bash
   ../run_hardware_free.sh                                # tests/hardware_free, accuracy included
   CHIP_ARCH=wormhole pytest test_ulp_budget_diff.py -q   # the guard's command-line tests
   ```

   Both from `python_tests`. Run `hardware_free/` through the script: pytest started
   on it directly finds the tt-metal root's `pytest.ini` instead of this suite's, and
   without `LLK_HARDWARE_FREE` the plugin opens a device.

## The rules the table enforces

- **Typed data with one writer.** Every row is a `ulp_table.Row`, an enum is written by
  its member name, and `dump_table` writes the whole file in one order; a hand edit must
  leave the file as the writer would
  (`test_the_table_on_disk_is_exactly_what_the_serializer_writes`). To normalise one:
  `python -c "from helpers.ulp_table import *; write_table_file(load_table(), TABLE_PATH)"`.
  A refused row raises `TableError`, whose `problem` says why.
- **The measurement is the number.** A ULP row's gate is its `measured` figure, and the
  loader refuses one without it. A worse measurement is a raised gate, recorded by a
  re-emit in a change of its own; `python -m helpers.ulp_budget_diff diff --base <old>
  --head <new>` lists every cell a change loosens.
- **Most specific key wins.** Key fields are `input`, `output`, `approx`, `dest` and
  `arch`, all optional; two rows matching one variant equally specifically are refused.
- **A row holds for the configuration it was measured in**: `FAST_MODE(No)` and
  `CLAMP_NEGATIVE(True)` compiled in, with no key axis for either.
- **The list is the statement.** A `domains` row carries no `measured` or `metric` and
  pins the `input` and `output` it was measured on. An entry never widens or narrows
  what the op claims; it reports where the claim holds, and at what figure.
- **Budgets do not transfer between architectures.** Every unkeyed number is a Wormhole
  measurement; elsewhere an op falls back to tolerance unless a row names that `arch`.
- **An exact op may not carry a wide budget.** Sign-bit ops, copies, integer results,
  predicates and constant fills are the canaries:
  `test_an_exact_op_never_carries_a_wide_budget` holds their budgets to the pack path's
  cost, and every swept cell of one must stay on the step budget.
- **Not enrolling** is one of two things: no row (the op keeps today's tolerance and is
  not swept; `test_every_unary_op_is_enrolled_or_excused` fails until one is chosen), or
  `sfpu_domains._UNARY_OPS_NOT_SWEPT`, with why it cannot be swept.

## Where an op holds its figure: domains

```yaml
Tanh:
- {input: Float16_b, output: Float16_b, approx: 'Yes', dest: 'No', near_zero_atol: 0.00042724609375, domains: [{low: -.inf, high: -1.0, ulp: 3}, {low: -1.0, high: -0.0078125, ulp: 10}, {low: -0.0078125, high: 0.0078125, ulp: 0}, {low: 0.0078125, high: 1.0, ulp: 10}, {low: 1.0, high: .inf, ulp: 3}]}
Log:
- {input: Float16_b, output: Float16_b, dest: 'No', domains: [{low: -.inf, high: 0.0}, {low: 0.0, high: .inf, ulp: 1}]}
```

Approximate `tanh` is 3 steps everywhere except on `[-1, 1)`, where it is 10, past
bfloat16's ceiling of 6: that entry is why the cell as a whole is not gated, while the
two others are gated at 3. `log` answers nothing below 0 and is 1 step from 0 up.

- An entry holds the lanes whose input, *as the kernel received it*
  (`ulp_sweep.received_inputs`: truncated to the 16-bit Dest for a `Float32` input at
  `dest: "No"`), satisfies `L <= x < H`; `U` is the worst ranked lane there outside the
  row's floor. An entry with no `ulp` has fewer than `DOMAIN_MIN_BIN_LANES` ranked lanes
  in each of its binades; its strays are held to the row's largest figure.
- The list is contiguous and complete: `-.inf` to `.inf`, each `high` the next `low`,
  every interior edge `0` or a power of two.
- The emitter draws it from the worst ranked lane per signed binade of the received
  input (`ulp_domains.partition`): split where it crosses the ceiling, a binade under
  `DOMAIN_MIN_BIN_LANES` folded into its neighbour, a gateable run under
  `DOMAIN_MIN_LANES` folded into the neighbour at least as loose. A lane the floor
  covers counts as ranked at 0 when the list is drawn, so the near-zero band draws no
  entry of its own.
- The whole-cell contract `accuracy_contract` hands callers that do not read domains is
  derived: gated at the largest figure when every entry fits the ceiling, tolerance
  otherwise. The hand-written `atol`/`rtol` anchors (`GeluAppx`, `SigmoidAppx`) are
  op-wide rows the emitter keeps beside the lists, which such a cell falls back to.
- The sweep gates entry by entry, the headroom report judges entry by entry from the
  reading the `--ulp-measure` record carries per entry, and the PR guard judges segment
  by segment over the union of both revisions' edges, so a list redrawn with the same
  figures is no change.

## Gotchas

- **Any host pytest run in `tests/` wipes `/tmp/tt-llk-build`**: chain
  `--compile-producer` and `--compile-consumer`, and run host suites afterwards.
- **The sweep is marked `accuracy`**, which every LLK workflow deselects; run it by name
  or `-m accuracy`.
- **Both runs take the same op set**: every unary op with a block in the table, whatever
  its rows say. `--ulp-emit` changes what is written, not what is collected.
- **A plain gate run does not judge tolerance cells**: it skips them with the measured
  maximum in the skip reason. To hold them to the figures their rows record:

  ```bash
  CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py -n auto --ulp-measure=ulp.jsonl
  python -m helpers.ulp_budget_diff headroom --table helpers/sfpu_accuracy_budget.yaml --measured ulp.jsonl
  ```

  Each line of `ulp.jsonl` is one `ulp_table.Measurement`.
- **`--ulp-measure` can fail the sweep**: the sweep's records are required, since the
  headroom report judges only the cells it reads, so a sweep cell whose record is not
  written fails (`RecordDropped`), and a path that cannot be prepared is a usage error.
- **`CHIP_ARCH` must be set**, and `--ulp-emit` refuses to run off Wormhole, because the
  emitter writes no `arch`.

## Where things live

| file | what |
|---|---|
| `python_tests/test_unary_sfpu_ulp.py` | the harness: one device run per variant |
| `python_tests/helpers/ulp.py` | the metric: `ulp_distance`, the elementwise verdict, the near-zero floor |
| `python_tests/helpers/ulp_sweep.py` | what the sweep feeds, what it masks, and the emitter |
| `python_tests/helpers/sfpu_accuracy_budget.yaml` | the table |
| `python_tests/helpers/ulp_table.py` | the table as typed rows, its reader and writer, and the `--ulp-measure` records |
| `python_tests/helpers/sfpu_accuracy_budget.py` | how a row is resolved |
| `python_tests/helpers/ulp_domains.py` | the `domains` list and the rule that draws it; `Problem` and `TableError` |
| `python_tests/helpers/ulp_budget_diff.py` | the PR guard and the headroom report |

## Notes on the table

A row's own `note` says what is particular to that cell; what follows is what the rows
do not say.

- **The pack step.** Sign-bit ops, copies and selections are exact in fp32 and read one
  step on a 16-bit output from a wider input: the output rounding on the way out of
  Dest, not the op. The narrowing rows carry exactly that measured step.
- **`SfpuElwpow` and `SfpuXlogy`** stay on the tolerance metric: their error is a
  property of the op's composition and grows with the operands however the domain is
  drawn. pow's error is relative and flat, so one `rtol` suits every float column;
  xlogy's is absolute and linear in x and splits nearly 5x between formats, so each is
  keyed per output.
- **`GeluAppx` and `SigmoidAppx`** share one op-wide `atol: 0.13` row: a coarse
  3-segment LUT with one cause, so a retune that moves one should move the other in the
  same diff. Their per-cell rows are numberless and do not shadow that anchor.
- **`I1`** rounds its result to bf16 unless the input is `Float32` (`calculate_i1`,
  `#ifndef INP_FLOAT32`), whatever `dest_acc` is: a 16-bit input's `Float16` and
  `Float32` rows measure that rounding, not kernel drift.
- **`SfpuBinaryFmod` and `SfpuBinaryRemainder`** share an implementation and measure
  identically. **`SfpuMask`** selects an input or zero and never computes a value, so it
  is exact like the predicates.
- **`Erfc`** is tracked by #51137 (input clamp, under #52178), the one issue no code
  names.
- The not-measurable rows are explained op by op in
  `test_sfpu_accuracy_budget._UNMEASURABLE_CELLS_ACKNOWLEDGED`, and the lanes a tracked
  defect names in `ulp_sweep._KNOWN_NONFINITE_LANES`.
