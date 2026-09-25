# SFPU accuracy: the ULP sweep, and how to enrol an op in it

Every SFPU op declares how closely its output must match its golden, in
`python_tests/helpers/sfpu_accuracy_budget.yaml`. For most ops that declaration is a
tolerance. For the ops enrolled here it is a **step budget**: "every element is within
N representable values of the reference", checked against *every value the input format
can take*.

This document is how you add an op to that second group.

## Quick start

```bash
cd tests/python_tests

# What the hardware measures for your op, over the whole format.
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py -k MyOp --ulp-emit \
  --compile-producer -n 8
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py -k MyOp --ulp-emit \
  --compile-consumer

# ...which rewrites your op's block in the table. Read the diff, then gate on it.
git diff helpers/sfpu_accuracy_budget.yaml
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py -k MyOp --compile-consumer
```

`--ulp-emit` **writes the checked-in table**. It refuses to do so unless you are on
Wormhole, you are the xdist controller, and no test in the session failed — but it is
still a deliberate act, so read the diff before committing it.

## Why a step budget rather than a tolerance

`atol=0.05` is about 6 bfloat16 steps at 1.0 and about 0.01 at 512. A tolerance loose
enough to pass an op's tail is blind in the middle of its range, and PCC stays above
0.99 through error levels no consumer would accept. A step count is the same strictness
everywhere.

And the sweep measures a different number than a functional driver does. The drivers
sample a few thousand points from an op's *safe* domain; a budget measured that way
describes the sample. Approximate `Reciprocal` reads **1 ULP** on `uniform(0.1, 1.1)`
and **128 ULP** over every bfloat16 value there is.

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

`Float32` has 2^32 values and one device run holds 2^16, so it cannot be enumerated. The
sweep strides the format's total order instead. Every binade holds the same number of
representable values, so each gets an equal share: one run reaches 261 binades from 0 to
3.4e38. Ask `ulp_sweep.is_exhaustive(input_format)` if you need to know which you got.

The domain is deliberately **not** clipped to the op's safe range. Undefined inputs are
swept and then masked out of the statistics, so they still reach hardware.

## Lanes the sweep does not count

`measurable_mask` drops four kinds, none of them a budget question:

- **either side NaN** — an op undefined at an input lands here on its own;
- **the two sides disagreeing about being non-finite** — `sin(2.6e28)` returning `inf`
  against a golden of `-1`; one such lane ranks at ~48,000 steps;
- **subnormal inputs** — the hardware flushes them and the golden does not, so
  `ceil(5.69e-39)` is 1 in the model and 0 on silicon: 16,129 bfloat16 steps. Measured,
  that class alone was the whole of `Ceil`'s, `Floor`'s and `Sqrt`'s apparent error;
- **the sweep's own zero padding** — the tensor is 65,536 lanes and bfloat16 has 65,279
  finite values, so the last 257 are padding rather than data.

The second kind is a *failure*, not a non-question, so `nonfinite_failures` reports it
separately — bounded to the op's registered domain, because outside it an op is not
claiming anything.

## Enrolling an op, step by step

### 1. Check it can be gated at all

A step count needs a per-element ULP, which means a float output. `ulp.has_ulp_gate(fmt)`
is the authority. Integer formats want bit equality; `Bfp4_b`, `Bfp2_b` and the MX
formats keep their block-aware lattice compares.

Your op also has to be in `sfpu_domains.sfpu_unary_ops()` and have a registered domain,
or the sweep has nothing to feed it.

### 2. Give it a block in the table

The emitter passes an op's key line through verbatim, so a *new* op needs one by hand
first. Add the name and nothing else:

```yaml
MyOp:
  - {out: Float16_b, max_ulp: 1}  # placeholder, replaced by the emit run below
```

### 3. Measure it

Run the two commands under **Quick start**. The emitter rewrites your op's block with
what the hardware reported, and preserves anything it did not measure — a `Float32` row
from an older run, an `arch:`-keyed entry, a row carrying a `near_zero_atol` floor.

### 4. Read what it wrote

```yaml
MyOp:  # measured by: exhaustive Float16_b/Float16/Bfp8_b/Bfp4_b + strided Float32 sweep, wormhole, 2026-09-23, except where a row says otherwise
  - {in: Float16_b, out: Float16_b, max_ulp: 2}  # max 1 ULP
  - {in: Float16, out: Float16_b, metric: tolerance}  # max 14337 ULP, budget would be 15771 > 6-step ceiling
  - {in: Float16_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized, so tolerance
```

Three verdicts:

- **`max_ulp: N`** — enrolled. `N` is the measurement plus 1.1x headroom, floored at 1
  except where the measurement was 0.
- **`metric: tolerance`, "budget would be N > C-step ceiling"** — past
  `usable_budget_ceiling`, so a step budget would no longer be *tighter* than the
  tolerance it replaces. The op keeps tolerance + PCC on that cell and the number is
  recorded so nobody re-derives it.
- **`metric: tolerance`, "block-quantized"** — a block float output. The sweep
  enumerates a format in value order, so sixteen adjacent values share a `Bfp8_b` block
  and the exponent fits all of them: the best case for quantization, not a
  representative one. `Abs` reads **393 steps** there from random mixed-magnitude blocks
  and **3** from the sorted sweep, so neither number is enrollable.

### 5. Gate on it

Re-run without `--ulp-emit`. Green means the budgets in the table hold over the whole
format. Then run the host guards, which check things the sweep cannot:

```bash
pytest test_sfpu_accuracy_budget.py test_ulp_sweep.py -q
```

## The rules the table enforces

- **No number is a guess.** The trailing comment on a row is the measurement it came
  from. A budget may only be *raised* by re-measuring and updating that comment in the
  same change. `llk-sfpu-ulp-budget-guard` fails a pull request that raises one without
  it; the `ulp-budget-raise-approved` label is the override, and it still reports which
  rows it admitted without a fresh measurement.
- **Most specific key wins.** A row's key fields are `in`, `out`, `approx`, `dest` and
  `arch`, all optional. Two rows matching one variant equally specifically are an
  authoring error, not a tie-break, and the loader refuses them.
- **Budgets do not transfer between architectures.** Every unkeyed number was measured
  on Wormhole. Off it, an op falls back to tolerance unless a row names that `arch`
  itself. Re-measure before trusting any of it on Blackhole.
- **An exact op may not carry a wide budget.** Sign-bit ops, copies, integer results,
  predicates and constant fills are the flakiness canaries: if one fails, the golden or
  the datapath moved. `test_an_exact_op_never_carries_a_wide_budget` holds them.
- **`Bfp8_b` charges for block quantization.** A budget there is denominated in bfloat16
  steps, two to one `Bfp8_b` step, and does not forgive the shared exponent. Only ops
  whose result a shared exponent represents exactly are enrolled on it.

## Not enrolling an op

Two ways, and they mean different things:

- **No row in the table** — the op resolves to today's tolerance and the sweep skips it.
  Fine as an interim state, but `test_every_unary_op_is_enrolled_or_excused` will fail
  until you pick one of these two deliberately.
- **`sfpu_domains._UNARY_OPS_NOT_SWEPT`** — the op cannot be swept at all. Say why in a
  comment next to it.

## A budget that is nearly right

If an op is exact over most of its range and catastrophic in a narrow band near zero,
the measurement will be dominated by the band and the cell will demote to tolerance.
`near_zero_atol` is the floor under the budget for exactly that: the lanes where the
reference crosses zero, judged on absolute error instead of steps. `Gelu` and most of
`Erfinv` are gated that way. The emitter cannot re-derive a floor, so it refuses to
regenerate a row carrying one — you set those by hand, with the measurement beside them.

## Gotchas

- **Any host pytest run in `tests/` wipes `/tmp/tt-llk-build`.** Chain
  `--compile-producer` and `--compile-consumer` in one go; if you run a host suite in
  between, recompile.
- **The sweep is marked `accuracy`**, which every LLK workflow deselects. Run it by name
  or by `-m accuracy`. `nightly` would *not* have kept it out of `llk-e2e`.
- **`--ulp-emit` widens the op set** to every measurable unary op, not just the enrolled
  ones. Restricting it to ops that already carry a budget is the loop the sweep exists
  to break. Pass the flag to the **producer too** -- without it the producer collects the
  narrow gating set and the consumer fails on missing ELFs, not on budgets.
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
| `python_tests/helpers/ulp_budget_diff.py` | the CI comparison, and the headroom report |
