# `$advisor-challenger` — measure how much `$shard-advise` adds to a decoder already optimized without it

**Input:** a finished, tagged `optimized_decoder.py` from an arm or snapshot that never ran the advisor.
**Output:** the best *measured* decoder, plus a complete accounting of what the advisor said and what it was
worth. A **no-change outcome is a measured contribution of zero** — a result, not a failure.

## What this stage is

A contribution measurement. Hold the incumbent fixed, add one stage whose content is nothing but advisor
usage, and measure the difference. The frozen incumbent is the **control**, so

```
final_ms <= incumbent_ms
```

is not a safety property — **the delta is the measurement**. Freezing before the capture keeps the control
uncontaminated. Ties go to the incumbent as a conservative attribution rule.

Consequence: **anything that suppresses a candidate understates the advisor's contribution.** Prefer
measuring a doubtful candidate over dropping it.

## Division of labour

The advisor is a deterministic pass. It enumerates legal placements completely — every block-sharded
height×width and every width up to the core count, deduped by resulting shard shape. What it then selects
is not ranked on measured time: a per-op runtime estimate exists in the optimizer and is cached, but neither
selecting file carries a cost or runtime term and the chain-level quantity computed is **bytes**. So:

| the advisor supplies | you supply |
|---|---|
| the **op set** that is placed differently from the shipped graph | the **geometry** |
| the **direction** — into L1, out to DRAM, different grid | the chain extent, and every measurement |

**Treat an advised core count as one candidate, not as a recommendation.** Two independent reasons:

- *It is not ranked.* Selection has no latency term, so a small grid that fits the chain in L1 wins over a
  larger one that computes faster. A 4096-wide (128-tile) norm was advised 11 and 22 cores where 32 gives
  exactly 4 tiles per core and 64 is also legal — and both 32 and 64 are in the advisor's own candidate set.
  Nothing in it distinguishes an exactly-dividing grid from a padded one.
- *It is only legal inside the advisor's own plan.* The advisor re-derives the whole graph from
  DRAM-interleaved inputs; it never sees your shipped layouts. So a single advised placement dropped onto an
  otherwise-unmodified graph can violate a constraint of the shipped neighbour it never modelled —
  `per_core_K=9 is illegal` on a K=96-tile matmul came from exactly this. A rejection of that kind is a
  partial-application failure, not evidence the direction was wrong.

So: take the op **set** and the **direction**, derive your own geometry from tile divisibility, and always
measure at least one exactly-dividing grid. Test the advised value if you like, but never only at or below
it — a sweep bounded above by the advised value cannot distinguish "advisor was right" from "we never looked
higher".

`compute_config` / `math_fidelity` in the advised IR mirror the captured graph. They are not advice;
override them with your own fidelity decision.

## The advice is a whole-graph plan; you apply parts of it

This mismatch is the single most common source of confusion in this stage. The advisor re-derives the entire
graph from DRAM-interleaved inputs — it never sees your shipped layouts — so its output is **one plan whose
parts are only jointly valid**. You apply one chain at a time onto an otherwise-unmodified decoder.

Consequences:

- A placement that is legal in the plan can be **illegal in your graph**, because it violates a constraint of
  a shipped neighbour the advisor never modelled. `per_core_K=9 is illegal` on a K=96-tile matmul is this.
- **On a neighbour-constraint failure, extend the applied set before abandoning the direction.** Pull the
  neighbour's advised placement in too and re-measure: that is a chain extension, and it is the move most
  likely to make the pair self-consistent. Only after that fails is a compatible geometry of your own the
  next option, and only after *that* is the direction itself in doubt.
- An op the advisor reports as unplaceable may be unplaceable **only under its own upstream choice**. Check
  what it did to the producer before accepting that verdict; the shipped graph may already repair it with a
  conversion costing a microsecond or two.

## Chains

**A chain is a run of ops all resident in L1, preferably on one consistent shard spec.** A DRAM crossing is
the hard boundary; an internal L1 regrid is in-chain cost, not a split. The pricing below is what supports
this; `$optimize` step 6 / OPT-003 states the same definition, and the optimizer implements it as
`L1ChainConfig`, so do not reinvent it.

Detect boundaries from **conversion ops present in the profile**, never from differing grids: a grid change
is often absorbed by the consumer, so grid-differencing over-counts.

**The advisor declares its own boundaries, in `reshards[]` — not in `ops[]`.** These are the
`to_layout` / `to_memory_config` edges; in tt-mlir they are what the optimizer materializes on a chain edge.
So each shipped conversion can be compared against the advice on the same producer→consumer edge, and
`reconcile.py` does this as `advised_here`:

| shipped boundary | advice on that edge | meaning |
|---|---|---|
| present | **absent** | the advisor keeps this run in L1 — **the candidate**, and the pattern behind every win in this corpus |
| present | present | agreement on the boundary; nothing to screen |
| present | undetermined | no paired advised op either side; treat as unscreened, not as absent |

The match is on op *names*, so a repeated edge collapses to one key: presence is a strong signal, absence a
weaker one. It orders the worklist; it never substitutes for a measurement.

Indicative cost per conversion, measured on three Blackhole decoders. **Read your own numbers out of
`reconcile.py`'s `boundary` rows** — these scale with tensor size and differ by architecture; the table is
for calibration, not for arithmetic:

| move | device op | µs/op |
|---|---|---|
| L1→L1 regrid | `Reshard` | 1.4–1.9, roughly flat |
| DRAM→L1 | `InterleavedToSharded` | 1.2–2.6, scales with tensor size |
| L1→DRAM | `ShardedToInterleaved` | 1.5–3.5, scales, dearer direction |
| retilize | `Untilize` / `Tilize` | **6.7–10.0** — largest class, and absent from the traced graph |

Also: `reconcile.py` refuses a perf report that is not bounded to **one** replay, because it detects the op
sequence repeating. Three of the nine cells in the reference corpus committed un-bounded reports covering 2,
2 and 10 iterations, and every window share computed from those is smaller than the truth by that factor.
Bound the report with `--start-signpost` / `--end-signpost`.

**Chains do not interact appreciably** — two independent chains measured 0.13 and 0.16 µs of interaction.
So do **not** enumerate pairwise combinations. Grow one chain instead:

```
chain := maximal run of ops all resident in L1
loop:  pull in the adjacent op across the nearest DRAM crossing   (removes 1.2-3.5 us, or 6.7-10 retilize)
       then unify the grid inside the chain                        (each removed Reshard is 1.4-1.9 us)
       stop when neither move wins; cap the depth
```

The one combination worth enumerating is **across layer kinds** (step 6).

## Procedure

### 1. Freeze the incumbent, before running the advisor

Its own perf harness at `DECODE_BATCH`. **10 untimed warm-up replays, then n = 5 timed** — or blocks of 100
replays, which gave the tightest floor in this corpus (0.03 %). Record `repeats_ms` and take the
**median**, not the min: min-of-n is biased low by an amount that grows with n, so cells with different n
are not comparable.

`incumbent.json`: `decode_batch`, `requested_decode_batch`, `measured_at`, `repeats_ms`, `incumbent_ms`
(median), `harness`, `harness_scope` (what the harness measures end to end), `shipped_policy` and
`shipped_weight_dtypes` sourced from what **executed** — the final `tt-perf-report` CSV or the selected
candidate JSON, never `resolved_policy.constructor_defaults`.

Save the op-level `tt-perf-report` CSV per layer kind. The window it reports must be within ~5× of
`incumbent_ms`; a wider gap means the harness measures something other than the decode path.

### 2. Capture once per layer kind, at the shipped precision

Copy `scripts/capture_template.py`; construct the decoder with the **shipped policy**, not class defaults.
Pass `--pipeline-options allow-bf16-dram-sharded-matmul=true` if any traced weight is bf16. `captured_at`
must be after `measured_at`.

Record the **layer share** of any kind you cannot capture. `sparse_matmul` and the SSM / gated-delta ops
are terminal in the tracer; if such a kind dominates, *"the advisor cannot reach N of M layers"* is the
correct result and must be stated, not discovered later.

### 3. Reconcile — `scripts/reconcile.py`, once per layer kind

It partitions the measured window so that every device op lands in exactly one bucket and the buckets sum
to 100 %: `chain`, `boundary`, `dram_resident`, `agrees_with_shipped`, `untraced`. It fails loudly rather
than emitting a gap.

Read four things out of it:

- **the ranked chain list** — ordered by `advisor_removes_us` (the µs of shipped conversions the advice does
  *not* place on that edge), then by total conversion value; *never* by the advised ops' window share. A
  chain whose ops total under 1 % of the window can be worth several per cent once its DRAM round trips are
  gone — in this corpus a chain at 0.948 % op share won 5–6 %.
- **`advised_boundaries`** — how many shipped conversions the advisor drops, agrees with, and leaves
  undetermined, with the µs. `us_advisor_drops` is the stage's upper bound on conversion-removal value;
  screening cannot beat it, so if it is small, say so early rather than after nine measurements.
- **`material_ops_on_le_2_cores`** — a material op left on ≤2 cores needs a measured attempt or a quoted
  hard error. Nothing else discharges it.
- **`agrees_with_shipped`** — where the advisor independently re-derived your config. Since it is blind to
  your layouts, agreement is genuine re-derivation, and on one decoder it was 82 % of the window. Its
  marginal contribution here is zero by construction; **report the number as a headline**, and do not read it
  as wasted advice. How much a from-scratch optimization it would have saved is a different experiment.

Advised ops pair with device classes automatically, so an abort means an input problem — an unbounded perf
report, a wrong file, a window that cannot be the decode path. Fix the input, not the output: hand-authoring
this file is the one failure the gate cannot detect for you.

### 4. Screen, in the order the reconciliation gives

Each chain as one unit, one variable per measurement, against the frozen incumbent. Every chain ends with
`repeats_ms` and a verdict, or `below_threshold` with its conversion value. Screen `dram_resident` rows
too — *"leave this in DRAM"* is advice, and de-sharding an op has won here.

Every kept candidate: its own op-level CSV as `perf_report`, and the incumbent's oracle at the incumbent's
own PCC bar. Name the oracle and say if it is synthetic. **A faster decoder that fails its oracle is a
regression with a good number.**

Screen DS-matmul advice **last**. It has not won a measurement in this corpus, and where it agrees with a
shipped DS config there is nothing to screen.

### 5. Decide by non-overlap

Ship a change iff **every candidate repeat beats every incumbent repeat**. No noise floor, no min baseline.
This is only comparable across cells if n is fixed — the false-positive rate is `1/C(2n,n)`, so 5 % at
n=3 against 0.40 % at n=5. **Confirm the winner in a fresh process** before shipping: cross-process
variance is otherwise unmeasured, and per-process work happens once per process.

### 6. Combine across layer kinds

Where the model has several layer kinds and the reported latency is a weighted composite, the candidate
space is the **product** of per-kind winners, not the union of one-kind-varied sets. The composite is
arithmetic over independently measured series, so these cost no extra device time. Record every measured
set with its `chains`, `measured_ms`, `repeats_ms` and `oracle_passed`; ship the best measured set.

### 7. Ship

Write the winner into `tt/optimized_decoder.py` and **keep every losing knob, default-off**, so rejected
candidates stay re-measurable. `final.json`: `outcome`, `changed`, `final_ms`, `incumbent_ms`,
`repeats_ms`, the winning config, the oracle, and `iterations[]`. If nothing won, ship the incumbent
unchanged and say so with the numbers.

`README.md`: the accounting from step 3, what was screened with its number, what was not and why, and the
unreachable share.

## Iterating

You may apply what won and go again — but **re-profile first** and re-run `reconcile.py` on the new CSV;
the old ranking describes a graph that no longer exists. Record `reranked_from`. Re-capture only after a
topology rewrite that changes an op's shape; a dtype change is not one. Cap at 3 captures per layer kind.

Extension is not free: growing L1 residency hits capacity walls, and a bad placement in this project wedged
PCIe and needed a `tt-smi` reset. Do not run extension experiments alongside a measured stage.

## What this stage cannot reach

`ttnn.sparse_matmul` and the SSM / gated-delta ops are terminal in the tracer, so a model whose time is in
its experts or its linear attention is largely invisible here — state the share rather than implying
coverage.

**Conversion time this stage must leave alone.** DRAM round trips are the thing worth minimising, and
`reconcile.py` splits them by who wants them. Only `us_advisor_drops` is yours to screen. Two other pots are
real cost and out of scope:

- **`us_advisor_agrees`** — the advisor places a conversion here too. Removing it may well win, but the win
  would be your idea, and booking it against the advisor contaminates the measurement.
- **`us_not_advisor_comparable`** — `ReshapeView`, `FillPad`, `Copy`. Not placement edges, so the advisor
  never expresses them. Not small either: six `ReshapeView` kernels ran 8–19 µs each on 110 and 16 cores in
  one cell, 88.5 µs of a 1355 µs window.

Report both with their µs and hand them to `$optimize`. Eliminating conversions the advisor does not propose
is a better-paying activity than this stage — it is just a different experiment. Sweeping expert and norm grids **directly**, without the advisor, is a separate and often
better-paying activity.
