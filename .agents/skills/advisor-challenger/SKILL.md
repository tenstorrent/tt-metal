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

**The advised core count is a *detection* signal, not a *selection* one.** Two independent reasons:

- *It is not ranked.* Selection has no latency term at any level, so a small grid that fits the chain in L1
  wins over a larger one that computes faster. For normalization ops the core-count term is worse than
  unranked: the rulebook overwrites it with the *input* operand's grid volume, which on a decode shape
  cannot vary with the candidate at all. Scored on the three ladders anyone has measured, the advised grid
  reached **82 %** of the achievable win while a fixed rule that never consults the advisor reached 99 %.
- *It is only legal inside the advisor's own plan.* The advisor re-derives the whole graph from
  DRAM-interleaved inputs; it never sees your shipped layouts. So a single advised placement dropped onto an
  otherwise-unmodified graph can violate a constraint of the shipped neighbour it never modelled —
  `per_core_K=9 is illegal` on a K=96-tile matmul came from exactly this. A rejection of that kind is a
  partial-application failure, not evidence the direction was wrong.

So: trust it for **which op** and **which direction**, and get the geometry from the ladder —
`reconcile.py --legal-grids <tensor width in tiles>`, swept on **both sides** of the advised value. The
response curve is not monotonic: on one decoder the advised value sat at a local *maximum* with the optimum
below it, and the two cells that swept only at and above the advice both shipped the wrong rung.

**And the win in this class is a threshold, not a gradient.** Measured on three decoders, a reduction's
response to core count is flat over a wide middle and falls off a cliff at one core: 8→44 cores varied by
0.2 % where 1 core cost 13.7 %. Almost all the value is the **first step off one core**. You do not need the
optimum — you need to notice the op is on the cliff (which `reconcile.py` now tells you before you spend any
device time) and land anywhere on the ladder.

**Which middle rung you pick is worth about 0.1 pp, not 1 pp.** Measured on the first v3 cell: 16 / 22 / 32
cores gave 0.543590 / 0.544064 / 0.544007 ms — a spread of 0.08 pp, with 22 and 32 inside each other's noise,
against 1 core at 0.553349. So sweep the ladder to confirm you are on the plateau and to find the knee, not
because the rungs differ much. An earlier draft of this file said 1 pp; that came from a summary sentence
rather than from the measurement beside it.

⚠ **Before calling a low core count starved, ask what the op's sharding semantics are.**
`nlp_create_qkv_heads_decode` height-shards over *batch*, so its core count **is** the batch size —
exactly, in all 23 rows of the reference corpus. One core at batch 1 is the op's own semantics and the
advisor advising 1 is correct. This was flagged as an opportunity in that corpus and disproved.

`compute_config` / `math_fidelity` in the advised IR mirror the captured graph. They are not advice;
override them with your own fidelity decision.

## The advice is a whole-graph plan. Apply it whole, then ablate.

**This is the rule that changed most in v3, and it is the one with the largest measured effect.** The advisor
re-derives the entire graph from DRAM-interleaved inputs — it never sees your shipped layouts — so its output
is **one plan whose parts are only jointly valid**. v2 screened it one chain at a time, building up from the
incumbent, and never applied the plan as written. On the one reference cell where that counterfactual could
be measured afterwards:

| configuration | median ms | Δ | differential PCC |
|---|---|---|---|
| frozen incumbent | 0.807535 | — | — |
| what the cell shipped, after 14 measurements | 0.768104 | −4.88 % | 1.0 |
| **the advisor's plan, implemented from its IR** | 0.723320 | **−10.43 %** | **1.0** |
| that plan plus its advised 11-core norm | **0.663507** | **−17.84 %** | 0.99999107 |

**3.7× what building up produced, and the −10.43 % is bit-identical**, so no correctness rule blocked it.
Nobody tried it. Across the corpus the three best outcomes were the three cells whose *first* candidate was
the advisor's placement, and the four worst never applied it.

**So the procedure is:**

**⚠ Bounded by what the decoder can express, and that bound is a measurement.** The advice is per-op
memory configs and shard shapes; the decoder exposes **policy knobs**. Where no knob exists, the placement
cannot be applied at all — the first cell to run this returned `hard_error: no generic final_ir-to-decoder
execution bridge exists`, and in the reference corpus **0 of 15 cells** applied a plan, the one success being
a hand-written per-model patch. So apply the **maximal expressible subset**, and make the remainder a number:

> `inexpressible[]` — every advised placement with no knob, each with **its op cost from the profile**. The
> cost is what stops this being a loophole: a cell that declares everything inexpressible has to show that
> the excuse covers most of the window, and that is then a finding about the decoder's knob surface rather
> than about the advisor.

```
1. apply_all = every advised placement THE DECODER CAN EXPRESS, at once, built from `advised_plan.ops` in
   the reconciliation, which reads final_ir.mlir. NOT from report.json: it has no shard shape, and a plan
   rebuilt with a guessed shard width is a different plan (a guessed (32,48) for a specified (32,64) cost
   one analysis a week and produced a retracted "the advice is illegal" finding). Adding a knob is IN scope
   when it is a memory-config or program-config change; out of scope when it needs a graph rewrite -- say
   which
2. first drop every op in `unfixable_ops` -- the advisor has already declared those impossible, with the
   exact runtime error it got from tt-metal's own constraint machinery
3. if what remains will not run, remove ONLY the failing item, and record the isolated single-op test that
   names it. A TT_FATAL while implementing advice is evidence about your reconstruction until an
   isolated single-op test in the exact advised config says otherwise
4. measure. apply_all is the FLOOR, not the ceiling
5. ablate one advised item at a time. An item whose REMOVAL is faster is a finding about the advisor that
   no build-up order can generate; an item that changes nothing gets dropped with a measurement behind it
6. build up from the incumbent only for what apply_all could not reach
7. re-advise after any change that adds, removes or reorders ops -- `ttnn-advise` costs ~18 s end to end,
   less than one harness measurement. Pure memory-config changes leave the advice byte-identical, because
   the advisor discards input memory configs and re-places everything; that is a property of the change,
   not a rule
```

Two mechanisms make build-up lose, and both are worth knowing. **Sub-floor chains never get tested** — 60 %
of the disagreed-on cost in the reference corpus sat in `below_threshold` chains, individually unmeasurable
and collectively obvious. And **"not tried" becomes indistinguishable from "tried and lost"**: four cells'
unproven deviations look identical in the artefacts to a measured rejection. Step 5 is what fixes that.

Consequences of the plan being jointly valid:

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
this. **This definition is deliberately broader than `$optimize`'s**, which asks for a chain on *one
consistent shard spec* (step 6 / OPT-003, "preserve the decode residual layout through norms and residual
adds"). Here an internal regrid is priced, not forbidden: tt-mlir models the chain as `L1ChainConfig`, which
carries a per-edge `memReconfigEntryMap`, so a reconfigure inside a chain is a cost the optimizer itself
represents. Use the broader definition and read the regrid off the profile — but do not reinvent either.

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

## One discipline that prevents most of what goes wrong here

Every serious failure in the reference corpus was **a lookup that failed and returned a reassuring
answer**: a contract key not found, so the gate passed on a default; an ancestry test that could not hold,
so it reported clean; a `git add` that matched nothing, so the cell was tagged empty; a branch name that
did not resolve, so the cell was "skipped" in one log line. None raised an error. All produced a
*comfortable* wrong state, which costs far more than a crash because nobody investigates a green run.

> **After any lookup that gates a decision, assert that what you were looking for is actually there** —
> count the rows, compare the trees, check the ref resolved, print the value you read. Never let absence
> fall through to a default, and never report success from a step whose exit status you did not check.

That applies to this stage's own artefacts too: if you read a field out of `report.json`,
`reconciliation_*.json` or a CSV and it is missing, that is a stop, not a zero.

## Procedure

### 0. Preflight, and the layer counts

**The tool.** This stage drives `ttnn-advise` directly. The driver already exports `TTMLIR_ADVISOR_HOME` and
sources tt-mlir's own bootstrap, so normally you only confirm it:

```
ls $TTMLIR_ADVISOR_HOME/build/bin/ttnn-advise
git -C $TTMLIR_ADVISOR_HOME rev-parse --short HEAD      # expect the pin: 618cd4e75d
```

If it is missing, `$shard-advise` SETUP.md part A has the build. **Do not rebuild while a measured stage is
running:** `tt-mlir/third_party/tt-metal/src/tt-metal` symlinks the live tt-metal checkout, and every
`cmake --build` overwrites `$TT_METAL_HOME/ttnn/ttnn/_ttnn.so`, breaking `import ttnn` for anything running.

**Exclusivity, and it cannot be established afterwards.** Hosts are shared, and `tt-smi` reports board
presence rather than utilisation, so there is no retrospective evidence that a measurement had the device
to itself. Sample the device-using processes at the **start and end of every measurement** and record both;
two reference cells predate that instrumentation and can never be shown to be clean.
**And do not run a capture while a measurement is in flight** — the container that runs `ttnn-advise` maps
the same device, so a capture during a timed stage corrupts it. Sequence them.

**The layer counts, and they are load-bearing.** The full-model estimate multiplies per-layer microseconds by
`layers_of_kind`, so a wrong count scales the headline directly. Derive them from the model's own config, not
by assumption:

- `total_layers` = `num_hidden_layers`.
- The kinds come from the config's explicit `layer_types` list where one exists. Otherwise from the interval
  field — gemma-3/4's `sliding_window_pattern = 6` over 48 layers means every 6th layer is full attention, so
  **8 full + 40 sliding**; a qwen3-next with a full-attention interval of 4 over 64 layers is **16 full + 48
  linear**. Confirm against the layer list the decoder actually builds; that code is the authority.
- A MoE model's expert layers are a kind too, even though the tracer cannot reach them (see the last section).

Record them in `incumbent.json` as `layer_counts: {<kind>: <count>}`. They **must sum to `num_hidden_layers`** —
every layer belongs to exactly one kind. `reconcile.py` aborts if `--layers-of-kind` disagrees with this or if
the counts do not sum to `--total-layers`, and the gate fails a cell that records no counts at all: an
unrecorded count means an unchecked headline.

### 1. Freeze the incumbent, before running the advisor

**The harness decides whether this stage can measure anything at all, so build it before you build anything
else.** In the reference corpus the single cell with a tight harness is the single cell that found a material
win, and two cells were unmeasurable before the advisor was even consulted.

**Copy `scripts/harness_template.py` and fill its two model hooks. Do not write your own harness.** It fixes
the protocol below and refuses to run under it, writes every field the gate reads, and emits the
`PERF_DECODE` / `PERF_DECODE_END` signposts that bound the profile to one replay. One process measures one
configuration, so a candidate cannot inherit the incumbent's warm-up:

```
CHALLENGER_MODEL_DIR=<md> CHALLENGER_DECODE_BATCH=<b> python harness_template.py \
    --label incumbent --policy <the artifact the shipped policy executed from> \
    --out models/autoports/<md>/doc/advisor_challenger/incumbent.json
```

`--policy` must name a JSON carrying `shipped_policy` and `shipped_weight_dtypes` **as they executed** — read
off the final `tt-perf-report` CSV or the datatype sweep's selected candidate, never
`resolved_policy.constructor_defaults`. Write that file first; the harness refuses to time anything without it.

What the template enforces, and why:

1. **≥10 untimed warm-up replays**, recorded as `warmup_replays`. Not 1. A device still settling produces
   repeats that fall monotonically, and in four of nine corpus cells the first timed repeat alone was
   **45–73 % of the reported noise floor** — a ramp misread as variance. On one cell, discarding it takes the
   floor from 6.282 µs to 1.693 µs, which flips the whole stage from unmeasurable to measurable.
2. **n = 5 timed blocks, each the mean of `ITERS ≥ 50` replays.** Averaging inside the block tightens
   the floor by roughly √ITERS; the corpus cell that did this reached 0.03 % against 1.0–1.4 % for the cells
   that timed one replay per repeat. Record `iters_per_repeat` — floors from different protocols are not
   comparable numbers.
3. `incumbent_ms` = **median** of the block means. Not the min: a harness that reports `best_ms = min(...)`
   invites it, and all nine corpus cells recorded the min.
4. Record `noise_floor_ms` = the spread of `repeats_ms`, and **use it** (step 3a). Every corpus cell already
   recorded this number and none of them acted on it.
5. **Never time a candidate after the incumbent in the same process.** With any residual ramp the later run
   is simply warmer, which hands the candidate a free win under a non-overlap rule that assumes exchangeable
   samples.
6. **The first process of a session may not be comparable to the rest — recommended, not required.**
   ⚠ The evidence is **a single cell**: one observation of 11.838 µs against 0.196 µs. That is a 60×
   difference and worth avoiding, but n=1 does not support a mandate, and the first v3 cell's three controls
   (ordinals 3, 4, 5; floors 0.284 / 0.926 / 0.490 µs) show no ordinal trend either way. So record
   `process_ordinal` always — it is free and it makes the question answerable — and prefer to discard a warm
   process, without treating its absence as a defect. Measured: the first
   harness process of a session recorded a floor of **11.838 µs** where the identical configuration in a
   later process recorded **0.196 µs** — a **60×** difference from JIT-cache warmth *between* processes, which
   no amount of per-process warm-up can remove. Since the floor decides `feasibility.verdict`, a cell whose
   control happened to be the first thing it ran silently changed what it was allowed to screen. So: run the
   harness once with `--label warmup_discard` and delete the output, before the incumbent. The template
   records `process_ordinal` and the gate warns if the incumbent's is 1.

`incumbent.json` — the template writes all of these except `layer_counts` (step 0): `decode_batch`,
`requested_decode_batch`,
`warmup_replays`, `iters_per_repeat`, `measured_at`, `repeats_ms`, `median_ms`, `incumbent_ms` (median),
`noise_floor_ms`, `harness`, `harness_scope`, `signposts`, and `shipped_policy` / `shipped_weight_dtypes`
sourced from what **executed** — the final `tt-perf-report` CSV or the selected
candidate JSON, never `resolved_policy.constructor_defaults`.

**State `harness_scope` explicitly, and say whether the metric is measured or derived.** It is unset in all
nine corpus cells, and they do not measure the same thing: most time one decoder layer, while one reports a
per-model composite computed as `Σ layer_count × per-layer median` — 937 ms of arithmetic, not a measurement.
A derived metric's spread is the spread of medians and is far tighter than real run-to-run variance, so
non-overlap against it will declare wins that a real measurement would not support. If the metric is derived,
label it and do not compare its floor with any other cell's.

Save the op-level `tt-perf-report` CSV per layer kind, bounded to **one** replay with
`--start-signpost` / `--end-signpost`. The window it reports should be within a few percent of
`incumbent_ms` when both cover one layer.

### 2. Capture once per layer kind, at the shipped precision

Copy `scripts/capture_template.py`; construct the decoder with the **shipped policy**, not class defaults.
Pass `--pipeline-options allow-bf16-dram-sharded-matmul=true` if any traced weight is bf16. `captured_at`
must be after `measured_at`.

Record the **layer share** of any kind you cannot capture, as `reachable_by_advisor` in `final.json`: the
layer kinds captured, the untraced window share, and what fraction of model decode time that leaves unadvised.
A contribution of zero must be readable as *"nothing to find"* or *"could not look"*, and in the reference
corpus 4 of 7 zeros were the second kind with nothing in the artefacts saying so.

**Check whether the op is really terminal before recording it as such.** `uncapturable.ops` means *the tracer
cannot trace this*; `report.json`'s `unfixable_ops` means *the advisor will not place this*. Two different
questions, and copying entries from the second into the first is how four ops that already had handlers got
recorded as blockers. Validate an entry with an actual trace attempt, and validate the op exists at all with
`getattr(ttnn, name)` — one corpus cell recorded a blocker on an op that does not exist.

**Record the capture's own scope**, as `capture_scope` in `report.json`: ops attempted, any model method you
substituted before tracing, and every private env knob with its value. Fifteen cells wrote fifteen capture
scripts of 54 to 290 lines and nothing compared them; five stopped at the same terminal op in four different
places, from 30 ops captured down to 5. Without the field, cross-cell coverage numbers silently mix captures
that attempted very different amounts of the layer. **Substitutions matter most:** where the template patches
`_decode_rope` because the tracer cannot resolve `memory_config()` before layout assignment, the advice for
that region is advice for a stand-in — and a rope-side change cannot reach the capture at all. Say so.

**Re-advise after a topology change.** `ttnn-advise` costs ~18 s end to end (measured: 18.4 / 18.4 / 18.1 /
18.6 s, device open through artefact write) — less than one harness measurement — so there is no cost argument
for screening every candidate against one start-of-run capture. The advisor discards the input's memory configs
and re-places everything, so it responds to **topology**: adding, removing or reordering ops changes the advice,
while a pure memory-config or program-config change leaves it byte-identical (verified four times over). Record
which capture each candidate was screened against.

### 2a. Consider capturing more than one consecutive layer

**Switching between the python capture and the IR hardcodes every graph input and output to DRAM
interleaved.** So with a one-layer capture the advisor is *forced* to place a conversion at both layer
boundaries, whatever the shipped graph does — 12 to 16 reshards per corpus cell carry
`producer: "input", from: "dram/interleaved/1x1"` for this reason. Those edges are artifacts of the capture,
not recommendations, and `reconcile.py` leaves them `undetermined` rather than attributing them either way.

Capturing **N consecutive layers** shrinks the damage to `1/N` of the layer boundaries and, more usefully,
moves the `N-1` interior boundaries *inside* the advised graph — where the advisor can choose to keep them in
L1. That is the one place it can see the inter-layer handoff at all. Declare it with
`reconcile.py --layers-in-window N`, or the replay guard will correctly reject the repeating op sequence.

The cost is L1 capacity: `spill.ran` is true in every corpus cell at **one** layer, with up to 4 spills, so
2–3 layers will spill more and a spilled plan applies piecemeal less well.

**This is an OPTIONAL experiment, not a requirement — and that is a change from an earlier draft.** Which
made it conditional on `spill.ran`; re-derived over 21 corpus kind-runs, **`spill.ran` is true in 21 of 21**,
so the condition selects every cell and is no condition at all. And there is **no evidence anywhere that N≥2
helps** — the corpus never ran it. Requiring it would cost every cell a re-capture to answer a question one
cell can answer, so: run it on **one** cell, on the kind that spills least, and only when someone wants the
answer. Record `layers_in_window_reason` either way. It answers one question: does the advisor keep the interior layer
boundary in L1? If it does, that is a directly attributable win worth `(N-1)/N` of the handoff cost
(§`model_estimate.layer_handoff`) and justifies a larger N; if not, the DRAM hardcoding is not the binding
constraint and the idea is closed cheaply. Requiring N≥2 everywhere is *not* the rule — it would cost every
cell a re-capture to answer a question one cell can answer.

### 3. Reconcile — `scripts/reconcile.py`, once per layer kind

Always pass **`--incumbent incumbent.json`** *and* **`--ir shard_advise/<kind>/final_ir.mlir`**. The first
gives the harness noise floor, without which nothing below decides whether a measurement can mean anything.
The second is the advice itself: `report.json` is a summary that carries **no shard shape at all** and prints
a multi-range `CoreRangeSet` as its first range only, so 58 % of its advised core counts are understated and a
plan rebuilt from it cannot be implemented. Both files come out of the same capture.

It partitions the measured window so that every device op lands in exactly one bucket and the buckets sum
to 100 %: `chain`, `boundary`, `dram_resident`, `advisor_unfixable`, `agrees_with_shipped`, `untraced`. It
fails loudly rather than emitting a gap.

**Record verdicts with `--evidence`, and never edit its output.** Write a small JSON keyed by chain id, cliff
candidate or `<device>#<n>`, carrying the fields you measured, and regenerate the reconciliation with it. An
unknown identifier is fatal, so a stale evidence file cannot annotate the wrong row. v2 had no such path —
the tool wrote `verdict: pending`, the gate demanded it filled, and the skill forbade hand-editing — and the
four cells that escaped that in four different ways were graded on *which* violation they chose.

Read these out of it:

- **`advised_plan`** — the advice as the advisor wrote it, with shard shapes, from the IR. This is what
  candidate #1 is built from, and `advised_plan.unfixable_ops` is what to drop from it first.
- **`cliff_candidates`** — material ops left on ≤2 cores where the advisor wants strictly more, ranked by
  per-model µs, each with its legal ladder. **Screen these first.** Over the reference corpus this rule flags
  5 of 14 cells, contains all three double-digit wins, and no unflagged cell produced one; it was then used to
  *predict* an unscreened −12.44 %/layer win before measuring it. Ranking by it instead of by boundary value
  moves the winning candidate to rank 1 in all four cells whose win was of this class, from 2nd/2nd/2nd and
  4th-of-27. It costs no device time.
- **the ranked chain list** — ordered by `advisor_removes_us` (the µs of shipped conversions the advice does
  *not* place on that edge), then by total conversion value; *never* by the advised ops' window share. A
  chain whose ops total under 1 % of the window can be worth several per cent once its DRAM round trips are
  gone — in this corpus a chain at 0.948 % op share won 5–6 %.
- **`advised_boundaries`** — how many shipped conversions the advisor drops, agrees with, and leaves
  undetermined, with the µs. `us_advisor_drops` is the stage's upper bound on conversion-removal value;
  screening cannot beat it, so if it is small, say so early rather than after nine measurements.
- **`material_ops_on_le_2_cores`** — a material op left on ≤2 cores *where the advisor disagrees*. That is
  advisor-attributable, and needs a measured attempt or a quoted hard error; nothing else discharges it. Its
  sibling `starved_ops_not_attributable` holds the ones the advisor places the same way: real waste, but
  improving them is a direct grid sweep with no advisor contribution. Report those, hand them to `$optimize`,
  and do not screen them here.
- **`agrees_with_shipped`** — where the advisor independently re-derived your config. Since it is blind to
  your layouts, agreement is genuine re-derivation, and on one decoder it was 82 % of the window. Its
  marginal contribution here is zero by construction; **report the number as a headline**, and do not read it
  as wasted advice. How much a from-scratch optimization it would have saved is a different experiment.

**What this script is authoritative about, and what it is only suggesting.** It prints the split, and its
output carries a `limitations[]` list. Take it seriously:

| hard — trust it over your own reading | soft — your own reading of the IR beats it |
|---|---|
| the measured window, per-op µs and shares | which advised op a given device op is |
| `accounting_closes_100pct` | `advised_here` and `advisor_removes_us` |
| the single-replay and window-ratio guards | the chain ranking, computed from the soft column |
| `by_conversion_class` µs | whether an `unresolved` class is a real boundary |

Pairing is by normalised name, falling back to **position** when nothing matches. Positional pairs are
guesses; each row carries `pair_confidence` and the header prints what share of the window rests on them
(1–5 % across the reference corpus). If it prints **DEGRADED**, stop: a large unexplained `untraced` share or
a device-to-advised fan-out over 2.5× means the profile and the advice probably describe different graphs.
Generic op names match across any transformer, so a wrong pairing does *not* announce itself as a name
failure — this check is the one that catches it.

Nothing in it is a measurement. If your reading of the IR contradicts its pairing, **write that in the README
with the IR evidence** — do not edit the JSON.

Advised ops pair with device classes automatically, so an abort means an input problem — an unbounded perf
report, a wrong file, a window that cannot be the decode path. Fix the input, not the output: hand-authoring
this file is the one failure the gate cannot detect for you.

### 3a. Obey the feasibility verdict before spending device time

`feasibility.verdict` compares what the advisor proposes removing against the spread of the incumbent's own
repeats. Non-overlap cannot resolve an effect smaller than that spread, however good the advice is.

| verdict | what it means | what to do |
|---|---|---|
| `measurable` | some chains clear the floor | screen those individually, group the rest |
| `aggregate_only` | the total clears the floor, no single chain does | **apply the top chains together as one candidate first.** Screening them one at a time returns zero regardless of the advice — record `combined_with` |
| `regrid_only` | the **boundary** ceiling is under the floor, but an op is on the cliff | **screen `cliff_candidates`.** This is not a zero — see below |
| `not_measurable` | the ceiling is under the floor **and** no op is on the cliff | **do not screen.** Report a contribution of zero *with this arithmetic as the reason*, marking chains `not_measurable` |
| `unknown` | no `repeats_ms` | fix the incumbent first |

**The ceiling has two channels and only prices one of them.** It counts boundary conversions the advice does
not place, so re-gridding an op that stays inside its own L1 chain removes no boundary and prices at exactly
`0.000 µs` — while measuring up to 236.8 µs/layer on hardware. In the reference corpus **two of the three
biggest wins came from cells whose ceiling said zero and whose every chain read `below_threshold`**, and a
third cell trusted such a ceiling and published a zero over a −12.44 % win it had already written and shipped
disabled. So a zero ceiling is a statement about one channel, never a stopping condition: if `cliff_ops` is
non-zero you owe at least one screening before publishing anything.

⚠ **`below_threshold` is a dismissal, not a measurement.** 70 of the 134 chains dismissed that way in the
reference corpus were ≥5× their own cell's noise floor. And note what a chain's µs is: its ops' *incumbent
cost*, not the saving on offer — "57× the floor" means "a 1.7 % saving there would be measurable", not that a
57× win exists.

⚠ **Do not try to rescue an overlap with more replays per block.** Measured: 250 → 1,800 replays per
measurement made the noise floor **3–4× worse** (0.4–0.7 → 1.3–3.0 µs) and still did not separate the
candidate. The `sqrt(ITERS)` argument assumes i.i.d. noise within a run; a longer window picks up slow drift,
and drift does not average down. ~50 replays per block is near the sweet spot and 200 is past it. The term
worth attacking is the **cross-process** one (step 1): re-measure in more independent processes at the same
block size, and otherwise record `not_measurable` with its arithmetic rather than spending device time on it.

### 4. Screen, in the order `screening_order` gives

The reconciliation emits it: the advised plan whole, then `cliff_candidates`, then `chains`, then the
ablations. One variable per measurement, against the frozen incumbent, and everything measured records
`repeats_ms`. A chain ends with a verdict or `below_threshold` with its conversion value. Screen
`dram_resident` rows too — *"leave this in DRAM"* is advice, and de-sharding an op has won here.

**Do not screen `advisor_unfixable` rows.** The advisor has already declared those ops unplaceable and given
you the exact runtime error it got from tt-metal's own constraint machinery. In the reference corpus 41 of 54
such declarations were screened anyway and cells recorded the identical error string back. If you believe a
declaration is wrong, disprove it with an **isolated single-op test in the exact advised config** — not with
a whole-decoder measurement.

**DS-matmul advice: screen the buffer type, not the grid.** A `ds_family` agreement carries no grid
information by design, and widening a DRAM-sharded matmul is a measured dead end — turning DS off on one
decoder's projections, exactly the advisor's 12→77-core direction, was **+65.2 % slower**, and matmul
widening won 1 of 7 measured attempts. DS matmuls are DRAM-bandwidth-bound, so core count is not the limiting
resource. What *is* screenable is a change of buffer type or program config; one cell kept a `linear`
12→55-core change worth 129 µs, so this is a default, not a prohibition.

Every kept candidate: its own op-level CSV as `perf_report`, and its own correctness result.

#### The correctness rule

A resharding is arithmetically a no-op, so it is tempting to treat correctness as the previous stage's
problem. It is not: **some ops compute the wrong answer under particular shard specs**, and changing
placement is exactly what triggers such a bug. But the rule has to tell that failure apart from one thing it
resembles and is not:

| | signature | correct response |
|---|---|---|
| a kernel bug under a shard spec | PCC drops **materially**, in one direction | reject however fast it is, and file a tt-metal bug — that is a result |
| floating-point **reassociation** | PCC moves in the last few decimals | benign, and **guaranteed** whenever a reduction's core count changes |

v2 said "if a placement-only candidate moves PCC at all, reject it", which cannot make that distinction and
made the highest-yield transformation in this stage permanently unshippable. **So:**

> Build an **absolute** oracle: the candidate against a reference the change cannot move — the HuggingFace
> layer, or the model's own functional decoder — at **the model's own PCC bar**, read out of the model's own
> test. Do not invent a bar. Measure the **incumbent** against the same reference too. **Ship if the
> candidate is within the bar and no worse than the incumbent.**
>
> A **differential** oracle against the frozen incumbent is a useful observation and must be recorded, but
> it is **not a veto**. A differential delta that is large is the kernel-bug signal — and "large" means a
> departure the absolute comparison also sees.

Record `oracle_kind` (`absolute` | `differential`), `oracle_pcc_bar`, `oracle_bar_source` (the file:line the
bar came from) and `incumbent_pcc_vs_reference`. The gate fails a bar tighter than the model's own test bar
without a recorded justification.

**Why this is a correctness change and not just a permissive one.** In the reference corpus one cell's
shipped norm change moved a differential PCC by 0.0177 and shipped; another's moved it by 0.0000089 and was
rejected — the rule punished the *less* perturbing change. And a differential oracle cannot tell which side
moved: built absolutely for one discarded candidate, the candidate scored **0.99931** against the model's own
bfloat16 reference and the **shipped incumbent 0.98347**, failing the model's 0.995 bar. The old rule can
ship the less accurate configuration and reject the better one. It happened twice.

Record **`oracle_weights`: `real` or `synthetic`**, and prefer real for anything you ship. Two traps the
corpus fell into:

- **A synthetic-weight PCC does not bound the real-weight PCC.** Random weights have none of the outliers real
  ones do, and these policies quantise the MLP to `BFLOAT4_B` — so the oracle is weakest exactly where the
  precision risk is highest. One cell ran on synthetic weights only because real ones were not cached on the
  host and HF egress was disabled; that is a plumbing check, and should be labelled as one.
- **An oracle that compares the implementation against itself cannot fail.** "eager vs traced replay" and
  "preservation relative to the frozen incumbent" both pass automatically for any placement change that keeps
  tracing working — one cell reported PCC exactly 1.0 this way, and another shipped with
  `absolute_pcc_current_environment_passed: False`. Compare against a reference the change cannot move.

Note the second bullet is about the **veto**, not about recording: a differential comparison is still worth
having, it just cannot be the thing that decides. And a per-layer PCC oracle is not a substitute for the
model's own regression suite — in one cell a narrow oracle passed a candidate that violated the per-head norm
contract, and only the broader optimized-decoder run caught it.

### 5. Decide by non-overlap

Ship a change iff **every candidate repeat beats every incumbent repeat**. No noise floor, no min baseline.
This is only comparable across cells if n is fixed — the false-positive rate is `1/C(2n,n)`, so 5 % at
n=3 against 0.40 % at n=5. **Confirm the winner in a fresh process** before shipping: cross-process
variance is otherwise unmeasured, and per-process work happens once per process.

**Prove the candidate is control-plus-one-knob.** Record the diff of the executed policy against the frozen
policy and check it has **exactly one changed field**. Build the candidate policy with
`dataclasses.replace` on the frozen one, never from a fresh constructor: the one cell that checked found its
candidate policies had inherited constructor defaults and silently changed several dormant fields, and it had
to remeasure six candidates and two confirmations. Fourteen cells never checked.

**Record what the knob actually moves**, as `op_under_test: {name, incumbent_grid, candidate_grid,
legal_ladder}` per screened candidate. Two arms of one reference model use the **same env knob name with
different defaults** (88 in one, 8 in the other), so "the 88-core candidate" is a 1→88 change in one arm and an
8→88 change in the other — and the two deltas look comparable while measuring different things. This field is
what makes cross-arm comparison mean anything, and it makes the cliff check mechanical.

**And record what the candidate assumes about shape**, as `candidate_shape_assumptions` per shipped knob: tile
rows, divisibility, grid shape. The largest wins in the reference corpus are **batch-pinned by construction** —
one norm memory config hardcodes a one-tile-row shard height, so at batch 64 it fails with
`Shard height 32 must match physical height 64` and at batch 8 it fails at build — and nothing in `final.json`
said so. A reader saw "−13.4 %/layer" with no hint that it evaporates off batch 32.

⚠ **Do not "simplify" the protocol to eager timing.** Traced decode replay is not just what production does:
for the highest-yield candidate class host and device costs move in **opposite** directions. A sharded norm
adds ~46 µs of host dispatch and saves ~65 µs of device time, so it reads as a **+45.6 µs regression** eagerly
and a **−65.2 µs win** under traced replay. Measured on two independent models — **an eager harness would have
rejected every norm win in the reference corpus.** It is also why `*_profile` runs come out 1–3 % slower than
their timed counterparts, and why mixing measurement modes produces contradictory rankings.

### 6. Combine across layer kinds — and decide on the full-model estimate

Everything up to here is **per layer**, because that is what the profile and the harness cover. Detection has
to stay per-layer: an effect is compared against the per-layer noise floor. But **choose between candidates on
the full-model estimate**, which `reconcile.py` gives as `model_estimate.this_kind_us` per kind:

```
full model  =  SUM over layer kinds of ( per_layer_window_us x layers_of_kind )     <- counts from step 0
chain value =  advisor_removes_per_model_us  =  advisor_removes_us x layers_of_kind
uncertainty =  noise_floor_us x layers_of_kind          <- scales with the estimate, quote it
```

**Report the stage's result as a full-model estimate before and after, with that band.** Scaling a per-layer
delta to the model scales its error by the same factor: on one corpus cell a 6.282 µs per-layer floor becomes
a **± 201 µs** model-level band over 32 layers, so a claimed model saving smaller than that is inside its own
error bar. A model number quoted without the band reads far more precise than the measurement supports.

Per-layer ranking picks the wrong candidate across kinds. In one corpus model a sliding-attention chain worth
1.629 µs/layer is **65.2 µs/model** over 40 layers, while a full-attention chain worth *more* per layer
(2.146 µs) is only **17.2 µs/model** over 8. Rank within a kind by the per-layer number; rank *across* kinds,
and pick the winner, by the per-model number.

Where the reported latency is a weighted composite, the candidate space is the **product** of per-kind
winners, not the union of one-kind-varied sets. The composite is arithmetic over independently measured
series, so these cost no extra device time. Record every measured set with its `chains` (or `set`),
`measured_ms`, `repeats_ms` and `oracle_passed`; ship the set that minimises the full-model estimate.

**And products are required *within* a kind too, not only across kinds.** Once two or more candidates in the
same kind each pass non-overlap and touch disjoint ops, **their product is a required candidate** — additivity
is not predictable, the reference corpus has one super-additive and one sub-additive case, and in both the
product beat every isolate (−13.24 % against a best isolate of −7.60 % on one cell; −2.82 % against −1.86 % on
another). Only 2 of 15 cells built one, and both gained. This is the same measurement the apply-all-first order
produces from the other direction; where both exist they should agree.

**One thing the model estimate can expose that is not yours to fix.** `model_estimate.layer_handoff` reports
whether the layer takes its input from DRAM while leaving its output in L1. If it does, consecutive layers do
not hand off in L1 and that conversion is paid once per layer — 33.6 µs and 48.0 µs across the model in two
corpus cells, while a third opens and closes on `L1_WIDTH_SHARDED` and pays nothing, so it is a decoder
implementation choice rather than a framework limit. The advisor is never asked about a layer boundary, so
this is out of scope here: **report it upstream, do not screen it, and never book it as advisor contribution.**

### 7. Ship

Write the winner into `tt/optimized_decoder.py` and **keep every losing knob, default-off**, so rejected
candidates stay re-measurable.

**Keep the profile you already ran.** Save the op-level report for the frozen incumbent *and* for the shipped
winner, with the full invocation — `--group-by category --summary-file … --stacked-csv …`, not just the plain
CSV. It costs nothing extra: `tt-perf-report` already computes `Total %`, `Bound`, `Cores`, `DRAM %`, `FLOPs %`,
an op-category split and per-op advice, and v2 read three columns of it. Only 1 of 15 cells kept a before/after
pair, which is why op-level verification is impossible for the other 14 — and why the largest coverage win in
that corpus still has no number attached. Report the category split (`Compute` / `TM` / `DM` / `Other`) for
both, and split the non-compute share into **layout-induced** (tilize/untilize/typecast/fill-pad/
interleaved↔sharded/reshard/copy) and **graph-structural** (permute/transpose/reshape/slice/concat/head-ops):
only the first is reachable by a layout advisor, and the second belongs to `$optimize`. A high non-compute
share is a hypothesis worth recording, not a threshold to fail on — and treat the tool's own per-op advice the
same way: it says "increase grid size" for DS matmuls where widening measured +65 % slower.

`final.json`: `outcome`, `changed`, `final_ms`, `incumbent_ms`, `repeats_ms`, the winning config, the oracle
fields from step 4, `iterations[]`, and the stage's **headline metric**, which nothing else computes for you:

```json
"model_estimate": {
  "before_us": 62429.0,          // SUM over kinds of reconciliation model_estimate.this_kind_us
  "after_us":  61265.2,          // the same sum with each kind's measured winner applied
  "band_us":   1462.3,           // SUM over kinds of uncertainty_per_model_us -- linear, not quadrature
  "per_kind": {"sliding_attention": {"before_us": 51585.8, "after_us": 50742.1, "layers": 40},
               "full_attention":    {"before_us": 10843.2, "after_us": 10523.1, "layers": 8}},
  "method": "per-layer measured delta x layers_of_kind, summed"
}
```

Rules for it: **compute `after_us` per kind and then sum** — never scale one cell-wide ratio across kinds,
because the winners differ by kind and that is the whole reason the per-model number exists (step 6). Sum the
band **linearly**, not in quadrature: these are repeats of one harness on one device, not independent trials, so
the conservative sum is the honest one. If `after_us - before_us` is smaller than `band_us`, say so in the same
breath — that is a result inside its own error bar. If nothing won, `after_us == before_us` and the stage ships
the incumbent unchanged.

Also required in `final.json`: **`advised_plan_verbatim`** — the apply-all candidate of step 4, with its own
verdict and `measured_ms`, or `hard_error` naming the item that would not run and the single-op test that
isolated it. "Not tried" and "tried and lost" must not look the same in the artefacts; in the reference corpus
four cells never applied the advice and recorded no reason, and their output is indistinguishable from a
measured rejection. And **`reachable_by_advisor`** (step 2), so a zero says which kind of zero it is.

**A blocked outcome is a passing outcome, and it must be cheap to report.** `final.json` takes
`could_not_do[]`: each entry names what was not measurable, why, and the evidence (the hard error, the
missing artefact, the knob that does not separate the effect). This exists because the alternative has a
cost: when the only way to pass is to produce a number, a cell will produce one. In the reference corpus a
cell that needed advisor evidence it could not generate wrote documentation citing a tool path that does
not exist on its own host, with plausible hashes and op counts — fabricated provenance for a command that
could not have run. Nothing detected it; a byte-identity comparison did, days later. **Reporting "I could
not do this" must be strictly easier than inventing it.** One reference cell's only knob drove the norm and
the residual chain together, so a norm-only effect was not screenable in principle — that is a result, and
it needs somewhere to go.

**Try to refute your own win before shipping it.** Every shipped change needs one *disconfirming*
measurement, recorded as `disconfirmation`:

- **order swap** — re-measure the incumbent in a fresh process *after* the candidate. A candidate that only
  wins in the later process won a warm-up, not a placement. This is the failure the per-process rule and
  `process_ordinal` exist to prevent, and it is worth one measurement to show it did.
- **knob off** — re-measure with the shipped knob disabled and confirm you get the incumbent back. That
  catches the case where the knob was never what changed.

Neither is expensive, and the analysis that produced this stage refuted about **1 in 6 of its own
recommendations** on follow-up. Assume yours has the same rate.

`README.md`: the accounting from step 3, what was screened with its number, what was not and why, and the
unreachable share. **Say what you could not do**, in a field of its own: one reference cell's only knob drove
the norm *and* the residual chain together, so a norm-only effect was not screenable in principle — a different
result from "screened and found nothing", and nothing distinguished them.

Artefact hygiene: **gzip the decision trace** and keep it — it is the only artefact that answers "why this
grid", and one cell committed it uncompressed at 51 MB while another gzipped 118 MB to 7 MB. `.gitignore` the
raw profiler dumps; two cells had to delete them by hand after a 1.2 GB push was rejected.

## Iterating

You may apply what won and go again — but **re-profile first** and re-run `reconcile.py` on the new CSV;
the old ranking describes a graph that no longer exists. Record `reranked_from`. Re-capture only after a
topology rewrite that changes an op's shape; a dtype change is not one. Cap at 3 captures per layer kind.

Extension is not free: growing L1 residency hits capacity walls, and a bad placement in this project wedged
PCIe and needed a `tt-smi` reset. Do not run extension experiments alongside a measured stage.

## Scope: what this stage is not

02b measures one thing — what the advisor adds to an already-optimized decoder. It is **not** a second
`$optimize` pass, and the fastest way to ruin its result is to do optimize's work inside it and book the
gain here. Three categories to report and hand over rather than screen, all of them separated for you by
`reconcile.py`:

| surfaced as | why it is out of scope |
|---|---|
| `advised_boundaries.us_advisor_agrees` | the advisor places the same conversion, so removing it is your idea |
| `starved_ops_not_attributable` | a starved op the advisor also starves — a direct grid sweep |
| `model_estimate.layer_handoff` | the advisor is never asked about a layer boundary |

Precision is likewise not this stage's axis: the datatype sweep chose it, and `compute_config` /
`math_fidelity` in the advised IR only mirror what was traced. Changing them turns a placement measurement
into a precision one and drags in a real-weight correctness bar (step 4) for no advisor attribution.

Sweeping expert and norm grids **directly**, without the advisor, is a separate and often better-paying
activity — it belongs to `$optimize`, and one model's norms went 85 µs → 9 µs that way with no advisor
involved.

## What this stage cannot reach

**Less than v2 thought — check before you record a blocker.** Coverage, not placement, was the largest single
limiter in the reference corpus: on **nine cell/kinds the advisor was never shown the layer**, discarding 58 to
77 % of the window before it got a look, and one of those cells published a flat zero that two tracer handlers
later turned into 11 candidates worth 632 µs/model. All five real gaps are now closed in the tracer pin
this stage runs. Verified handler by handler in the pinned checkout, and the split matters because only the
first list belongs to the tracer this stage actually uses: **direct-TTNN (`ttnn_emit_tracer`, the default)** —
`sparse_matmul`, mutable-state `ttnn.copy` (recorded in `cache_alias`, not `weight_cache`, or it orphans the
destination's placeholder), `paged_fused_update_cache`, `ones_like`, `pow`/`pow_tensor`, `softplus` and
`repeat_interleave`; **interception tracer only** — `TracedTensor.__getitem__`.

⚠ **The tracer that runs is not necessarily the one in the checkout.** `ttnn_jit` is installed into the
toolchain venv as a plain directory rather than as an editable install, so `ttnn-advise` imports
site-packages: a `git checkout` of another tt-mlir branch changes the recorded `advisor_commit` while the
module that traces stays exactly as it was. There is also a stale `build/lib.../ttnn_jit` copy inside the
checkout. `capture_template.py` therefore fingerprints the **imported** tracer and compares it against the
checkout's, and the gate fails a mismatch — the tracer is what decides whether a layer is visible at all,
which is the difference between a real zero and a coverage zero. Of the ten ops cells recorded as blockers, **four already
had handlers and one did not exist at all**. So: attempt the trace, read the traceback, and only then record a
kind as unreachable — with its layer share.

What genuinely remains: `rotary_embedding_hf` cannot be traced by the `interception` tracer at all (TTIR has
only `rotary_embedding_llama`, which needs a `trans_mat` operand the HF op has no equivalent for), and
`rearrange` needs its einops pattern parsed. Neither blocks a decoder this stage has met.

**Conversion time this stage must leave alone.** DRAM round trips are the thing worth minimising, and
`reconcile.py` splits them by who wants them. Only `us_advisor_drops` is yours to screen. Two other pots are
real cost and out of scope:

- **`us_advisor_agrees`** — the advisor places a conversion here too. Removing it may well win, but the win
  would be your idea, and booking it against the advisor contaminates the measurement.
And one pot that is **unresolved, not out of scope**:

- **`us_unresolved`** — `ReshapeView`, `FillPad`, `Copy`. The advisor does not state these as placement
  edges, but a `ReshapeView` can carry a hidden layout change and act as a real chain boundary despite
  looking shape-only. Not small: six of them ran 8–19 µs each on 110 and 16 cores, 88.5 µs of a 1355 µs
  window. The profile cannot decide it — `Input 0 Memory` gives the memory class with no grid and no output
  column, so even a `Reshard` that regrids 30→40 cores reads as unchanged. **Check the edge in the IR.** If
  it does change layout it is a boundary like any other; if it does not, it is cost for `$optimize`.

Report `us_advisor_agrees` with its µs and hand it to `$optimize` — eliminating conversions the advisor does
not propose is a better-paying activity than this stage, just a different experiment. Resolve `us_unresolved`
rather than skipping it: the stage's bias rule says a suppressed candidate understates the contribution.

**And rank the profile's own conversion ops independently of the advisor.** The worklist above is derived from
the advice, so it inherits the advice's blind spots. The single largest number in the reference corpus was
found this way and no advisor-derived list could have surfaced it: one decoder pays **3,983 µs/layer — 25 % of
its layer — in tile↔row-major conversions**, 191 ms across the model, on ops *already spread over 109 of 110
cores* and running at about **1 % of the memory roofline**. The advisor's ceiling for it is 0.000 µs and that
is **correct** — the advice places those conversions too, because they are legally required — so the stage
filed a quarter of a 27B model's decode time under `boundary`: reported, out of scope, uncredited. Both
behaved correctly.

So: list every conversion op in the profile by cost with its effective bandwidth, and report the top ones
whatever the advisor thinks. A conversion moving 80 KB in 819 µs is a finding. **It is not yours to fix** — the
cause there was a graph-shape choice, a 4-element convolution window sitting on the 32-wide tile axis, so the
tiled form was 8× padded — but naming it is free and it goes to whoever owns the decoder.
