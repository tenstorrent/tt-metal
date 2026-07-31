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

Record the **layer share** of any kind you cannot capture. `sparse_matmul` and the SSM / gated-delta ops
are terminal in the tracer; if such a kind dominates, *"the advisor cannot reach N of M layers"* is the
correct result and must be stated, not discovered later.

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
2–3 layers will spill more and a spilled plan applies piecemeal less well. Start at N=2 on a kind that
currently spills 0–1 times, and check the one question that settles it: does the advisor keep the interior
layer boundary in L1? If it does, that is a directly attributable win worth `(N-1)/N` of the handoff cost
(§`model_estimate.layer_handoff`), and it justifies a larger N.

### 3. Reconcile — `scripts/reconcile.py`, once per layer kind

Always pass **`--incumbent incumbent.json`**. Its `repeats_ms` give the harness noise floor, and without that
number nothing below decides whether a measurement can mean anything.

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
| `not_measurable` | even the total is below the floor | **do not screen.** Tighten the harness (more replays per timed block) or report a contribution of zero *with this arithmetic as the reason*, marking chains `not_measurable` |
| `unknown` | no `repeats_ms` | fix the incumbent first |

This is the difference between a real zero and an unmeasurable one, and the gate fails a cell that keeps a
chain while sitting below its own floor. In the reference corpus one cell had a ceiling of 0.65× its floor and
shipped a win anyway; another had 4.31× with no single chain above 1×, screened one at a time, and reported no
change.

### 4. Screen, in the order the reconciliation gives

Each chain as one unit, one variable per measurement, against the frozen incumbent. Every chain ends with
`repeats_ms` and a verdict, or `below_threshold` with its conversion value. Screen `dram_resident` rows
too — *"leave this in DRAM"* is advice, and de-sharding an op has won here.

Every kept candidate: its own op-level CSV as `perf_report`, **and its own correctness result** — the
incumbent's oracle at the incumbent's own PCC bar, recorded per chain as `oracle_passed` / `oracle_pcc`, not
inherited from the final winner. **A faster decoder that fails its oracle is a regression with a good number.**

**Why the oracle is load-bearing in a placement stage.** A resharding is arithmetically a no-op, so it is
tempting to treat correctness as the previous stage's problem. It is not: **some ops compute the wrong answer
under particular shard specs**, and changing placement is exactly what triggers such a bug. The failure is
often partial — wrong on edge tiles or on some cores — so it shows up as a PCC that drops materially rather
than as a crash. If a placement-only candidate moves PCC at all, do not tune the threshold: reject the
candidate however fast it is, and report the op and the shard spec as a tt-metal bug. That is a real result
of this stage, not a nuisance.

A **differential** oracle catches this well and cheaply: same weights on both sides, candidate against the
frozen incumbent. Synthetic weights are adequate — a kernel that computes the wrong answer does so whatever
the weight distribution — which is why this stage does not need real weights for placement work (step 4's
`oracle_weights`).

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

In this corpus the oracle was exercised against an actual change in only three of nine cells; in the six
no-change cells it was trivially satisfied. Treat it as unproven and be explicit about what yours covers.

Screen DS-matmul advice **last**. It has not won a measurement in this corpus, and where it agrees with a
shipped DS config there is nothing to screen.

### 5. Decide by non-overlap

Ship a change iff **every candidate repeat beats every incumbent repeat**. No noise floor, no min baseline.
This is only comparable across cells if n is fixed — the false-positive rate is `1/C(2n,n)`, so 5 % at
n=3 against 0.40 % at n=5. **Confirm the winner in a fresh process** before shipping: cross-process
variance is otherwise unmeasured, and per-process work happens once per process.

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

**One thing the model estimate can expose that is not yours to fix.** `model_estimate.layer_handoff` reports
whether the layer takes its input from DRAM while leaving its output in L1. If it does, consecutive layers do
not hand off in L1 and that conversion is paid once per layer — 33.6 µs and 48.0 µs across the model in two
corpus cells, while a third opens and closes on `L1_WIDTH_SHARDED` and pays nothing, so it is a decoder
implementation choice rather than a framework limit. The advisor is never asked about a layer boundary, so
this is out of scope here: **report it upstream, do not screen it, and never book it as advisor contribution.**

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

`ttnn.sparse_matmul` and the SSM / gated-delta ops are terminal in the tracer, so a model whose time is in
its experts or its linear attention is largely invisible here — state the share rather than implying
coverage.

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
rather than skipping it: the stage's bias rule says a suppressed candidate understates the contribution. Sweeping expert and norm grids **directly**, without the advisor, is a separate and often
better-paying activity.
