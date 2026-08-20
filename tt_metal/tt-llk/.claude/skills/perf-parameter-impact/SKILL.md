---
name: perf-parameter-impact
description: Analyze LLK performance-report CSV files and create evidence-backed parameter-impact reports in a Cursor Canvas. Use when asked how formats, fidelity, dimensions, accumulation, indexing, loop factors, or other parameters affect TILE_LOOP or another marker in a .post.csv report.
user_invocable: true
---

# Perf Parameter Impact

## Goal

Turn an LLK `.post.csv` into a reproducible report that distinguishes absolute
cost, paired effects, interactions, bottlenecks, confounding, null results, and
practical recommendations.

## Inputs and defaults

Obtain the CSV path, marker (default `TILE_LOOP`), parameters of interest, and
preferred baselines. If parameters are omitted, inspect every non-metric column
that varies. For MX comparisons, use `Float16_b` as the default baseline unless
the user requests another one; never silently pool `Float16` and `Float16_b`.

## Workflow

### 1. Establish test semantics

Read the CSV header and representative rows. Inspect the current Python test
generator and, when needed, the C++ kernel to determine:

- how formats and variants are generated;
- which parameters are linked by construction;
- whether register formats are inferred or explicit;
- what dimensions, tile counts, faces, and `loop_factor` mean;
- whether unpack-to-dest bypasses a stage;
- whether the CSV reflects the current test sweep.

Flag stale-report risk when current test axes are absent from the CSV. Never
claim an independent effect for parameters the test does not vary independently.

### 2. Filter and audit

Filter exactly on `marker == <requested marker>`. When useful, save the result
beside the source as `<report-stem>_<MARKER>.csv`; preserve the source.

Report source and filtered row counts, duplicate configurations, and missing
metrics. Classify columns as:

- configuration;
- marker or run metadata;
- cycle metrics such as `mean(L1_TO_L1)` and stage isolates;
- congestion metrics;
- code-size metrics such as `TEXT_SIZE(...)`.

For each requested configuration column, report distinct values, counts,
missingness, whether it is constant, and whether it is tied to another axis.
Do not chart constants; state that their effect is not estimable.

### 3. Build meaningful labels

Use composite labels when one physical path spans several columns: input A/B
plus register A/B, output plus destination mode, `MxFp4 (2x register A/B)`, or
`MxFp4 (2x register A/B + direct indexing)`. Keep full category labels visible
without requiring hover.

### 4. Summarize absolute cost

Positive cycle data is often skewed. Use geometric means for pooled cycle
summaries unless another statistic is justified, and label the statistic
exactly. Include sample count and preferably median and interquartile range in
supporting data.

Useful groupings include input/register mode, output format, fidelity,
geometry/dimensions, and fidelity-specific views when fidelity changes the
bottleneck. Show `L1_TO_L1`, `UNPACK_ISOLATE`, `MATH_ISOLATE`, and
`PACK_ISOLATE` together when available. Lower cycles means faster.

Do not create a pooled “absolute cost by fidelity” chart that averages across
all formats. It hides format and register-path effects. Show absolute format
cost within one fidelity at a time instead.

### 5. Estimate effects with exact pairs

Do not present unbalanced group averages as parameter effects. For parameter
`P`:

1. Choose baseline and treatment values.
2. Match rows on every relevant configuration column except `P` itself and
   columns that merely restate `P`. Classify each column `P` changes before
   dropping it from the key — see below.
3. Before joining, require the pairing key to be unique within each arm. Handle
   intentional repeats as described below; never allow a many-to-many join.
4. Keep exact baseline/treatment matches.
5. For lower-is-better metrics compute
   `speedup = baseline_cycles / treatment_cycles`.
6. Report geometric-mean and median speedup, dispersion, win/tie/loss counts,
   and exact pair coverage.

Use explicit denominators such as `1,344/1,344 matched pairs`, not “all pairs.”
A value above `1.0x` is a speedup. If pairing is impossible, label the result
descriptive rather than causal.

#### Columns changed by `P`

Dropping a column from the matching key because `P` determines it is only safe
when that column carries no performance effect of its own. Separate the two
cases:

- **Restatement.** The column is a different encoding of `P` and has no
  independent effect. Drop it from the key.
- **Linked axis.** The column has its own effect and the sweep yokes it to `P`.
  Dropping it manufactures pairs across unlike configurations and charges their
  combined difference to `P`.

Quasar matmul is the standing example: `matmul_dimensions()` still halves the
destination tile budget when `dest_acc=Yes`. Perf keeps dest-full tall and
wide (`(1, max_tiles)` and `(max_tiles, 1)` × `kt={1, 4}`), so dest_acc=No
is an 8-tile grid and dest_acc=Yes is a 4-tile grid. Geometry is not free to
match across `dest_acc`. Pairing while ignoring geometry compares different
shapes and calls the difference a `dest_acc` effect.

For a linked axis, do one of the following, in order of preference:

1. Restrict to the sub-region where the linked column takes the same value in
   both arms, and report the reduced coverage explicitly.
2. If no such sub-region exists, report the **bundled** effect of `P` together
   with its linked axes, name the bundle in the label, and state that the
   independent effect of `P` is not identifiable from this sweep.
3. Request the missing sweep combinations needed to separate them.

Never present a bundled effect as an independent one. This is the same
prohibition as step 1, applied at the pairing stage.

#### Duplicate pairing keys

Audit key multiplicity separately in the baseline and treatment arms before
joining. A plain merge is valid only when each key occurs at most once per arm;
otherwise it creates an `n × m` Cartesian product, overweights duplicated
configurations, and inflates the reported pair count.

Handle intentional repeats in one of these ways:

1. If runs are deliberately paired, add a stable run or repeat identifier to
   the key and verify a one-to-one join.
2. Otherwise, aggregate repeats once per arm and configuration key before
   calculating the treatment effect. State the per-key aggregation and retain
   repeat counts as supporting data.

If neither treatment is justified, do not calculate a paired effect. Report
duplicate counts and request enough run metadata to match repeats correctly.
Pair coverage denominators count unique matched keys after this handling, not
rows produced by a join.

### 6. Analyze interactions and confounding

Check interactions that can change conclusions:

- format × fidelity;
- output format × `k_dimm`;
- input path × fidelity;
- destination accumulation × geometry or tile count;
- direct indexing × eligible register path;
- unpack-to-dest × output/pack behavior.

Use a heatmap when two axes materially change an effect. Do not split every
chart by fidelity mechanically; keep one balanced report when the split adds
no information.

Flag structural confounding such as:

- `dest_acc`, `tile_cnt`, and geometry changing together;
- 2x register mode existing only for one format;
- direct indexing existing only for the 2x path;
- output restrictions on some inputs;
- `Int32` output occurring only with `Int8` input.

### 7. Build the bottleneck model

Interpret:

- `L1_TO_L1`: observed end-to-end cost;
- `UNPACK_ISOLATE`: unpack-stage cost;
- `MATH_ISOLATE`: math-stage cost;
- `PACK_ISOLATE`: pack-stage cost.

Assign each complete row to its largest isolate and summarize bottleneck share
overall and by fidelity. Use this to explain why a large unpack or pack gain
may not appear in `L1_TO_L1` once another stage is slower. Do not assume
end-to-end cost equals the maximum isolate exactly: overlap, synchronization,
and pipeline overhead can add residual cost.

### 8. Quantify null results

Treat “no effect” as measured evidence. For congestion, code size, indexing, or
another suspected null, state the comparison, exact pair coverage, central
effect, and range or dispersion. Distinguish exact equality from negligible
change. Compare each congestion stage with its corresponding isolate. Report
text-size deltas in bytes.

### 9. Validate surprising findings

For every strong or counterintuitive claim:

1. Recompute from filtered source rows.
2. Break down by fidelity and likely confounders.
3. Verify pair counts and duplicate keys.
4. Inspect representative raw rows.
5. Check test-generator eligibility restrictions.
6. Explain stage-local versus end-to-end behavior.

Never reuse chart constants from an earlier report without recomputing them
from the current CSV.

## Canvas report

Read and follow the available `canvas` skill before creating or editing a
`.canvas.tsx` file. Embed analyzed data in one self-contained canvas.

Order the report:

1. Executive finding and dominant performance model.
2. Bottleneck share overall and by fidelity.
3. Absolute-cycle views needed to inspect real costs.
4. Paired effects with explicit baselines and coverage.
5. Interaction heatmap where meaningful.
6. Geometry/mode coupling and confounding.
7. Quantified congestion, code-size, or other null results.
8. “What to actually do” with ranked conditional recommendations.
9. Method, filtering, pairing key, and caveats.

### Default absolute-cycle charts

For matmul-style reports with format, fidelity, and destination-accumulation
axes, use this design unless the user requests another:

1. Add clickable fidelity options for values the **current sweep actually
   emits**. Quasar matmul keeps LoFi–HiFi4 for Float16 / Float16_b and
   LoFi-only for MX; do not show empty HiFi2/3/4 MX panels as if they were
   measured.
2. Under the selected fidelity, show absolute cycles by composite
   input/register mode, output format, and geometry/destination accumulation.
   Geometry labels should name dest-full **tall vs wide** and `kt=1` vs
   `kt=4`, not the old 12/9 dest-fill factorization set.
3. Before placing `dest_acc=No` and `Yes` bars side by side, audit whether both
   arms contain identical geometries and other relevant configurations.
4. If common geometry exists, restrict the split bars to that common support,
   state the reduced coverage, and use the same color for each metric pair with
   `dest_acc` distinguished by opacity or another restrained treatment.
5. If no common geometry exists, as in the current Quasar matmul sweep (8-tile
   vs 4-tile dest-full), do not split input/register or output-format
   categories into adjacent `dest_acc` bars. Either omit that split in favor of
   the geometry chart or use separate panels labeled **observed bundled
   population: dest_acc + geometry**. Explicitly state that gaps between
   panels are descriptive and are not a `dest_acc` effect.
6. In the geometry chart, encode both geometry and destination accumulation in
   the x-axis label. Keep four metric bars per composite category and do not
   split them again.
7. Add a red min–max whisker to every absolute-cycle bar. Calculate the minimum
   and maximum from exactly the rows summarized by that bar.
8. Call this an **observed min–max range**, not variance. It is sensitive to
   extreme configurations and is not a repeated-run noise estimate.
9. Hover must show category, metric, `dest_acc` when applicable, geometric
   mean, minimum, maximum, and max-minus-min range.
10. State sample counts separately for every displayed subgroup and composite
    geometry/destination category.

Use a custom themed SVG when the built-in chart cannot draw per-bar min–max
whiskers. Use host-theme colors; never hardcode red or other colors.

Chart requirements:

- title each chart with metric, grouping, and baseline where relevant;
- label both axes and units;
- show exact hover values without a help (`?`) cursor;
- keep full category names visible;
- use `x` for speedup and `cycles` for absolute cost;
- show numeric y-axis ticks and do not print values directly on absolute bars;
- retain a legend for all metrics and destination-accumulation treatments;
- use red min–max whiskers only for absolute summaries, not paired speedups;
- caption source file, marker, row count, and aggregation;
- preserve precise hover values while rounding display labels;
- avoid repetitive charts when one heatmap suffices, unless the user asks for
  both views.

## Recommendation rules

Make recommendations conditional on the bottleneck:

- optimize unpack only when unpack is on or near the critical path;
- expect higher fidelity to increase math cost and often make math limiting;
- prefer smaller output formats when pack is exposed;
- describe stage-local improvement as latent headroom if another stage limits;
- recommend direct indexing only when eligible matched pairs show useful gain;
- request missing sweep combinations needed to separate confounded effects.

End on actions, not caveats.

## Delivery checklist

- [ ] Marker filter and row counts are explicit.
- [ ] Every varying requested parameter is covered.
- [ ] Constants and non-estimable axes are identified.
- [ ] Baselines, sample counts, and pair denominators are visible.
- [ ] Relative effects use exact matches.
- [ ] Pairing keys are unique per arm; intentional repeats use a stable run ID
      or are aggregated per arm/key before a one-to-one join.
- [ ] Columns dropped from the pairing key are restatements, not linked axes;
      any unavoidable bundle is labeled as such.
- [ ] Geometric means are labeled as geometric means.
- [ ] Bottleneck claims use isolate metrics.
- [ ] Confounding and eligibility restrictions are documented.
- [ ] Null results and strong claims are quantified and validated.
- [ ] Canvas labels, units, hover values, and source captions are complete.
- [ ] Absolute charts use fidelity selectors and per-bar min–max ranges;
      destination splitting is limited to common geometry, and confounded
      populations are separated and labeled as bundles.
- [ ] Min–max is described as observed range rather than variance.
- [ ] The report ends with practical recommendations.

## Example triggers

- “Analyze this matmul `.post.csv` for `TILE_LOOP` parameter impact.”
- “Compare MX with `Float16_b` and explain why unpack gains do not appear end
  to end.”
- “Show output-format sensitivity by fidelity and `k_dimm`.”
- “Quantify whether direct indexing, congestion, and text size matter.”
