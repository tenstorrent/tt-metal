# LLK perf infrastructure: the rules

How to change a perf test without breaking the gates.

Background: [architecture.md](architecture.md).

All commands run from `tt_metal/tt-llk/tests/python_tests`. No hardware is
needed for any gate in this document.

---

## Contents

- [0. The two rules](#0-the-two-rules)
- [1. The gates, and what each one means](#1-the-gates-and-what-each-one-means)
- [2. Add a new sweep parameter](#2-add-a-new-sweep-parameter)
  - [What each skipped step costs you](#what-each-skipped-step-costs-you)
  - [Why the version matters](#why-the-version-matters)
- [3. Add a new parameter class](#3-add-a-new-parameter-class)
  - [The naming rules](#the-naming-rules)
- [4. Add a new perf test](#4-add-a-new-perf-test)
  - [Where the two derivation paths differ](#where-the-two-derivation-paths-differ)
- [5. Rename a perf test](#5-rename-a-perf-test)
  - [The rules the gate enforces](#the-rules-the-gate-enforces)
  - [What a rename costs downstream](#what-a-rename-costs-downstream)
- [6. Delete a perf test](#6-delete-a-perf-test)
- [7. Rename a column](#7-rename-a-column)
- [8. Add or rename a run type](#8-add-or-rename-a-run-type)
- [9. Add or rename an efficiency metric](#9-add-or-rename-an-efficiency-metric)
- [10. Run a perf report](#10-run-a-perf-report)
  - [Where the output goes](#where-the-output-goes)
  - [Reading a number](#reading-a-number)
- [11. Check your branch for a regression](#11-check-your-branch-for-a-regression)
  - [Failure handling](#failure-handling)
- [12. Checklists](#12-checklists)
  - [I added a sweep parameter](#i-added-a-sweep-parameter)
  - [I added a perf test](#i-added-a-perf-test)
  - [I renamed a perf test](#i-renamed-a-perf-test)
  - [I renamed a column](#i-renamed-a-column)
- [13. File map](#13-file-map)

---

## 0. The two rules

**Rule 1 — a new CSV column must exist in the published table.**
Add it to `DB_SCHEMA` in `helpers/perf/wide_schema.py` (or
`wide_schema_quasar.py` for Quasar).
If you do not, the Parquet writer raises `PerfSchemaError` and the perf run
fails. The CSV is already on disk, so the failure comes at the end of a long run.

**Rule 2 — a change to a perf test's columns must be recorded.**
Update the test's entry in `helpers/perf/test_schemas.py` and increase its
`version`.
If you do not, `test_perf_header_gate.py` fails.

Everything below is these two rules applied to a specific case.

---

## 1. The gates, and what each one means

Run them all before you push:

```bash
python3 -m pytest test_perf_header_gate.py test_perf_report_hw_free.py \
                 test_perf_parquet.py test_perf_migrate.py test_perf_publish_run.py -q
```

| Failure message contains | Cause | Fix |
|---|---|---|
| `per-perf-test CSV schema(s) drifted` and `+['x']` | The test now emits column `x`, and the catalog does not list it | Add `x` to that test's `columns` and increase `version` — [§2](#2-add-a-new-sweep-parameter) |
| `per-perf-test CSV schema(s) drifted` and `-['x']` | The catalog lists `x`, and the test no longer emits it | Remove `x` from `columns`, add an `aliases` entry if it was renamed, increase `version` |
| `perf test found in source but missing from the catalog` | A new perf test has no catalog entry | Add one — [§4](#4-add-a-new-perf-test) |
| `in catalog but no longer a ... perf test` | A test was renamed or deleted | [§5](#5-rename-a-perf-test) or [§6](#6-delete-a-perf-test) |
| `test_name_aliases must include the current name` | A test was renamed in place, and the alias map was not updated | [§5](#5-rename-a-perf-test) |
| `a parameter field name is declared by more than one class` | Two parameter classes declare the same field | Rename one field — [§3](#3-add-a-new-parameter-class) |
| `collide with reserved headers` | A field name uses a reserved header name | Rename the field — [§3](#3-add-a-new-parameter-class) |
| `A parameter type is used more than once in a single test config` | One `templates=`/`runtimes=` list uses the same class twice | Use it once, or split it into two classes with unique field names |
| `A parameter class is declared more than once with different fields` | The later definition silently shadows the earlier one | Consolidate to one definition, or use distinct class names |
| `PerfRunType members ... drifted` | A run type was added or renamed | [§8](#8-add-or-rename-a-run-type) |
| `Efficiency metric names drifted` | A `*_pct` metric key changed | [§9](#9-add-or-rename-an-efficiency-metric) |
| `columns not in schema, would be DROPPED` | A CSV column is not in the published table | Rule 1: add it to `wide_schema.py` |
| `values that don't fit ... would become NULL` | A value does not match its declared type | Fix the type in `wide_schema.py`, or fix the value the test emits |
| `Perf report has duplicate column header(s)` | Two columns share one header at run time | Rename one parameter field |

Every gate message names the file to edit. Read the message before you read the
test.

---

## 2. Add a new sweep parameter

The case: a perf test must sweep something new, for example a `THROTTLE_LEVEL`.

### Procedure

1. **Look for an existing parameter class.** Open
   `helpers/test_variant_parameters.py`. If a class already carries the value,
   reuse it and go to step 4.
2. **If no class carries it, add one.** See [§3](#3-add-a-new-parameter-class).
3. **Check the field name.** The dataclass field name **becomes the CSV column
   name**. `THROTTLE_LEVEL.throttle_level` produces a `throttle_level` column.
4. **Pass the parameter in the test's `PerfConfig` call**, in `templates=` for a
   compile-time value or `runtimes=` for a runtime value:

   ```python
   configuration = PerfConfig(
       "sources/eltwise_binary_fpu_perf.cpp",
       formats,
       run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
       templates=[MATH_FIDELITY(math_fidelity), THROTTLE_LEVEL(throttle)],
       runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(8)],
       dest_acc=dest_acc,
   )
   ```

   **Keep the list a literal.** The gate reads `templates=` and `runtimes=` with
   `ast`. It handles three shapes: inline in the call, a named variable, or a
   dict entry. It cannot see a list built by a comprehension or returned by a
   helper.

5. **Add the column to the published table.** Edit
   `helpers/perf/wide_schema.py` and add one entry to `DB_SCHEMA`:

   ```python
   Column("throttle_level", "int64", True, "configuration"),
   ```

   - `nullable` is `True`. Always, for a sweep column. Other tests do not emit
     it, and their rows must be allowed to hold NULL.
   - `dtype` is one of `int64`, `float64`, `bool`, `string`. An enum is `string`.
   - `category` is `configuration` for a sweep parameter.
   - Keep the `configuration` block alphabetical.
   - For a Quasar-only column, edit `wide_schema_quasar.py` instead. Never mix
     Quasar columns into the WH/BH table.

6. **Add the column to the test's catalog entry.** Edit
   `helpers/perf/test_schemas.py`:

   ```python
   "perf_eltwise_binary": {
       "version": 5,                    # was 4 — increase it
       "columns": [
           ...,
           "throttle_level",            # keep the list sorted
           ...,
       ],
       "aliases": {...},                # unchanged
       "test_name_aliases": {...},      # unchanged
   },
   ```

7. **Run the gates.**

   ```bash
   python3 -m pytest test_perf_header_gate.py test_perf_report_hw_free.py -q
   ```

### What each skipped step costs you

| Skipped step | Symptom |
|---|---|
| 5 (`wide_schema.py`) | The perf run fails at the end with `PerfSchemaError: columns not in schema, would be DROPPED: ['throttle_level']`. The CSV survives; the Parquet does not. |
| 6 (`test_schemas.py`) | `test_perf_test_schemas_match` fails: `'perf_eltwise_binary' (schema v4): +['throttle_level'] -[]` |
| 6, `version` only | The gate passes and the harm is silent. Two reports of the test are no longer comparable, and nothing says so. **Always increase the version.** |

### Why the version matters

`version` is the only signal a downstream reader has that two reports of the same
test measure the same thing. The catalog is hand-maintained on purpose: a header
change becomes a reviewed diff in a PR, not a surprise in the data.

---

## 3. Add a new parameter class

The case: no existing class carries the value you must sweep.

### Procedure

1. **Choose the base class.**
   - `TemplateParameter` — a compile-time value. It becomes a `constexpr` in the
     generated header, so each value produces its own ELF.
   - `RuntimeParameter` — a runtime value, passed in a struct. It does not
     multiply the number of builds.
2. **Write the class** in `helpers/test_variant_parameters.py`:

   ```python
   @dataclass
   class THROTTLE_LEVEL(TemplateParameter):
       throttle_level: int = 0

       def convert_to_cpp(self) -> str:
           return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"
   ```

   A `RuntimeParameter` also implements `convert_to_struct_fields`.

3. **Give the field a globally unique name.**

### The naming rules

| Rule | Gate that enforces it |
|---|---|
| No two parameter classes may declare the same field name | `test_parameter_field_names_are_globally_unique` |
| A field name must not be a reserved header | `test_no_parameter_field_equals_a_fixed_header` |
| A class must be defined once, or identically | `test_no_shadowed_parameter_class` |
| One class appears at most once per `templates=`/`runtimes=` list | `test_no_param_type_used_twice_in_one_config` |

**Reserved header names.** Do not use any of these as a field name:

```
formats.input_A   formats.input_B   formats.register_A   formats.register_B
formats.output    formats.sfpu_src  formats.sfpu_dst
unpack_to_dest    dest_acc          marker               test_name
```

`marker` is reserved for a specific reason: the pipeline merges run types on
`marker`, and a parameter named `marker` would be renamed to `marker_x` /
`marker_y` by that merge. It would escape the duplicate-column gate and break
marker processing in silence.

`loop_factor` and `tile_cnt` are **not** reserved. They already are parameter
fields — `LOOP_FACTOR.loop_factor` and `TILE_COUNT.tile_cnt`. Do not declare them
again in a second class.

**A field with default `None` is optional.** The runtime emits a column for it
only when the call sets it, and the gate follows the same rule. `MATH_OP` uses
this: `pool_type` and `unary_extra` default to `None`, so they do not appear as
columns unless a test passes them.

**Two of the same quantity.** If a test needs an input tile count and an output
tile count, do **not** pass `TILE_COUNT` twice. Model them as two classes with
unique field names, for example `INPUT_TILE_CNT` and `OUTPUT_TILE_CNT`.

---

## 4. Add a new perf test

### Procedure

1. **Name the file.**
   - WH/BH: `perf_<stem>.py` at the `python_tests` root.
   - Quasar: `quasar/perf_<stem>_quasar.py`.

   The stem must match the functional test, `test_<stem>.py`. The stem is also
   the `test_name` value in the published table.

2. **Write the test.** Mark it, sweep it, run it:

   ```python
   @pytest.mark.perf
   @parametrize(
       formats=input_output_formats([DataFormat.Float16_b]),
       mathop=[MathOperation.Elwadd],
       tile_count=16,
   )
   def test_perf_my_op(perf_report, formats, mathop, tile_count):
       configuration = PerfConfig(
           "sources/my_op_perf.cpp",
           formats,
           run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
           templates=[MATH_OP(mathop=mathop)],
           runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(8)],
       )
       configuration.run(perf_report)
   ```

   The `@pytest.mark.perf` mark is required. Without it, CI deselects the test
   and the developer tools cannot find its CSV.

3. **Get the column list.** Do not write it by hand:

   ```bash
   python3 -c "import perf_schema_derive as d; print(d.derive_perf_test_schemas()['perf_my_op'])"
   ```

   For Quasar, pass `quasar=True`. You can also run the gate and copy the `+[...]`
   set from the failure message.

4. **Add the catalog entry** in `helpers/perf/test_schemas.py`:

   ```python
   "perf_my_op": {
       "version": 1,
       "columns": [ ...the derived list, sorted... ],
       "test_name_aliases": {"perf_my_op": "perf_my_op"},
   },
   ```

   A new test starts at `version` 1. `test_name_aliases` is **required**, and it
   must map the name to itself. `aliases` is optional; add it only when a column
   is renamed later.

   Add it to `PERF_TEST_SCHEMAS_QSR` for a Quasar test.

5. **Confirm every column is in the published table.** Compare your column list
   against `DB_SCHEMA` in `wide_schema.py`. Add whatever is missing, as nullable.

6. **Run the gates.**

### Where the two derivation paths differ

The static gate reads source with `ast`. The hardware-free report test runs the
real code with the device seams stubbed.

The static gate cannot see a parameter built at run time. If your test builds its
parameter list dynamically, the static gate under-reports, and
`test_perf_report_hw_free.py` is the exact check. Prefer literal lists; then both
agree.

---

## 5. Rename a perf test

A rename touches three places. Miss the third and the gate fails.

### Procedure

1. **Rename the file**: `perf_old_name.py` to `perf_new_name.py`.
2. **Rename the catalog key** in `helpers/perf/test_schemas.py`.
3. **Update `test_name_aliases`.** This is the step people forget:

   ```python
   "perf_new_name": {
       "version": 4,                     # unchanged: the columns did not change
       "columns": [...],
       "test_name_aliases": {
           "perf_new_name": "perf_new_name",   # the current name maps to itself
           "perf_old_name": "perf_new_name",   # every previous name maps forward
       },
   },
   ```

### The rules the gate enforces

| Rule | Reason |
|---|---|
| The current name must map to itself | Rename the key without editing the map, and the identity entry still points at the old name. That is the failure the gate is designed to catch. |
| Every value must equal the catalog key | An alias that points at a third name is a typo, not a rename. |
| An old name must not remain a catalog key | Point aliases at the surviving entry only. |
| Two entries must not claim the same old name | The history would be ambiguous. |

### What a rename costs downstream

`test_name` in the published Parquet is the module stem, and it is **not**
remapped in the repository. A renamed test starts a new series in any external
dashboard keyed on `test_name`. Per-config trends survive in the table; the
dashboard sees a discontinuity, not data loss.

That is what `test_name_aliases` is for: it is the old-to-new map a reader needs
to stitch the two series. Keep it complete.

Also update:

- cross-references in `helpers/perf/test_schemas.py` comments
- any skill or document that names the test
- `compare_test_and_perf.py` expectations, if the functional counterpart moved too

---

## 6. Delete a perf test

1. Delete the test file.
2. Delete its catalog entry from `helpers/perf/test_schemas.py`.
3. Leave `DB_SCHEMA` alone. A column another test still emits must stay. A column
   nobody emits does no harm — it is nullable, and removing it is a breaking
   change for the published table.
4. Run the gates. `in catalog but no longer a perf test` means step 2 is missing.

If the test was **absorbed** into another test rather than deleted, record the old
name in the surviving test's `test_name_aliases`, and update that test's
`columns` and `version` if its sweep grew.

---

## 7. Rename a column

The case: `formats.sfpu_math` becomes `formats.sfpu_src`.

1. Rename the field, or the header builder, at its single owner.
2. Update `DB_SCHEMA` in `wide_schema.py`: rename the `Column` entry.
3. For **every** affected test in `test_schemas.py`:
   - replace the old name in `columns` with the new name
   - add an `aliases` entry: `{"formats.sfpu_math": "formats.sfpu_src"}`
   - increase `version`
4. Run the gates.

`aliases` is what lets a downstream reader join a v3 report to a v4 report. A
rename without it silently splits the history in two.

This is the case that broke `main` once (#52058). A header rename reached the
tests before the catalog. Do all steps in one PR.

---

## 8. Add or rename a run type

A run type prefixes every metric and counter column, so the vocabulary is gated
globally.

1. Add the member to `PerfRunType` in `helpers/llk_params.py`.
2. Add the same name to `RUN_TYPE_NAMES` in `helpers/perf/schema.py`.
3. Add the timing columns to `wide_schema.py`. Timing columns are formula-driven:
   append the base name to `_TIMING_BASES`, and the `{mean, std}` grid follows.
4. Handle the run type in the kernel. `PERF_RUN_TYPE` is a template parameter, so
   each run type compiles its own ELF.
5. Run the gates. `PerfRunType members ... drifted` means step 2 is missing.

**Cost warning.** Cost scales with (variants x selected run types). Across 23
WH/BH perf modules there are about 70 (module x run-type) pairs today. A new run
type on every module is a large cost increase. To scope a local sweep, select
fewer tests with `-k`, or use `--perf-run-types` from the
`nstojictt/perf-gate-budget` branch — that flag is not yet on `main`.

---

## 9. Add or rename an efficiency metric

1. Add the `*_pct` key to the metric dictionary in `helpers/metrics.py`. Only
   keys that end in `_pct` are exported.
2. Add the same name to `METRIC_BASES` in `helpers/perf/schema.py`.
3. Run the gates. `Efficiency metric names drifted` names both directions: in
   source but not the catalog, and in the catalog but not source.

The gate reads the dictionary keys with `ast`, not with a text scan, so an
unrelated `"_pct"` string in a log line can neither trip nor evade it.

Metric columns are **not** in `DB_SCHEMA` yet. They join the published table as
nullable once a counter run captures their exact names (issue #51249).

---

## 10. Run a perf report

Use the two-phase producer/consumer flow. Never one serial invocation.

```bash
pytest --compile-producer -n 10 -m "perf and not accuracy" <selection>  # compile, then skip
pytest --compile-consumer -n 15 -m "perf and not accuracy" <selection>  # load ELFs, measure
```

`tests/run_llk_perf_<arch>.sh` is the CI runner and does exactly this.

**`-n 15` is not a performance choice.** The xdist worker index maps onto a
physical Tensix core: `row, col = divmod(int(worker_id[2:]), 8)`. `-n 15` promises
that only cores (0,0) to (1,6) are used. `-n auto` would address an 8x8 grid, and
Wormhole ships with harvested rows, so `-n auto` is **wrong**, not merely slower.
Compile parallelises freely; measurement does not.

Two more things to avoid:

- **Drop `-x` for a report run.** The CI runner passes it on the consumer phase,
  which is right for CI: fail fast. It is wrong for a report. `-x` aborts
  mid-sweep, and the combined CSV is then silently partial. Downstream that reads
  as measurements that went missing, not as a run that stopped early.
- **Do not narrow the sweep in the test file to make a run finish.** Narrow the
  selection with `-k` instead, and say so in the report.

### Where the output goes

```
perf_data/runs/<tag>/<test>/<test>.csv          raw loop totals
perf_data/runs/<tag>/<test>/<test>.post.csv     per-tile figures
perf_data/runs/<tag>/<tag>.parquet              one typed batch, all tests
perf_data/latest -> runs/<tag>                  symlink to the newest run
```

`perf_data/` is gitignored. A report is a build artifact, not a repository file.
What identifies it is the test, the architecture, the commit, and the exact
command. Record all four.

`PERF_KEEP_RUNS` bounds local history; it defaults to 10.

### Reading a number

- **`TILE_LOOP` is the number that matters.** It is the steady-state loop.
  `INIT` and `UNINIT` are one-time setup and teardown, a few hundred cycles each.
- **`<test>.csv` holds raw loop totals.** For cycles per tile, divide by
  `loop_factor * tile_cnt`, or read `<test>.post.csv`.
- **Lower is always better.** Every cycle metric measures time.
- **`std(...)` is usually absent.** `PerfConfig.run()` defaults to `run_count=1`,
  so each value is a single device execution. Raise `run_count` to get `std`
  columns at no extra compile cost.
- **`.cycles` counter columns are each bank's total elapsed zone time**, not
  cycles attributable to that counter. This trap has cost time before.
- **Isolate kernels are different binaries.** In `MATH_ISOLATE`, the other
  threads are not doing real work. Do not read an isolate as a share of
  `L1_TO_L1`.

---

## 11. Check your branch for a regression

Before you push:

```bash
S=tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh

$S blackhole perf_math_matmul                       # HEAD vs the branch point
$S blackhole perf_math_matmul --baseline v0.60.0    # vs a tag
$S wormhole perf_math_matmul --baseline 1a2b3c4 --current 9f8e7d6
$S blackhole perf_math_matmul --dry-run             # what would it measure?
```

In Claude Code: `/perf-regression-check perf_math_matmul`.

The default baseline is `git merge-base origin/main HEAD` — the commit you
branched from, not latest main. For "did *this* work regress perf", that is the
honest baseline.

What the script guarantees:

- Your checkout is never touched. Each commit runs in its own sparse worktree.
- Only **committed** code is measured. Commit your edits first.
- Runs are cached per (arch, test, variant, commit). `--iterations 5` after a
  3-iteration run measures only two more per side.
- Iterations are interleaved, so machine drift hits both sides equally.

Interpret the result with care:

- Comparison is per (marker, sweep config), on `mean(<run_type>)` cycles, median
  across iterations.
- If the sweep changed between the commits, expect "new points". They are
  reported, never counted as regressions.
- A delta near the threshold can still be noise. Raise `--iterations`.
- **Matmul is bistable.** A 2 to 6% step on `perf_matmul` or `perf_math_matmul`
  may be the known hardware state, not your change. See
  [architecture.md §10](architecture.md#10-the-gate-plan-and-the-evidence-behind-it).

Then use the dashboard's **Branch** tab to judge your CI perf run against recent
`main` nightlies with the same engine as the nightly scan.

### Failure handling

| Symptom | Action |
|---|---|
| `TENSIX TIMED OUT` or a device hang | `tt-smi -r`, then re-run the same command. Completed iterations are cached. |
| A missing header at compile time on an older commit | Re-run with `SPARSE_PATHS=` (empty) for a full checkout. |
| `no perf CSV` | Check the module name, and that the test carries the `perf` mark. |
| `does not exist at <sha>` | The test was added after that commit. Pick a later baseline. |

---

## 12. Checklists

### I added a sweep parameter

- [ ] The parameter class exists, and the field name is globally unique
- [ ] The field name is not a reserved header
- [ ] The class appears at most once in the `templates=`/`runtimes=` list
- [ ] The list is a literal
- [ ] `DB_SCHEMA` has the column, nullable, with the right dtype
- [ ] The test's `columns` list has the column
- [ ] The test's `version` is increased
- [ ] `test_perf_header_gate.py` and `test_perf_report_hw_free.py` pass

### I added a perf test

- [ ] The file is `perf_<stem>.py`, and the stem matches `test_<stem>.py`
- [ ] The test carries `@pytest.mark.perf`
- [ ] A catalog entry exists: `version` 1, derived `columns`, `test_name_aliases`
      mapping the name to itself
- [ ] Every column is in `DB_SCHEMA`
- [ ] The gates pass

### I renamed a perf test

- [ ] The file is renamed
- [ ] The catalog key is renamed
- [ ] `test_name_aliases` maps the new name to itself **and** the old name to the
      new name
- [ ] `version` is unchanged, if the columns did not change
- [ ] Skills and documents that name the test are updated
- [ ] The gates pass

### I renamed a column

- [ ] The single owner is updated
- [ ] `DB_SCHEMA` is updated
- [ ] Every affected test has the new name in `columns`
- [ ] Every affected test has an `aliases` entry, old to new
- [ ] Every affected test has an increased `version`
- [ ] The gates pass

---

## 13. File map

| File | What it holds |
|---|---|
| `helpers/perf/schema.py` | Header names, header builders, run-type and metric vocabularies |
| `helpers/perf/wide_schema.py` | `DB_SCHEMA`: the published WH/BH table |
| `helpers/perf/wide_schema_quasar.py` | The published Quasar table |
| `helpers/perf/test_schemas.py` | Per-test column catalog, versions, aliases |
| `helpers/perf/core.py` | `PerfReport`, `PerfConfig`, `combine_perf_reports` |
| `helpers/perf/parquet.py` | CSV to Parquet, Parquet to CSV, typed writer |
| `helpers/perf/publish_run.py` | CLI: one run to one `run.parquet` |
| `helpers/perf/migrate.py` | CLI: a historical archive to Parquet |
| `helpers/test_variant_parameters.py` | Every parameter class |
| `helpers/llk_params.py` | `PerfRunType` and the enums parameters carry |
| `helpers/metrics.py` | Efficiency metrics, the `*_pct` keys |
| `perf_schema_derive.py` | Static column derivation with `ast` |
| `test_perf_header_gate.py` | Every schema and uniqueness gate |
| `test_perf_report_hw_free.py` | The real report code, no hardware |
| `test_perf_parquet.py` | Parquet round trip and typing |
| `test_perf_migrate.py` | Migration determinism and leniency |
| `test_perf_publish_run.py` | The publish CLI |
