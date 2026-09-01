# LLK perf infrastructure

Perf tests write CSVs. Each run's CSVs become one typed table, which the dashboard and
the future PR gate read. Hardware-free gates keep the columns from drifting.

Run every command from `tt_metal/tt-llk/tests/python_tests`.

[The two rules](#the-two-rules) · [I want to…](#i-want-to) · [Tasks](#tasks) ·
[Common pitfalls](#common-pitfalls) · [How it works](#how-it-works)

## The two rules

| # | Rule | If you skip it |
|---|---|---|
| 1 | A new CSV column must exist in `DB_SCHEMA` (`helpers/perf/wide_schema.py`). | The Parquet writer raises `PerfSchemaError` at the end of the run. |
| 2 | A change to a test's columns must be recorded in `helpers/perf/test_schemas.py`, with `version` increased. | `test_perf_header_gate.py` fails. |

Before you push:

```bash
python3 -m pytest test_perf_header_gate.py test_perf_report_hw_free.py \
                 test_perf_parquet.py test_perf_migrate.py test_perf_publish_run.py -q
```

## I want to…

| Task | |
|---|---|
| Fix a gate that just failed | [→](#a-gate-failed) |
| Add a sweep parameter to a perf test | [→](#add-a-sweep-parameter) |
| Add a new parameter class | [→](#add-a-parameter-class) |
| Add a new perf test | [→](#add-a-perf-test) |
| Rename a perf test | [→](#rename-a-perf-test) |
| Delete or absorb a perf test | [→](#delete-or-absorb-a-perf-test) |
| Rename a column | [→](#rename-a-column) |
| Add a run type or an efficiency metric | [→](#add-a-run-type-or-an-efficiency-metric) |
| Produce a perf report | [→](#produce-a-perf-report) |
| Check whether my branch regressed perf | [→](#did-my-branch-regress-perf) |
| Check what regressed in nightly | [→](#what-regressed-in-nightly) |
| Read a run as a table, or query history | [→](#read-a-run-as-a-table) |
| Avoid a mistake people keep making | [→](#common-pitfalls) |

---

# Tasks

## A gate failed

Match the assertion text.

| Message contains | Fix |
|---|---|
| `schema(s) drifted` … `+['x']` | The test emits `x`, the catalog does not list it. Add it to `columns`, increase `version`. |
| `schema(s) drifted` … `-['x']` | The catalog lists `x`, the test does not emit it. Remove it, add an `aliases` entry if renamed, increase `version`. |
| `found in source but missing from the catalog` | [Add a catalog entry](#add-a-perf-test) |
| `in catalog but no longer a … perf test` | [Rename](#rename-a-perf-test) or [delete](#delete-or-absorb-a-perf-test) |
| `must include the current name` | Renamed in place, alias map not updated → [rename](#rename-a-perf-test) |
| `declared by more than one class` | Two parameter classes share a field name. Rename one. |
| `collide with reserved headers` | A field name uses a [reserved header](#add-a-parameter-class). Rename it. |
| `used more than once in a single test config` | One list holds the same class twice → [pitfall](#using-one-parameter-class-twice-in-a-config) |
| `declared more than once with different fields` | A class is defined twice; the later shadows the earlier. Consolidate. |
| `PerfRunType members … drifted` | Add the name to `RUN_TYPE_NAMES`. |
| `Efficiency metric names drifted` | Add the name to `METRIC_BASES`. |
| `columns not in schema, would be DROPPED` | Rule 1. Add the column to `wide_schema.py`. |
| `values that don't fit … would become NULL` | The declared type is wrong, or the test emits a bad value. |
| `duplicate column header(s)` | Two columns share one header at run time. Rename one field. |

## Add a sweep parameter

The parameter class's field name **becomes the CSV column name**.

**1. Pass it** in `PerfConfig` — `templates=` for a compile-time value, `runtimes=` for a
runtime value. If no class carries it, [write one](#add-a-parameter-class).

```python
PerfConfig(
    "sources/eltwise_binary_fpu_perf.cpp", formats,
    run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
    templates=[MATH_FIDELITY(math_fidelity), THROTTLE_LEVEL(throttle)],
    runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(8)],
).run(perf_report)
```

Keep the list a literal — [why](#a-dynamically-built-parameter-list-is-invisible-to-the-gate).

**2. Add the column to `wide_schema.py`:**

```python
Column("throttle_level", "int64", True, "configuration"),
```

Always `nullable=True`; other tests do not emit it. `dtype` is `int64` | `float64` |
`bool` | `string` (an enum is `string`). Keep the block alphabetical. A Quasar-only
column goes in `wide_schema_quasar.py` — never mix the two tables.

**3. Add the column to `test_schemas.py`:**

```python
"perf_eltwise_binary": {
    "version": 5,                              # was 4 — increase it
    "columns": [..., "throttle_level", ...],   # keep sorted
    ...
},
```

**4. Run the gates.**

## Add a parameter class

In `helpers/test_variant_parameters.py`. Subclass `TemplateParameter` for a compile-time
value — it becomes a `constexpr`, so each value produces its own ELF — or
`RuntimeParameter` for a runtime value passed in a struct, with no extra builds.

```python
@dataclass
class THROTTLE_LEVEL(TemplateParameter):
    throttle_level: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"
```

A `RuntimeParameter` also implements `convert_to_struct_fields`.

Four naming rules, each with a gate:

| Rule | Gate |
|---|---|
| No two parameter classes declare the same field name | `test_parameter_field_names_are_globally_unique` |
| The field name is not a reserved header | `test_no_parameter_field_equals_a_fixed_header` |
| The class is defined once, or identically | `test_no_shadowed_parameter_class` |
| One class appears at most once per list | `test_no_param_type_used_twice_in_one_config` |

**Reserved**: the seven `formats.*` headers, `unpack_to_dest`, `dest_acc`, `marker`,
`test_name`. `loop_factor` and `tile_cnt` are **not** reserved — they already are the
fields of `LOOP_FACTOR` and `TILE_COUNT`, so do not declare them again.

A field defaulting to `None` is optional: it becomes a column only when a call sets it.

## Add a perf test

**1. Name the file** `perf_<stem>.py` at the `python_tests` root, or
`quasar/perf_<stem>_quasar.py`. The stem must match the functional test
`test_<stem>.py`, and is also the `test_name` value in the published table.

**2. Write it.** `@pytest.mark.perf` is required — without it CI deselects the test and
the developer tools cannot find its CSV.

```python
@pytest.mark.perf
@parametrize(formats=input_output_formats([DataFormat.Float16_b]), tile_count=16)
def test_perf_my_op(perf_report, formats, tile_count):
    PerfConfig(
        "sources/my_op_perf.cpp", formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(8)],
    ).run(perf_report)
```

**3. Get the column list** — do not write it by hand:

```bash
python3 -c "import perf_schema_derive as d; print(d.derive_perf_test_schemas()['perf_my_op'])"
```

Pass `quasar=True` for Quasar. The gate's `+[...]` failure output works too.

**4. Add the catalog entry** (`PERF_TEST_SCHEMAS_QSR` for Quasar). `test_name_aliases`
is required and must map the name to itself:

```python
"perf_my_op": {
    "version": 1,
    "columns": [ ...the derived list, sorted... ],
    "test_name_aliases": {"perf_my_op": "perf_my_op"},
},
```

**5. Confirm every column is in `DB_SCHEMA`**, as nullable. **6. Run the gates.**

## Rename a perf test

Rename the file, rename the catalog key, then update `test_name_aliases` — the third
step is the one people miss.

```python
"perf_new_name": {
    "version": 4,                              # unchanged: the columns did not change
    "columns": [...],
    "test_name_aliases": {
        "perf_new_name": "perf_new_name",      # current name maps to itself
        "perf_old_name": "perf_new_name",      # every previous name maps forward
    },
},
```

The gate requires the current name to map to itself, every value to equal the catalog
key, and an old name never to remain a key of its own. Renaming the key without editing
the map is the exact case it exists for.

Also update any skill or document that names the test, and see
[the downstream cost](#renaming-a-test-starts-a-new-dashboard-series).

## Delete or absorb a perf test

Delete the file and its catalog entry. Leave `DB_SCHEMA` alone: a column another test
emits must stay, and a column nobody emits is harmless — it is nullable, and removing it
breaks the published table.

**Absorbed** into another test: record the old name in the surviving test's
`test_name_aliases`, and update its `columns` and `version` if the sweep grew.

## Rename a column

1. Rename it at its single owner.
2. Rename the `Column` entry in `wide_schema.py`.
3. For **every** affected test in `test_schemas.py`: replace the name in `columns`, add
   `"aliases": {"<old>": "<new>"}`, increase `version`.
4. Run the gates.

`aliases` is what lets a reader join a v3 report to a v4 report; without it the history
silently splits in two. Do all four steps in one PR — a rename that reached the tests
before the catalog broke `main` once (#52058).

## Add a run type or an efficiency metric

**A run type** prefixes every metric and counter column, so the vocabulary is gated
globally.

1. Add the member to `PerfRunType` in `helpers/llk_params.py`.
2. Add the same name to `RUN_TYPE_NAMES` in `helpers/perf/schema.py`.
3. Append the base name to `_TIMING_BASES` in `wide_schema.py`; the `{mean, std}` grid
   follows from it.
4. Handle it in the kernel. `PERF_RUN_TYPE` is a template parameter, so each run type
   compiles its own ELF.

Cost scales with (variants × selected run types) — about 70 (module × run-type) pairs
across 23 WH/BH modules today. To scope a local sweep, select fewer tests with `-k`.

**An efficiency metric**: add the `*_pct` key to the metric dictionary in
`helpers/metrics.py` (only `_pct` keys are exported), then the same name to
`METRIC_BASES`. Metric columns are not in `DB_SCHEMA` yet; they join as nullable once a
counter run captures their exact names (#51249).

## Produce a perf report

Two phases, never one serial invocation:

```bash
pytest --compile-producer -n 10 -m "perf and not accuracy" <selection>  # compile, then skip
pytest --compile-consumer -n 15 -m "perf and not accuracy" <selection>  # load ELFs, measure
```

`tests/run_llk_perf_<arch>.sh` is the CI runner and does exactly this. `-n 15` is a
correctness constraint — [why](#-n-auto-addresses-cores-that-may-not-exist).

```
perf_data/runs/<tag>/<test>/<test>.csv          raw loop totals
perf_data/runs/<tag>/<test>/<test>.post.csv     per-tile figures
perf_data/runs/<tag>/<tag>.parquet              one typed batch, all tests
perf_data/latest -> runs/<tag>                  symlink to the newest run
```

`perf_data/` is gitignored — a report is a build artifact. What identifies it is the
test, the architecture, the commit and the exact command; record all four.
`PERF_KEEP_RUNS` bounds local history (default 10).

## Did my branch regress perf?

One test, any two commits, on your machine:

```bash
S=tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh

$S blackhole perf_math_matmul                     # HEAD vs the branch point
$S wormhole  perf_math_matmul --baseline 1a2b3c4 --current 9f8e7d6
$S blackhole perf_math_matmul --dry-run           # resolve refs, measure nothing
```

In Claude Code: `/perf-regression-check perf_math_matmul`. Defaults: baseline
`merge-base origin/main HEAD`, `--iterations 3`, `--threshold 0.05`; also
`--speed-of-light` and `--refresh`.

Each commit runs in its own sparse worktree, so your checkout is never touched and a
dirty tree is fine — but only **committed** code is measured. Runs are cached per
(arch, test, variant, commit), so raising `--iterations` only measures the difference.
Iterations interleave, so machine drift hits both sides equally.

Comparison is per (marker, sweep config) on `mean(<run_type>)` cycles, median across
iterations. A config with no baseline is a "new point", never a regression.

| Symptom | Action |
|---|---|
| `TENSIX TIMED OUT`, device hang | `tt-smi -r`, re-run. Completed iterations are cached. |
| Missing header at compile time on an older commit | Re-run with `SPARSE_PATHS=` (empty) |
| `no perf CSV` | Check the module name and the `perf` mark |
| `does not exist at <sha>` | The test postdates that commit. Pick a later baseline. |

On CI, use the dashboard's **Branch** tab: enter your GitHub username, pick your run,
**Compare vs main**.

## What regressed in nightly?

The LLK Perf Reporter dashboard.

| Question | Tab | Path |
|---|---|---|
| What regressed last night? | **Scan** | Pick the architecture → **Scan** |
| What changed since the last release? | **Scan** | **Load runs to pin…** → set Baseline → **Scan** |
| Which commit caused it? | **Scan** / **Trends** | Click the row → **View commits in this step** |
| How did this number move? | **Trends** | Load tests → pick test + metric → **Plot trend** |
| Unpack-, math-, or pack-bound? | **Stage breakdown** | Plot in Trends → **Stage breakdown** |
| What does this parameter cost? | **Param impact** | Pick test, marker, metric, parameter |

Scan flags a point beyond *k* sigma from a median baseline of recent nightlies, where
sigma is a robust MAD floored at 0.3% of the baseline (default *k* = 3). `sustained`
means the shift held two or more runs. One-sided groups rank first, because balanced
movement is noise. Read the coverage line first —
[why](#lost-coverage-looks-exactly-like-an-improvement).

## Read a run as a table

```bash
python3 -m helpers.perf.publish_run --csv-dir perf_data/latest --out run.parquet --arch wormhole
```

Point `--csv-dir` at **one** run, not at `perf_data`
([why](#pointing-publish_run-at-perf_data)). `--strict` fails on schema drift instead of
dropping and coercing. Requires `COMMIT_SHA`, `RUN_ID` and `PIPELINE` (`PR` | `nightly`).

A historical archive → one Parquet per run: `helpers/perf/migrate.py`. Parquet →
per-test CSVs: `parquet.parquet_to_csvs(path, out_dir)`. Query history: the
[warehouse seam](#storage).

---

# Common pitfalls

Mistakes that pass every gate, or fail far from their cause.

### Changing `columns` without increasing `version`

The gate compares the column **set**, so this passes, and nothing else notices either.
`version` is the only signal that two reports of the same test measure the same thing —
two v4 reports with different columns are silently incomparable, and it surfaces weeks
later as a trend nobody can explain.

### A dynamically built parameter list is invisible to the gate

The gate reads `templates=` and `runtimes=` with `ast`. It handles a list written inline,
assigned to a variable, or given as a dict entry — but not one built by a comprehension
or returned by a helper. Such a test under-reports its columns, the gate passes, and the
run then fails at the Parquet writer. Keep the lists literal; if you cannot,
`test_perf_report_hw_free.py` is the exact check.

### `-n auto` addresses cores that may not exist

The xdist worker index maps onto a physical Tensix core —
`row, col = divmod(int(worker_id[2:]), 8)`. `-n 15` promises that only cores (0,0) to
(1,6) are used. `-n auto` addresses an 8×8 grid, and Wormhole ships with harvested rows.
It is **wrong**, not merely slower. Compile parallelises freely; measurement does not.

### `-x` on a report run, or narrowing the sweep in the test file

`-x` is right for CI and wrong for a report: it aborts mid-sweep and the combined CSV is
silently partial, which downstream reads as measurements that went missing. Editing
`@parametrize` to make a run finish has the same effect and tends to survive into a
commit. Narrow with `-k` instead, and say so in the report.

### Reading `<test>.csv` as per-tile

`<test>.csv` holds **raw loop totals** for `TILE_LOOP`; the per-tile figure is only in
`<test>.post.csv`. The Parquet stores raw on purpose, because per-tile is derivable —
divide `mean(<RUN_TYPE>)` by `loop_factor * tile_cnt`, both columns. An absolute cycle
threshold derived from `.post.csv` does not transfer to the raw table.

### Trusting a small matmul delta

`perf_matmul` and `perf_math_matmul` are **bistable**: a configuration lands on one of
two discrete values, 2–6% apart on `L1_TO_L1` and up to 24% on `PACK_ISOLATE`, at low
probability per run. It is a state the hardware enters, not measurement error — repeating
runs does not average it out, and the affected set is not a fixed list.

**A 2–6% step on a matmul test is not evidence of a regression.** Everything else in the
sweep is stable well under 2%, and only packer-involved measurements are affected.

### Reading an isolate as a share of `L1_TO_L1`

Each run type is a **separate binary** — `PERF_RUN_TYPE` is a template parameter and the
kernel picks a different path with `if constexpr`. In `MATH_ISOLATE` the other threads
are not doing real work, so the three isolates do not sum to `L1_TO_L1`. Use **Stage
breakdown** for the bound-by question.

### Reading `.cycles` as time spent in that counter

A `<BANK>.<COUNTER>.cycles` column is that **bank's total elapsed zone time**, not cycles
attributable to the counter. It looks like a per-counter figure and is not one.

### Expecting `std(...)` to be there

`PerfConfig.run()` defaults to `run_count=1`, so almost every value is a single device
execution and the all-NaN `std(...)` columns are dropped. Your number therefore carries
the full single-shot jitter — and raising `run_count` averages on device inside one
variant at no extra compile, the cheapest noise lever available.

### Renaming a test starts a new dashboard series

`test_name` in the published Parquet is the module stem and is not remapped in the repo,
so any dashboard keyed on it sees a new series. Per-config trends survive in the table —
this is a discontinuity, not data loss, but nobody is told. Keep `test_name_aliases`
complete, and mention the rename in the PR body.

### Lost coverage looks exactly like an improvement

A config that stopped running has the same shape in a diff as one that got faster. Scan
reports a **coverage** line above the results: configs and tests that stopped or started
between the two runs. Read it before you read the regressions.

### Comparing against latest `main`

`perf_compare_commits.sh` defaults to `git merge-base origin/main HEAD`, deliberately.
The further apart the commits, the more unrelated change lands in the delta.

### Naming a parameter field `marker`

The pipeline merges run types on `marker`, so a parameter named `marker` would be renamed
to `marker_x` / `marker_y` by that merge — evading the duplicate-column gate and breaking
marker processing in silence.

### Using one parameter class twice in a config

```python
runtimes=[TILE_COUNT(4), TILE_COUNT(8)]   # two columns, one header
```

The field name is unique across classes, so the uniqueness gate does not fire —
`test_no_param_type_used_twice_in_one_config` catches it. Model two of the same quantity
as two classes with distinct field names.

### Pointing `publish_run` at `perf_data`

`--csv-dir` globs recursively, so every retained run is swept into one batch under one
run's provenance. Point it at `perf_data/latest`, or a specific `runs/<tag>`.

---

# How it works

Implementation detail. Context for engineers and agents; not needed to complete a task.

## The pipeline

```
perf test on silicon   ->   PerfReport   ->   per-test CSV   ->   perf_data/runs/<tag>/
                            |                |                   |
              one schema per report    <test>.csv (raw)     <tag>.parquet
              enforced                 <test>.post.csv      one typed batch, zstd
                                                                  |
                                             CI artifact  ->  warehouse  ->  dashboard + gate
```

The gates sit on the **transitions**, each guarding a boundary where a schema could drift
unnoticed: `PerfReport` rejects a second column schema in one report;
`test_perf_header_gate` and `test_perf_report_hw_free` check the CSV columns against the
catalog; the Parquet converter raises before writing anything the table does not declare.

## Who owns what

Four modules, four jobs, no duplication. None imports a device library — that is what
makes the schema layer testable on a laptop.

| Module | Owns |
|---|---|
| `helpers/perf/schema.py` | Header names and builders; run-type and metric vocabularies |
| `helpers/perf/wide_schema.py` | `DB_SCHEMA` — the published WH/BH table |
| `helpers/perf/wide_schema_quasar.py` | The published Quasar table |
| `helpers/perf/test_schemas.py` | Per-test columns, versions, aliases |

No header is ever a literal: `stat_column("L1_TO_L1", "mean")` → `mean(L1_TO_L1)`, and
likewise `metric_column`, `text_size_column`, `counter_base`, `cycles_of`.

## The published table

`DB_SCHEMA` is 82 columns; one row is one test configuration in one run. It is both the
table handed to the data team and the exact schema the Parquet is written with.

```python
Column("math_fidelity", "string", True, "configuration")          # origin="test"
Column("commit_sha", "string", False, "provenance", origin="ci")  # stamped by CI
```

Two views derive from that one list, so a column cannot exist in one and not the other:
`OUTPUT_SCHEMA` (75 test-origin columns, what a report is validated against) and
`PROVENANCE` (7 CI-origin columns — `test_name`, `commit_sha`, `arch`, `run_id`,
`timestamp`, `pipeline`, `pr_number`). Row key: `test_name`, `commit_sha`, `arch`,
`run_id`, plus the sweep columns.

Everything except `marker` and the mandatory provenance keys is **nullable**. That is
what lets 23 tests with different sweeps share one table — each fills its own columns and
the rest stay NULL, so the table need not be complete to be useful. Timing columns are
formula-driven: the full `{mean, std} × _TIMING_BASES` grid, so the schema is a superset
of any run configuration rather than of what one nightly sampled.

Deferred: counter and metric columns join as nullable once a counter run captures their
names (#51249); canonicalizing columns that name the same thing differently —
`c_dimm`/`k_dimm`/`r_dimm` vs `in0_c_dim`/`ct_dim` — needs sign-off (#51245).

## The per-test catalog

One entry per perf test — 23 WH/BH, 18 Quasar. Hand-maintained on purpose, so a header
change is a diff in a PR instead of a surprise in the data.

| Field | Job |
|---|---|
| `columns` | The reviewed column set |
| `version` | Two reports of the same test are comparable if the version matches |
| `aliases` | Old column name → new, so a reader can follow a rename |
| `test_name_aliases` | Old test name → current. Required, and must map the name to itself. |

`perf_schema_derive.py` re-derives each test's columns from source with `ast` — the fixed
format and flag headers, plus the parameter fields the test sets, plus `marker`. Quasar
tests derive from their sibling `test_*_quasar.py`, where `PerfConfig` is called.

## The gates

Thirteen, all hardware-free, run with the LLK pytest suites under the default
`not perf and not nightly` filter.

| Gate | Catches |
|---|---|
| `test_perf_test_schemas_match` (+ `_qsr`) | A test's columns drifted from the catalog |
| `test_test_name_aliases_*` (5 tests) | A test renamed without updating the alias map |
| `test_run_type_names_match_source` | A `PerfRunType` member added or renamed |
| `test_metric_bases_match_source` | An efficiency metric name changed |
| `test_parameter_field_names_are_globally_unique` | Two parameter classes share a field name |
| `test_no_shadowed_parameter_class` | A class declared twice with different fields |
| `test_no_parameter_field_equals_a_fixed_header` | A field name collides with a reserved header |
| `test_no_param_type_used_twice_in_one_config` | One list uses the same class twice |

Two derivation paths catch different things. `test_perf_header_gate.py` reads source with
`ast` — every test, cheaply, but blind to parameters built at run time.
`test_perf_report_hw_free.py` runs the real report code with the device seams stubbed —
exact, but slower.

## Run identity

| | `<tag>` (`PERF_RUN_TAG`) | `run_id` |
|---|---|---|
| Purpose | Directory and file name | `ROW_KEY` column in the table |
| Unique per | Invocation, and per CI shard | CI workflow — **shared by every shard** |
| Reaches the table? | No | Yes |
| CI value | `<run_id>-<arch>-<shard>` | `GITHUB_RUN_ID`, plus `-<attempt>` from attempt 2 |
| Local value | `local-<UTC timestamp>` | The tag |

CI sets the tag itself, because only the workflow sees the shard index. Naming files
after `run_id` would give all ten shards the same filename — fine while each stays in its
own directory, wrong the moment they are unzipped together. Attempt 1 stays bare, so rows
already archived keep the identity they were published with.

Three properties of the output directory: **one directory per invocation**, so a narrower
second run cannot leave the first run's directories in place and make the tree read as
complete while holding a blend of two runs; the **`latest` swap is atomic**, so a reader
sees the old run or the new one, never nothing; and **prune cannot delete the current
run**, which is protected by name rather than by being newest.

## Storage

**Parquet.** Two entry points — a live run's frames, or CSVs on disk — feed one core.
`build_run_batch` stamps provenance, concatenates every test's rows, aligns to the schema
(missing → NULL, extras dropped, order fixed), casts to the declared Arrow types, and
checks every non-nullable column is populated. The converter is **strict by default**: an
unknown column, or a value that does not fit its type, raises *before* anything is
written. Under `strict=False` it drops and coerces and logs both — a NULLed value is
otherwise indistinguishable downstream from a column a test never emitted.

**Migration.** `helpers/perf/migrate.py` converts an archive of past runs into the same
schema, reusing the same converter: deterministic (no clocks, sorted order), lenient (a
dirty run is recorded as `failed` and skipped rather than aborting everything), and
idempotent (temp file plus rename).

**Warehouse.** One interface — `load(parquet_path)` and `query(sql)` — chosen by
`PERF_WAREHOUSE`. `DuckDBWarehouse` is a local Parquet-native stand-in so the pipeline
can be built before the Snowflake table exists; `SnowflakeWarehouse` is the real target.
Retiring the stand-in is three lines, and downstream code depends only on the interface.
PR #53021, open.

## Where the perf gate stands

Designed, not shipped (#53752). Intended flow: a PR runs perf, the gate compares against
the baseline for the branch point, a regression blocks the merge, and the post-merge
sanity workflow publishes the new baseline.

The measurement study found the noise is mostly a fixed ~25-cycle offset rather than a
percentage; `MATH_ISOLATE` and `UNPACK_ISOLATE` are gate-ready with zero false failures
across 396,142 measurements; non-matmul `L1_TO_L1` moves at worst 1.59%; and repeating
runs does not help. Gate latency for `L1_TO_L1` on one card is about 9 minutes on
Wormhole, of which 65–79% is compile.

Two defensible thresholds follow, both **excluding matmul**: *2% slower AND more than 50
cycles slower*, or *2% slower on `TILE_LOOP` and `KERNEL` only*. The blocker is
[matmul bistability](#trusting-a-small-matmul-delta) — no threshold absorbs it, repeated
runs do not average it out, and the affected set is not a reproducible list.

Full study: `nstojictt/llk-perf-noise-baseline`, `docs/perf_evaluation/README.md`.

## Tools and file map

| Path | Holds |
|---|---|
| `helpers/perf/schema.py` | Header names, builders, run-type and metric vocabularies |
| `helpers/perf/wide_schema.py` · `_quasar.py` | The published tables |
| `helpers/perf/test_schemas.py` | Per-test columns, versions, aliases |
| `helpers/perf/core.py` | `PerfReport`, `PerfConfig`, `combine_perf_reports` |
| `helpers/perf/parquet.py` | CSV ↔ Parquet, the typed writer |
| `helpers/perf/publish_run.py` | CLI: one run → one `run.parquet` |
| `helpers/perf/migrate.py` | CLI: a historical archive → Parquet |
| `helpers/test_variant_parameters.py` | Every parameter class |
| `helpers/llk_params.py` | `PerfRunType` and the enums parameters carry |
| `helpers/metrics.py` | Efficiency metrics, the `*_pct` keys |
| `helpers/test_config.py` | `perf_run_tag`, `perf_run_dir`, the worker→core map |
| `perf_schema_derive.py` | Static column derivation with `ast` |
| `test_perf_header_gate.py` | Every schema and uniqueness gate |
| `test_perf_report_hw_free.py` | The real report code, no hardware |
| `test_perf_parquet.py` · `_migrate.py` · `_publish_run.py` | The storage layer |
| `compare_test_and_perf.py` | Audit a perf sweep against its functional counterpart |

Skills in `.claude/skills/`: `perf-regression-check` (one test, two commits),
`perf-report` (a full sweep with provenance), `perf-parameter-impact` (analyze a
`.post.csv`), `perf-optimization-audit`, `quasar-perf-test`.

Harness options: `--compile-producer` / `--compile-consumer`, `--speed-of-light`,
`--record-test-order`, `--test-order-file` with `--rewind-runner`, `PERF_KEEP_RUNS`. The
order-replay pair turns "is this the scheduler?" into a controlled experiment: replaying
one worker's sequence at `-n 1` removes core placement, ordering and concurrency at once.

`.github/CODEOWNERS` restricts `helpers/perf*`, `**/perf_*.py` and `**/test_perf_*.py`.
The schema is a contract with the data team, so it cannot drift through an unreviewed
edit.
