# Tasks

One procedure per task. Run every command from `tt_metal/tt-llk/tests/python_tests`.

Before you push:

```bash
python3 -m pytest test_perf_header_gate.py test_perf_report_hw_free.py \
                 test_perf_parquet.py test_perf_migrate.py test_perf_publish_run.py -q
```

| | Task |
|---|---|
| 1 | [A gate failed. Find the fix.](#1-a-gate-failed-find-the-fix) |
| 2 | [Add a sweep parameter to a perf test](#2-add-a-sweep-parameter-to-a-perf-test) |
| 3 | [Add a new parameter class](#3-add-a-new-parameter-class) |
| 4 | [Add a new perf test](#4-add-a-new-perf-test) |
| 5 | [Rename a perf test](#5-rename-a-perf-test) |
| 6 | [Delete or absorb a perf test](#6-delete-or-absorb-a-perf-test) |
| 7 | [Rename a column](#7-rename-a-column) |
| 8 | [Add a run type or an efficiency metric](#8-add-a-run-type-or-an-efficiency-metric) |
| 9 | [Produce a perf report](#9-produce-a-perf-report) |
| 10 | [Find out whether my branch regressed perf](#10-find-out-whether-my-branch-regressed-perf) |
| 11 | [Find out what regressed in nightly](#11-find-out-what-regressed-in-nightly) |
| 12 | [Read a run as a table](#12-read-a-run-as-a-table) |

---

## 1. A gate failed. Find the fix.

Match the assertion text. Every gate message names the file to edit.

| Message contains | Fix |
|---|---|
| `schema(s) drifted` … `+['x']` | The test emits `x` and the catalog does not list it. Add `x` to that test's `columns`, increase `version`. [§2](#2-add-a-sweep-parameter-to-a-perf-test) |
| `schema(s) drifted` … `-['x']` | The catalog lists `x` and the test no longer emits it. Remove it, add an `aliases` entry if renamed, increase `version`. [§7](#7-rename-a-column) |
| `found in source but missing from the catalog` | A new perf test has no catalog entry. [§4](#4-add-a-new-perf-test) |
| `in catalog but no longer a … perf test` | Renamed or deleted. [§5](#5-rename-a-perf-test) / [§6](#6-delete-or-absorb-a-perf-test) |
| `must include the current name` | Renamed in place, alias map not updated. [§5](#5-rename-a-perf-test) |
| `declared by more than one class` | Two parameter classes share a field name. Rename one. [§3](#3-add-a-new-parameter-class) |
| `collide with reserved headers` | A field name uses a reserved header. Rename it. [§3](#3-add-a-new-parameter-class) |
| `used more than once in a single test config` | One `templates=`/`runtimes=` list holds the same class twice. Use it once, or split into two classes. |
| `declared more than once with different fields` | A parameter class is defined twice; the later one shadows the earlier. Consolidate, or rename one. |
| `PerfRunType members … drifted` | Add the name to `RUN_TYPE_NAMES`. [§8](#8-add-a-run-type-or-an-efficiency-metric) |
| `Efficiency metric names drifted` | Add the name to `METRIC_BASES`. [§8](#8-add-a-run-type-or-an-efficiency-metric) |
| `columns not in schema, would be DROPPED` | Rule 1. Add the column to `wide_schema.py`. [§2](#2-add-a-sweep-parameter-to-a-perf-test) |
| `values that don't fit … would become NULL` | The declared type is wrong, or the test emits a bad value. Fix one of them. |
| `duplicate column header(s)` | Two columns share one header at run time. Rename one parameter field. |

---

## 2. Add a sweep parameter to a perf test

Example: the test must sweep `THROTTLE_LEVEL`.

**1. Find or write the parameter class.** Look in `helpers/test_variant_parameters.py`.
If nothing carries the value, write one — [§3](#3-add-a-new-parameter-class).

The dataclass field name **becomes the CSV column name**.
`THROTTLE_LEVEL.throttle_level` produces a `throttle_level` column.

**2. Pass it in the `PerfConfig` call** — `templates=` for a compile-time value,
`runtimes=` for a runtime value:

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

Keep the list a literal — see [pitfalls](pitfalls.md#a-dynamically-built-parameter-list-is-invisible-to-the-gate).

**3. Add the column to the published table** — `helpers/perf/wide_schema.py`:

```python
Column("throttle_level", "int64", True, "configuration"),
```

- `nullable` is `True`. Always. Other tests do not emit it.
- `dtype`: `int64` | `float64` | `bool` | `string`. An enum is `string`.
- `category`: `configuration` for a sweep parameter. Keep the block alphabetical.
- Quasar-only column → `wide_schema_quasar.py` instead. Never mix the two tables.

**4. Add the column to the catalog** — `helpers/perf/test_schemas.py`:

```python
"perf_eltwise_binary": {
    "version": 5,                    # was 4 — increase it
    "columns": [
        ...,
        "throttle_level",            # keep sorted
        ...,
    ],
    "aliases": {...},                # unchanged
    "test_name_aliases": {...},      # unchanged
},
```

**5. Run the gates.**

```bash
python3 -m pytest test_perf_header_gate.py test_perf_report_hw_free.py -q
```

---

## 3. Add a new parameter class

**1. Choose the base class.**

| Base | Value | Cost |
|---|---|---|
| `TemplateParameter` | Compile-time. Becomes a `constexpr` in the generated header. | Each value produces its own ELF. |
| `RuntimeParameter` | Runtime. Passed in a struct. | No extra builds. |

**2. Write it** in `helpers/test_variant_parameters.py`:

```python
@dataclass
class THROTTLE_LEVEL(TemplateParameter):
    throttle_level: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"
```

A `RuntimeParameter` also implements `convert_to_struct_fields`.

**3. Check the field name against all four naming rules.**

| Rule | Gate |
|---|---|
| No two parameter classes declare the same field name | `test_parameter_field_names_are_globally_unique` |
| The field name is not a reserved header | `test_no_parameter_field_equals_a_fixed_header` |
| The class is defined once, or identically | `test_no_shadowed_parameter_class` |
| One class appears at most once per `templates=`/`runtimes=` list | `test_no_param_type_used_twice_in_one_config` |

**Reserved header names.** Not usable as a field name:

```
formats.input_A   formats.input_B   formats.register_A   formats.register_B
formats.output    formats.sfpu_src  formats.sfpu_dst
unpack_to_dest    dest_acc          marker               test_name
```

`loop_factor` and `tile_cnt` are **not** reserved — they already are the fields of
`LOOP_FACTOR` and `TILE_COUNT`. Do not declare them again.

**A field with default `None` is optional.** It becomes a column only when a call
sets it. `MATH_OP.pool_type` and `MATH_OP.unary_extra` work this way.

---

## 4. Add a new perf test

**1. Name the file.**

| Arch | Path | Derived from |
|---|---|---|
| WH/BH | `perf_<stem>.py` at the `python_tests` root | itself |
| Quasar | `quasar/perf_<stem>_quasar.py` | the sibling `test_<stem>_quasar.py` |

The stem must match the functional test `test_<stem>.py`. The stem is also the
`test_name` value in the published table.

**2. Write the test.**

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

`@pytest.mark.perf` is required. Without it CI deselects the test, and the
developer tools cannot find its CSV.

**3. Get the column list.** Do not write it by hand:

```bash
python3 -c "import perf_schema_derive as d; print(d.derive_perf_test_schemas()['perf_my_op'])"
```

Pass `quasar=True` for a Quasar test. The gate's `+[...]` failure output works too.

**4. Add the catalog entry** — `helpers/perf/test_schemas.py`
(`PERF_TEST_SCHEMAS_QSR` for Quasar):

```python
"perf_my_op": {
    "version": 1,
    "columns": [ ...the derived list, sorted... ],
    "test_name_aliases": {"perf_my_op": "perf_my_op"},
},
```

`test_name_aliases` is required and must map the name to itself. `aliases` is
optional; add it only when a column is renamed later.

**5. Confirm every column is in `DB_SCHEMA`.** Add what is missing, as nullable.

**6. Run the gates.**

---

## 5. Rename a perf test

Three places. The third is the one people miss.

**1.** Rename the file: `perf_old_name.py` → `perf_new_name.py`.

**2.** Rename the catalog key in `helpers/perf/test_schemas.py`.

**3.** Update `test_name_aliases`:

```python
"perf_new_name": {
    "version": 4,                     # unchanged: the columns did not change
    "columns": [...],
    "test_name_aliases": {
        "perf_new_name": "perf_new_name",   # current name maps to itself
        "perf_old_name": "perf_new_name",   # every previous name maps forward
    },
},
```

What the gate enforces:

| Rule | Reason |
|---|---|
| The current name maps to itself | Catches a key renamed in place with a stale map |
| Every value equals the catalog key | An alias pointing at a third name is a typo |
| An old name is not also a catalog key | Aliases point at the surviving entry only |
| Two entries do not claim the same old name | The history would be ambiguous |

Also update any skill or document that names the test, and
`compare_test_and_perf.py` if the functional counterpart moved too.

See [pitfalls](pitfalls.md#renaming-a-test-starts-a-new-dashboard-series) for the
downstream cost.

---

## 6. Delete or absorb a perf test

**Deleted:**

1. Delete the test file.
2. Delete its catalog entry from `helpers/perf/test_schemas.py`.
3. Leave `DB_SCHEMA` alone. A column another test still emits must stay. A column
   nobody emits is harmless — it is nullable, and removing it breaks the published
   table.

**Absorbed into another test:** record the old name in the surviving test's
`test_name_aliases`, and update that test's `columns` and `version` if its sweep grew.

---

## 7. Rename a column

Example: `formats.sfpu_math` → `formats.sfpu_src`.

1. Rename the field, or the header builder, at its single owner.
2. Rename the `Column` entry in `wide_schema.py`.
3. For **every** affected test in `test_schemas.py`:
   - replace the old name in `columns`
   - add `"aliases": {"formats.sfpu_math": "formats.sfpu_src"}`
   - increase `version`
4. Run the gates.

`aliases` is what lets a reader join a v3 report to a v4 report. Without it the
history silently splits in two.

Do all four steps in one PR. A rename that reaches the tests before the catalog
broke `main` once (#52058).

---

## 8. Add a run type or an efficiency metric

### A run type

A run type prefixes every metric and counter column, so the vocabulary is gated globally.

1. Add the member to `PerfRunType` in `helpers/llk_params.py`.
2. Add the same name to `RUN_TYPE_NAMES` in `helpers/perf/schema.py`.
3. Append the base name to `_TIMING_BASES` in `wide_schema.py`. The `{mean, std}`
   column grid follows from it.
4. Handle the run type in the kernel. `PERF_RUN_TYPE` is a template parameter, so
   each run type compiles its own ELF.
5. Run the gates.

**Cost.** Cost scales with (variants × selected run types). Across 23 WH/BH perf
modules there are about 70 (module × run-type) pairs today. To scope a local sweep,
select fewer tests with `-k`. (`--perf-run-types` does the same by run type, but is
not yet on `main` — it lives on `nstojictt/perf-gate-budget`.)

### An efficiency metric

1. Add the `*_pct` key to the metric dictionary in `helpers/metrics.py`. Only keys
   ending in `_pct` are exported.
2. Add the same name to `METRIC_BASES` in `helpers/perf/schema.py`.
3. Run the gates.

Metric columns are **not** in `DB_SCHEMA` yet. They join the published table as
nullable once a counter run captures their exact names (#51249).

---

## 9. Produce a perf report

Two phases. Never one serial invocation.

```bash
pytest --compile-producer -n 10 -m "perf and not accuracy" <selection>  # compile, then skip
pytest --compile-consumer -n 15 -m "perf and not accuracy" <selection>  # load ELFs, measure
```

`tests/run_llk_perf_<arch>.sh` is the CI runner and does exactly this.

`-n 15` is a correctness constraint, not a speed choice — see
[pitfalls](pitfalls.md#-n-auto-addresses-cores-that-may-not-exist).

Output:

```
perf_data/runs/<tag>/<test>/<test>.csv          raw loop totals
perf_data/runs/<tag>/<test>/<test>.post.csv     per-tile figures
perf_data/runs/<tag>/<tag>.parquet              one typed batch, all tests
perf_data/latest -> runs/<tag>                  symlink to the newest run
```

`perf_data/` is gitignored. A report is a build artifact. What identifies it is the
test, the architecture, the commit and the exact command — record all four.
`PERF_KEEP_RUNS` bounds local history (default 10).

For a full sweep with provenance checks, the `perf-report` skill drives this.

---

## 10. Find out whether my branch regressed perf

**Locally, before you push** — one test, any two commits:

```bash
S=tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh

$S blackhole perf_math_matmul                       # HEAD vs the branch point
$S blackhole perf_math_matmul --baseline v0.60.0    # vs a tag
$S wormhole perf_math_matmul --baseline 1a2b3c4 --current 9f8e7d6
$S blackhole perf_math_matmul --dry-run             # resolve refs, measure nothing
```

In Claude Code: `/perf-regression-check perf_math_matmul`.

| Option | Default | Meaning |
|---|---|---|
| `--baseline <ref>` | `merge-base origin/main HEAD` | What to compare against |
| `--current <ref>` | `HEAD` | What to judge |
| `--iterations <N>` | `3` | Sweeps per side, median vs median |
| `--threshold <T>` | `0.05` | Flagged in both directions |
| `--speed-of-light` | off | Applied to both sides |
| `--refresh` | off | Ignore cached runs |

The script uses a sparse worktree per commit, so your checkout is never touched and
a dirty tree is fine. Only **committed** code is measured. Runs are cached per
(arch, test, variant, commit), so raising `--iterations` after a run only measures
the difference. Iterations interleave, so machine drift hits both sides equally.

Comparison is per (marker, sweep config) on `mean(<run_type>)` cycles, median across
iterations. A config with no baseline is reported as a "new point", never as a
regression.

**On CI** — use the dashboard's **Branch** tab: enter your GitHub username, pick
your branch's perf run, **Compare vs main**. It judges your run against recent
`main` nightlies with the same engine as the nightly scan.

| Symptom | Action |
|---|---|
| `TENSIX TIMED OUT`, device hang | `tt-smi -r`, re-run the same command. Completed iterations are cached. |
| Missing header at compile time on an older commit | Re-run with `SPARSE_PATHS=` (empty) for a full checkout |
| `no perf CSV` | Check the module name, and that the test carries the `perf` mark |
| `does not exist at <sha>` | The test postdates that commit. Pick a later baseline. |

---

## 11. Find out what regressed in nightly

Use the LLK Perf Reporter dashboard.

| Your question | Tab | Path |
|---|---|---|
| What regressed last night? | **Scan** | Pick the architecture → **Scan** |
| What changed since the last release? | **Scan** | **Load runs to pin…** → set Baseline → **Scan** |
| Which commit caused it? | **Scan** / **Trends** | Click the row → **View commits in this step** |
| How did this number move over time? | **Trends** | **Load latest run's tests** → pick test + metric → **Plot trend** |
| Unpack-, math-, or pack-bound? | **Stage breakdown** | Plot in Trends → **Stage breakdown** |
| What does this parameter cost? | **Param impact** | Pick test, marker, metric, parameter |

Scan flags a point when it deviates beyond *k* sigma from a median baseline of recent
nightlies, where sigma is a robust MAD floored at 0.3% of the baseline. Default `k` is
3. `sustained` means the shift held for two or more runs. Ranking puts **one-sided**
groups first, because balanced movement is noise.

Always read the coverage line — see
[pitfalls](pitfalls.md#lost-coverage-looks-exactly-like-an-improvement).

---

## 12. Read a run as a table

**One run's CSVs → one typed Parquet:**

```bash
python3 -m helpers.perf.publish_run --csv-dir perf_data/latest --out run.parquet --arch wormhole
```

Point `--csv-dir` at **one** run — `perf_data/latest` or a specific `runs/<tag>` —
never at `perf_data` itself, or every retained run is swept into one batch.
`--strict` fails on schema drift instead of dropping and coercing.

Requires `COMMIT_SHA`, `RUN_ID` and `PIPELINE` (`PR` | `nightly`) in the environment.

**A historical archive → one Parquet per run:** `helpers/perf/migrate.py`.

**Parquet → per-test CSVs:** `parquet.parquet_to_csvs(path, out_dir)`.

**Query history:** the `PerfWarehouse` seam — `load(parquet_path)` and `query(sql)`,
selected by `PERF_WAREHOUSE`. See [reference.md](reference.md#the-warehouse-seam).
