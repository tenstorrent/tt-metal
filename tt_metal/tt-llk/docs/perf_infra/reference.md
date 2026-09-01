# Reference

How each part works. Context for engineers and agents; not needed to complete a task
in [tasks.md](tasks.md).

| | Section |
|---|---|
| 1 | [The pipeline](#the-pipeline) |
| 2 | [The schema layers](#the-schema-layers) |
| 3 | [The published table](#the-published-table) |
| 4 | [The per-test catalog](#the-per-test-catalog) |
| 5 | [The gates](#the-gates) |
| 6 | [Parquet](#parquet) |
| 7 | [Run identity and output layout](#run-identity-and-output-layout) |
| 8 | [Historical migration](#historical-migration) |
| 9 | [The warehouse seam](#the-warehouse-seam) |
| 10 | [Where the perf gate stands](#where-the-perf-gate-stands) |
| 11 | [Tools](#tools) |
| 12 | [File map](#file-map) |

---

## The pipeline

```
perf test on silicon          PerfConfig(...).run(perf_report)
      |                       one row per sweep configuration
      v
PerfReport                    lazy frames, merged on `marker`, validated 1:1
      |                       ── enforces one schema per report
      v
per-test CSV                  <test>.csv (raw loop totals)
      |                       <test>.post.csv (per tile)
      |                       ── test_perf_header_gate, test_perf_report_hw_free
      v
perf_data/runs/<tag>/         one directory per invocation
      |                       `latest` symlink swapped atomically
      v
<tag>.parquet                 one typed batch, every test of the run, zstd
      |                       ── strict converter: raises before it writes
      v
CI artifact                   the run directory, per shard per architecture
      |
      v
warehouse                     PerfWarehouse seam — DuckDB now, Snowflake next
      |
      v
dashboard + PR gate           llk-perf-reporter reads the table; the gate is in design
```

The gates sit on the **transitions**, not the nodes. Each guards a boundary where a
schema could drift without anyone noticing.

## The schema layers

Four modules, four jobs. Each has exactly one owner.

| Module | Owns | Imports a device library? |
|---|---|---|
| `helpers/perf/schema.py` | Header names and header builders; the run-type and metric vocabularies | No |
| `helpers/perf/wide_schema.py` | `DB_SCHEMA` — the published WH/BH table | No |
| `helpers/perf/wide_schema_quasar.py` | The published Quasar table | No |
| `helpers/perf/test_schemas.py` | Per-test columns, versions, aliases | No |

None of them imports a device library. Every gate depends on that: it is what makes
the whole schema layer testable on a laptop.

Header builders, so no header is ever a literal:

```python
stat_column("L1_TO_L1", "mean")     # -> "mean(L1_TO_L1)"
metric_column("L1_TO_L1", "x")      # -> "L1_TO_L1_x"
text_size_column("L1_TO_L1")        # -> "TEXT_SIZE(L1_TO_L1)"
counter_base("FPU", "FPU_COUNTER")  # -> "FPU.FPU_COUNTER"
cycles_of("FPU.FPU_COUNTER")        # -> "FPU.FPU_COUNTER.cycles"
```

## The published table

`DB_SCHEMA` is 82 columns. One row is one test configuration in one run. It is both
the table handed to the data team and the exact schema the Parquet is written with.

Each column declares who fills it:

```python
Column("math_fidelity", "string", True, "configuration")          # origin="test"
Column("commit_sha", "string", False, "provenance", origin="ci")  # stamped by CI
```

Two views derive from that one list, so a column cannot exist in one and not the other:

| View | Columns | Meaning |
|---|--:|---|
| `OUTPUT_SCHEMA` | 75 | `origin="test"`. A report is validated against these. |
| `PROVENANCE` | 7 | `origin="ci"`. The publish layer stamps them. |

Provenance columns: `test_name`, `commit_sha`, `arch`, `run_id`, `timestamp`,
`pipeline` (`PR` | `nightly`), `pr_number` (NULL for nightly).

Row key: `test_name`, `commit_sha`, `arch`, `run_id`, plus the sweep-parameter columns.

Everything except `marker` and the mandatory provenance keys is **nullable**. That is
what lets 23 tests with different sweeps share one table: each fills its own columns
and the rest stay NULL. New columns join as nullable, so the table need not be
complete to be useful.

Quasar has its own table. Quasar columns never enter the WH/BH table, and WH/BH
columns never enter a Quasar batch.

Timing columns are formula-driven, not enumerated by hand: the full
`{mean, std} × _TIMING_BASES` grid, so the schema is a superset of any run
configuration rather than of what one nightly happened to sample.

Two deferred items are marked in the source:

| Deferred | Issue |
|---|---|
| Counter and metric columns join as nullable, once a counter run captures their exact names | #51249 |
| Canonicalizing columns that name the same thing differently (`c_dimm`/`k_dimm`/`r_dimm` vs `in0_c_dim`/`ct_dim`; `formats.input_A` vs `input_format`) | #51245 |

## The per-test catalog

One entry per perf test — 23 WH/BH, 18 Quasar:

```python
"perf_eltwise_binary": {
    "version": 4,
    "columns": [ ...14 sorted column names... ],
    "aliases": {"formats.sfpu_math": "formats.sfpu_src"},
    "test_name_aliases": {
        "perf_eltwise_binary": "perf_eltwise_binary",
        "perf_eltwise_binary_fpu": "perf_eltwise_binary",
    },
},
```

| Field | Job |
|---|---|
| `columns` | The reviewed column set. A change is a reviewed diff, never a surprise. |
| `version` | Two reports of the same test are comparable if the version matches. |
| `aliases` | Old column name → new. A reader can follow a rename across versions. |
| `test_name_aliases` | Old test name → current. Required, and must map the name to itself. |

The catalog is hand-maintained on purpose. A header change becomes a diff in a PR
instead of a surprise in the data.

`perf_schema_derive.py` re-derives each test's columns from source with `ast`.
Columns = the fixed format and flag headers, plus the parameter fields the test sets,
plus `marker`. Quasar tests derive from their sibling `test_*_quasar.py`, which is
where `PerfConfig` is called.

## The gates

Thirteen, all hardware-free. They run with the LLK pytest suites, under the default
`not perf and not nightly` marker filter.

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

Two derivation paths, and they catch different things:

| | Reads | Sees | Misses |
|---|---|---|---|
| `test_perf_header_gate.py` | Source, with `ast` | Every test, cheaply | Parameters built at run time |
| `test_perf_report_hw_free.py` | The real report code, device seams stubbed | Exactly what the code emits | Nothing, but it is slower |

`test_metric_bases_match_source` reads dictionary keys with `ast`, not a text scan, so
an unrelated `"_pct"` string in a log line can neither trip nor evade it.

## Parquet

Two entry points, one shared core:

```
live run   -> {test_name: DataFrame} ------\
                                            >-- build_run_batch -> to_table -> run.parquet
CSV files  -> convert_csvs_to_parquet ----/
```

`build_run_batch`:

1. **stamp provenance** — add `test_name` and the run context
2. **concat** — stack every test's rows into one frame
3. **to_table** — align to the schema (missing → NULL, extras dropped, order fixed),
   then cast each column to its declared Arrow type
4. **validate_batch** — every non-nullable column must be fully populated

The converter is **strict by default**: an unknown column, or a value that does not
fit its declared type, raises *before* anything is written. No lossy file reaches the
archive. Under `strict=False` it drops and coerces, and logs both — the coerced case
matters, because a NULLed value is indistinguishable downstream from a column a test
never emitted.

`TEXT_SIZE(...)` columns are in `DROPPED_COLUMNS`: the published table omits them on
purpose, so they are removed before the unknown-column check rather than flagged as
drift.

`parquet_to_csvs` reverses a batch into per-test CSVs. With the defaults, a round trip
reproduces the same batch — that is how the pipeline tests prove the conversion is
lossless.

## Run identity and output layout

```
perf_data/runs/<tag>/<test>/<test>.csv          raw loop totals
perf_data/runs/<tag>/<test>/<test>.post.csv     per-tile figures
perf_data/runs/<tag>/<tag>.parquet              one typed batch, all tests
perf_data/latest -> runs/<tag>                  symlink to the newest run
```

Two identifiers, easy to confuse:

| | `<tag>` (`PERF_RUN_TAG`) | `run_id` |
|---|---|---|
| Purpose | Directory and file name | `ROW_KEY` column in the table |
| Unique per | Invocation, and per CI shard | CI workflow — **shared by every shard** |
| Reaches the published table? | No | Yes |
| CI value | `<run_id>-<arch>-<shard index>` | `GITHUB_RUN_ID`, plus `-<attempt>` from attempt 2 |
| Local value | `local-<UTC timestamp>` | The tag |

CI sets the tag itself, because only the workflow can see the shard index. Naming
files after `run_id` would give all ten shards the same filename — fine while each
stays in its own directory, wrong the moment they are collected into one archive.

A re-run keeps `GITHUB_RUN_ID` and bumps `GITHUB_RUN_ATTEMPT`, so attempt 2 publishes
as `<run>-2`. Attempt 1 stays bare, so rows already archived keep the identity they
were published with.

Three properties of the output directory:

- **One directory per invocation.** Nothing is ever written twice. A shared mutable
  directory lets a narrower second run leave the first run's test directories in
  place, so the tree reads as complete while holding a blend of two runs.
- **The `latest` swap is atomic.** The link is created under a temporary name and
  renamed, so a reader sees the old run or the new one, never nothing.
- **Prune cannot delete the current run.** `PERF_KEEP_RUNS` (default 10) bounds local
  history. The run that just finished is protected by name, not by being newest — a
  clock that jumped backwards must not be able to remove it.

## Historical migration

`helpers/perf/migrate.py` converts an archive of past runs into the same schema, one
Parquet per run, reusing the same converter.

- **Deterministic.** Same archive → same batches, same report. No clocks; runs and
  CSVs process in sorted order.
- **Lenient.** A dirty run never aborts the migration. It is recorded as `failed` and
  skipped, and every other run still migrates.
- **Idempotent.** Each batch is written to a temporary file and renamed, so a run
  whose output exists is safely skipped and a crash mid-write leaves nothing a later
  pass mistakes for complete.

Provenance comes from an optional `run_meta.json` sidecar in the run directory.
Anything it omits falls back to the folder name, or to a default.

Only the raw per-test CSVs migrate. `.post.csv` and `.counters.csv` are excluded, for
the reason in [pitfalls](pitfalls.md#reading-testcsv-as-per-tile).

## The warehouse seam

```python
class PerfWarehouse:
    def load(self, parquet_path: str) -> int: ...   # ingest one run's batch
    def query(self, sql: str) -> pd.DataFrame: ...  # analytical read
```

`get_warehouse()` picks a backend from `PERF_WAREHOUSE`. Downstream code — publish,
compare, dashboard, gate — depends only on the interface.

| Backend | Role |
|---|---|
| `DuckDBWarehouse` | Local, Parquet-native stand-in, so the pipeline can be built and tested before the Snowflake table exists |
| `SnowflakeWarehouse` | The real target. `write_pandas` to load, `fetch_pandas` to query; the connector is imported lazily. |

Retiring the stand-in is three lines: delete `warehouse_duckdb.py`, drop the `duckdb`
branch in the factory, remove `duckdb` from requirements.

If the data team prefers a stage plus `COPY INTO`, or Snowpipe, only
`SnowflakeWarehouse.load` changes.

Status: PR #53021, open.

## Where the perf gate stands

A PR gate is designed but not shipped (#53752). The intended flow: a PR runs perf, the
gate compares against the baseline for the branch point, a regression blocks the
merge, and the post-merge sanity workflow publishes the new baseline.

What the measurement study established, on one card per architecture:

| Finding | Number |
|---|---|
| Most points are deterministic across runs | identical cycle counts |
| Ordinary noise is a fixed cycle offset, not a percentage | ~25 cycles (BH), 27 (WH) |
| `MATH_ISOLATE` and `UNPACK_ISOLATE` are gate-ready | 0 false failures / 396,142 |
| Non-matmul `L1_TO_L1` is stable | worst movement 1.59% |
| Repeating runs does not help | p99 essentially unchanged |
| Gate latency, `L1_TO_L1`, unsharded, one card | ~9 min (WH), ~6 min (BH) |
| Compile share of that cost | 65–79% |

Two defensible thresholds follow, and both **exclude matmul**:

- **2% slower AND more than 50 cycles slower.** Two clauses, because the fixed
  component is constant in cycles and the proportional one in percent, so no single
  number bounds both.
- **2% slower, applied to `TILE_LOOP` and `KERNEL` only.** Dropping `INIT`/`UNINIT`
  removes the small-number problem at the source, so no cycle clause is needed.

The blocker is matmul bistability —
[pitfalls](pitfalls.md#trusting-a-small-matmul-delta). It is a hardware or kernel
question: no threshold absorbs it, repeated runs do not average it out, and the
affected set is not a reproducible list. Every hypothesis the harness controls
(sweep configuration, a fixed bad set, a bad run, core placement, execution order,
concurrency, cold build) has been excluded. Only measurements that involve the packer
are affected, and the block-float pack path is immune where the standard
floating-point path is not.

Full study, with the measurements and the reproduction commands:
`nstojictt/llk-perf-noise-baseline`, `docs/perf_evaluation/README.md`.

## Tools

**In the repository**

| Tool | Purpose |
|---|---|
| `helpers/perf/` | `core`, `schema`, `wide_schema`, `wide_schema_quasar`, `parquet`, `migrate`, `publish_run`, `test_schemas` |
| `publish_run` CLI | One run's CSVs → one typed `run.parquet`, with CI provenance |
| `migrate` CLI | A historical archive → one Parquet per run |
| `perf_schema_derive.py` | Static column derivation with `ast`; no device imports |
| `compare_test_and_perf.py` | Audit a perf sweep against its functional counterpart |

**Developer tools** (`.claude/skills/`)

| Tool | Purpose |
|---|---|
| `perf-regression-check` | One test, any two commits, on your machine |
| `perf-report` | A full sweep end to end, with provenance |
| `perf-parameter-impact` | Analyze a finished `.post.csv` |
| `perf-optimization-audit` | Audit a kernel change for a suspect optimization |
| `quasar-perf-test` | Create or repair a Quasar perf test |

**Harness options**

| Option | Purpose |
|---|---|
| `--compile-producer` / `--compile-consumer` | The two-phase flow |
| `--record-test-order` | Record which test ran on which worker, in which order |
| `--test-order-file` + `--rewind-runner` | Replay that exact sequence |
| `--speed-of-light` | Fold runtime parameters into the template list |
| `PERF_KEEP_RUNS` | Run directories to retain locally (default 10) |

`--record-test-order` with `--rewind-runner` is what turns "is this the scheduler?"
into a controlled experiment: replaying one worker's sequence at `-n 1` removes core
placement, ordering and concurrency at once.

**Ownership.** `.github/CODEOWNERS` restricts `helpers/perf*`, `**/perf_*.py` and
`**/test_perf_*.py`. The schema is a contract with the data team, so it cannot drift
through an unreviewed edit.

## File map

| File | Holds |
|---|---|
| `helpers/perf/schema.py` | Header names, builders, run-type and metric vocabularies |
| `helpers/perf/wide_schema.py` | `DB_SCHEMA` — the published WH/BH table |
| `helpers/perf/wide_schema_quasar.py` | The published Quasar table |
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
| `test_perf_parquet.py` | Parquet round trip and typing |
| `test_perf_migrate.py` | Migration determinism and leniency |
| `test_perf_publish_run.py` | The publish CLI |
