# Common pitfalls

Mistakes that either pass every gate, or fail far from their cause.

| | Pitfall | Cost |
|---|---|---|
| 1 | [Changing `columns` without increasing `version`](#changing-columns-without-increasing-version) | Silent. History becomes incomparable. |
| 2 | [A dynamically built parameter list is invisible to the gate](#a-dynamically-built-parameter-list-is-invisible-to-the-gate) | Silent until the Parquet writer raises |
| 3 | [`-n auto` addresses cores that may not exist](#-n-auto-addresses-cores-that-may-not-exist) | Wrong numbers, not slow ones |
| 4 | [`-x` on a report run](#-x-on-a-report-run) | A partial CSV that reads as missing data |
| 5 | [Narrowing the sweep in the test file](#narrowing-the-sweep-in-the-test-file) | An incomparable report |
| 6 | [Reading `<test>.csv` as per-tile](#reading-testcsv-as-per-tile) | Off by `loop_factor * tile_cnt` |
| 7 | [Trusting a small matmul delta](#trusting-a-small-matmul-delta) | Chasing a hardware state |
| 8 | [Reading an isolate as a share of `L1_TO_L1`](#reading-an-isolate-as-a-share-of-l1_to_l1) | Different binaries, not components |
| 9 | [Reading `.cycles` as time spent in that counter](#reading-cycles-as-time-spent-in-that-counter) | It is the bank's elapsed zone time |
| 10 | [Expecting `std(...)` to be there](#expecting-std-to-be-there) | One execution per variant by default |
| 11 | [Renaming a test starts a new dashboard series](#renaming-a-test-starts-a-new-dashboard-series) | A discontinuity nobody flagged |
| 12 | [Lost coverage looks exactly like an improvement](#lost-coverage-looks-exactly-like-an-improvement) | A regression read as a win |
| 13 | [Comparing against latest `main`](#comparing-against-latest-main) | Somebody else's change in your delta |
| 14 | [Naming a parameter field `marker`](#naming-a-parameter-field-marker) | Breaks marker processing, evades the gate |
| 15 | [Using one parameter class twice in a config](#using-one-parameter-class-twice-in-a-config) | Duplicate columns |
| 16 | [Pointing `publish_run` at `perf_data`](#pointing-publish_run-at-perf_data) | Every retained run in one batch |

---

## Changing `columns` without increasing `version`

The gate compares the column **set**. Update `columns` and leave `version` alone, and
every gate passes.

Nothing else notices either. `version` is the only signal a reader has that two
reports of the same test measure the same thing. Two v4 reports with different
columns are silently incomparable, and the damage shows up weeks later in a trend
that has no explanation.

**Always increase `version` when `columns` changes.** It costs one character.

## A dynamically built parameter list is invisible to the gate

`test_perf_header_gate.py` reads `templates=` and `runtimes=` with `ast`. It handles
three shapes:

```python
PerfConfig(..., templates=[MATH_FIDELITY(f)])       # inline
templates = [MATH_FIDELITY(f)]; PerfConfig(**cfg)   # a named variable
cfg = {"templates": [MATH_FIDELITY(f)]}             # a dict entry
```

It cannot see a list built by a comprehension or returned by a helper. Such a test
under-reports its columns to the gate, so the gate passes — and the run then fails
at the Parquet writer, because the real column is not in `DB_SCHEMA`.

**Keep the lists literal.** If you cannot, `test_perf_report_hw_free.py` is the exact
check: it runs the real report code with the device seams stubbed.

## `-n auto` addresses cores that may not exist

The xdist worker index maps onto a physical Tensix core:

```python
row, col = divmod(int(worker_id[2:]), 8)   # helpers/test_config.py
```

`gw0` is core (0,0); `gw8` is (1,0). `-n 15` is a promise that only cores (0,0) to
(1,6) are used. `-n auto` addresses an 8×8 grid, and Wormhole ships with harvested
rows.

**`-n auto` is wrong, not merely slower.** Compile parallelises freely; measurement
does not.

## `-x` on a report run

The CI runner passes `-x` on the consumer phase. That is right for CI — fail fast.

It is wrong for a report. `-x` aborts mid-sweep, and the combined CSV is then
silently partial. Downstream that reads as measurements that went missing, not as a
run that stopped early.

**Drop `-x` when the report is for analysis.**

## Narrowing the sweep in the test file

Editing `@parametrize` to make a run finish produces a report that cannot be compared
with any other report of that test, and the edit tends to survive into a commit.

**Narrow the selection with `-k` instead, and say so in the report.**

## Reading `<test>.csv` as per-tile

`<test>.csv` holds **raw loop totals** for `TILE_LOOP`. The per-tile figure exists
only in `<test>.post.csv`.

The Parquet stores the raw form on purpose, because per-tile is derivable and a
second copy would not be: divide `mean(<RUN_TYPE>)` by `loop_factor * tile_cnt`. Both
are columns.

An absolute cycle threshold derived from `.post.csv` does not transfer to the raw
table. A relative threshold does.

## Trusting a small matmul delta

`perf_matmul` and `perf_math_matmul` measurements are **bistable**: a given
configuration lands on one of two discrete values, 2 to 6% apart on `L1_TO_L1` and up
to 24% on `PACK_ISOLATE`, at low probability per run.

This is a state the hardware enters, not measurement error. Repeating runs does not
average it out, and the affected set is not a fixed list — any matmul configuration
can do it on any run.

**A 2 to 6% step on a matmul test is not evidence of a regression.** Every other test
in the sweep is stable to well under 2%.

Only measurements that involve the packer are affected. `MATH_ISOLATE` and
`UNPACK_ISOLATE` are clean. See
[reference.md](reference.md#where-the-perf-gate-stands).

## Reading an isolate as a share of `L1_TO_L1`

Each run type is a **separate binary**. `PERF_RUN_TYPE` is a template parameter, and
the kernel selects a different code path with `if constexpr`.

In `MATH_ISOLATE` the other two threads are not doing real work. The three isolates
do not sum to `L1_TO_L1`, and their ratio to it means nothing.

Use **Stage breakdown** in the dashboard for the bound-by question, which is what
that comparison is usually reaching for.

## Reading `.cycles` as time spent in that counter

A `<BANK>.<COUNTER>.cycles` column is that **bank's total elapsed zone time**, not
cycles attributable to the counter.

This has cost time before. It looks like a per-counter figure and is not one.

## Expecting `std(...)` to be there

`PerfConfig.run()` defaults to `run_count=1`, so almost every value in a report is a
**single device execution**, and the all-NaN `std(...)` columns are dropped.

Two consequences:

- A number you compare carries the full single-shot jitter.
- Raising `run_count` averages on device inside one variant, at no extra compile and
  no extra pytest run. It is the cheapest noise lever available.

## Renaming a test starts a new dashboard series

`test_name` in the published Parquet is the module stem, and it is **not** remapped in
the repository. Any external dashboard keyed on `test_name` sees a renamed test as a
new series.

Per-config trends survive in the table — this is a discontinuity, not data loss. But
nobody is told about it.

**`test_name_aliases` is the old-to-new map a reader needs to stitch the two series.**
Keep it complete, and mention the rename in the PR body so dashboard owners can
remap.

## Lost coverage looks exactly like an improvement

A configuration that stopped running produces the same shape in a diff as a
configuration that got faster: it is present in the baseline and absent, or better,
in the candidate.

The dashboard's Scan reports a **coverage** line above the results: configs and tests
that stopped or started running between the two runs.

**Read the coverage line before you read the regressions.**

## Comparing against latest `main`

`perf_compare_commits.sh` defaults its baseline to `git merge-base origin/main HEAD`
— the commit you branched from, not the tip of `main`.

That default is deliberate. The further apart the two commits, the more unrelated
change lands in the delta. For "did *this* work regress perf", the branch point is
the honest baseline.

Also: **only committed code is measured.** Uncommitted edits are invisible to a ref.

## Naming a parameter field `marker`

`marker` is reserved, and for a specific reason. The pipeline merges run types on
`marker`. A parameter named `marker` would be renamed to `marker_x` / `marker_y` by
that merge — so it would evade the duplicate-column gate and break marker processing
in silence.

The full reserved list is in [tasks.md §3](tasks.md#3-add-a-new-parameter-class).
`loop_factor` and `tile_cnt` are **not** reserved: they already are the fields of
`LOOP_FACTOR` and `TILE_COUNT`, so do not declare them a second time.

## Using one parameter class twice in a config

```python
runtimes=[TILE_COUNT(4), TILE_COUNT(8)]   # two columns, one header
```

Each use emits the same CSV header. The field name is unique across classes, so the
uniqueness gate does not fire — `test_no_param_type_used_twice_in_one_config` is the
one that catches this.

**Model two of the same quantity as two classes with distinct field names**, for
example `INPUT_TILE_CNT` and `OUTPUT_TILE_CNT`.

## Pointing `publish_run` at `perf_data`

`--csv-dir` globs recursively. Point it at `perf_data` and every retained run is
swept into one batch, under one run's provenance.

**Point it at one run**: `perf_data/latest`, or a specific `runs/<tag>`.
