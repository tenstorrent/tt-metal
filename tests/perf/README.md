# Runtime performance tests

One generic flow for every runtime perf benchmark: run the binary, compare each case with its golden, report,
and optionally update the golden. Nothing in `tests/perf` knows about a specific benchmark.

## How a run works

Running `pytest ... tests/perf/test_suites.py::test_perf[pgm_dispatch]`:

1. `test_suites.py` has one test per entry in `suites.yaml`; `registry.py` loads the `pgm_dispatch` entry: which
   binary to run, its golden, its repetitions and its pass/fail thresholds.
2. The `perf` fixture from `plugin.py` hands the suite to `session.py`, which drives the rest.
3. `runner.py` starts the benchmark binary and collects the JSON file it writes.
4. `contract.py` reads that JSON and reduces it to one value per case, e.g. the min of the repetitions.
5. `compare.py` checks each value against `golden.py`'s stored value for this environment: PASS, REGRESSION
   (too slow), STALE (too fast, the golden needs updating), or NEW, MISSING, ERROR.
6. Cases outside tolerance are run once more and only keep that status if the re-run agrees.
7. `session.py` writes `measurements.json`, and `report.py` prints the summary and writes it to the GitHub step
   summary.
8. With `--perf-update`, `update.py` rewrites the golden. CI never does this.
9. The test passes or fails based on the comparison.

The benchmark's only job is to follow the contract in `perf_contract.hpp`:

- **Declare what it measures.** At startup it calls `declare_metric`, e.g. `IterationTime` in seconds, lower is
  better, take the min, and names its swept arguments, e.g. `kernel_size`.
- **Write JSON.** Google Benchmark binaries already do; the declaration lands in the JSON header and each case is
  a row like `brisc_only_trace/kernel_size:256`. Other binaries write the same shape with `write_result()`.

## Pieces

| Piece | Where |
|---|---|
| Output contract | `perf_contract.hpp` (all binaries), `perf_contract_benchmark.hpp` (Google Benchmark binaries) |
| Suite registry | `suites.yaml`: binary, golden, repetitions, thresholds |
| Goldens | `goldens/<suite>.json` next to the benchmark source, one value per case per environment |
| Results | `generated/perf/<suite>/<environment>/`: raw output, logs, `measurements.json`, `summary.md` |

## The contract

A binary declares each gated metric once and reports it for every case:

```cpp
#include "perf/perf_contract_benchmark.hpp"
tt::perf::declare_metric("IterationTime", {"s", tt::perf::Better::Lower, tt::perf::Aggregate::Min});
```

Swept arguments are named (`->ArgName("kernel_size")`) so cases read `variant/kernel_size:256` and group
without extra configuration. Context that explains a perf shift, such as AICLK, is recorded with
`add_case_context` or `add_run_context` and shown next to the comparison. Binaries that are not Google Benchmark
executables call `tt::perf::write_result` with the path in `$TT_PERF_OUTPUT`.

## Running

```bash
python -m pytest --noconftest -p tests.perf.plugin "tests/perf/test_suites.py::test_perf[pgm_dispatch]" \
    --perf-environment=wh_n300_perf [--perf-filter=REGEX] [--perf-all]
```

`--perf-filter` replaces any `--benchmark_filter` in the suite's args, so it can select cases outside the suite.

A case fails when it is more than `regression_pct` worse (REGRESSION) or more than `improvement_pct` better
(STALE, the golden needs updating). Cases outside tolerance are re-run once and only fail if the re-run agrees.
When more than a quarter of the suite is outside tolerance the shift is systematic, so nothing is re-run.
NEW, MISSING and ERROR cases also fail. Suites with `enforce: false` report without failing on REGRESSION or
STALE until their noise is understood.

## Updating goldens

Updates never run in CI. From a CI run's artifacts, or from a local run on a matching machine:

```bash
python -m tests.perf update --from-run <github-run-id> [--suite NAME] [--force]
python -m pytest ... --perf-update [--force]
```

Only cases outside tolerance in both the run and its re-run are written, using the value nearer the old golden.
In-band cases are never rewritten, so small regressions cannot accumulate. Without `--force` the update writes
nothing if any case regressed, errored, is new, or (on an unfiltered run) is missing. With `--force` those are
accepted: regressions and new cases are written, missing cases are removed. Filtered runs never remove cases.

## Comparing two runs

```bash
python -m tests.perf report generated/perf/pgm_dispatch/wh_n300_perf/measurements.json [base.json] [--all]
```
