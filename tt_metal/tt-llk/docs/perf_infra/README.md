# LLK perf infrastructure

Two documents. Read the one that matches your question.

| Document | Read it when |
|---|---|
| [architecture.md](architecture.md) | You want to know what the perf pipeline is, what it was before, and where it goes next. |
| [rules.md](rules.md) | You must change a perf test, add a parameter, rename a test, or fix a gate failure. |

## The 60-second version

1. A perf test writes a CSV. One row is one sweep configuration.
2. Every column name comes from one module. Nobody writes header strings by hand.
3. A catalog records the exact columns of every perf test, plus a version number.
4. A gate re-derives those columns from the source and fails the PR on any drift.
5. Each run writes to its own directory, `perf_data/runs/<tag>/`.
6. Each run also writes one typed Parquet file with the same schema for all tests.
7. CI uploads the run directory. The Parquet goes to the warehouse.
8. The dashboard and the future PR gate read the warehouse.

## The two rules that break a build

- **A new CSV column must exist in `helpers/perf/wide_schema.py`.** If it does not,
  the Parquet writer raises `PerfSchemaError` and the perf run fails.
- **A change to any perf test's columns must be recorded in
  `helpers/perf/test_schemas.py`, with the `version` increased.** If it is not,
  `test_perf_header_gate.py` fails.

[rules.md](rules.md) gives the full procedure for each case.
