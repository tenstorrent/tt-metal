# Generic operation → C++ ProgramDescriptor-factory migration

This tt-metal-owned tool freezes a recorded eval run, reproduces its original
baseline, and validates an authored native port. All orchestration, mapping,
review guidance and synthetic tests are versioned together here, on one
tt-metal branch. The operation being migrated is a separate change.

Start with [PORT_FLOW.md](PORT_FLOW.md) for the complete flow:

1. [Export exact source and evidence](EXPORT_RUN.md) from a read-only DB snapshot.
2. [Reproduce the recorded baseline](MIGRATION_WORKFLOW.md) in an isolated checkout.
3. [Map the operation](MAPPING.md) and author its native
   [ProgramDescriptor factory](FACTORY_CONTRACT.md) and cache tests.
4. [Validate source/native parity and cache behavior](PORT_FLOW.md), then complete
   the [independent review gate](REVIEW.md).

Run entry points from this tt-metal repository root, for example:

```bash
python3 -m tools.generic_op_to_factory.export_run --help
python3 -m tools.generic_op_to_factory.migration_workflow --help
python3 -m tools.generic_op_to_factory.validate_port --help
```

The orchestration modules use the Python standard library. Live PostgreSQL
export additionally needs `psycopg2`; offline verification does not. Runtime
builds and golden tests need the selected checkout's own built environment.

The evaluator repository is still a historical **input**: its recorded Git
revision supplies golden suites, helpers and pytest plugins. It need not contain
any of these migration tools, and no evaluator branch or submodule-pin change
is part of installing the tool. Separate isolated checkouts reproduce source
and target revisions; they do not split ownership of the migration flow.

JUnit parsing is owned locally in `classify_failures.py`, initially copied from
`tt_ops_code_gen` revision `034527ad845a7b61596139802c4af567983a14bb`.
The synthetic SQLite schema fixture comes from that revision's `eval/db.py`.
Neither imports the evaluator's DB initializer nor changes its recorded runtime
plugins. Review these compatibility snapshots when the DB/result format changes.

CPU-only tool tests (activate this checkout's existing `python_env` first):

```bash
PYTHONPATH="$PWD" ./scripts/run_safe_pytest.sh --run-all --no-precompile \
  tools/generic_op_to_factory/tests -q -o addopts=--import-mode=importlib
```

These tests use synthetic IDs, temporary Git repositories, fake build/test
runners and small C++20 contract fixtures compiled when a host compiler is
available. They do not measure native-operation performance or replace real
build, golden, cache and review evidence for each port.

Missing shared headers require deterministic recovery or an explicitly approved
[dependency substitution](DEPENDENCY_SUBSTITUTIONS.md). The latter is recorded
as a changed baseline, never as exact historical runtime reproduction.
