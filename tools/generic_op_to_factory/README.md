# Generic operation → C++ ProgramDescriptor-factory migration

This tt-metal-owned tool post-processes a **complete evaluated run branch** and
validates an authored native port with two golden suites on one target build.
The branch's final commit, not the DB's `starting_commit`, supplies the operation,
supporting tt-metal changes and evaluator gitlink. Historical reconstruction is
an explicit legacy diagnostic, not a prerequisite. All orchestration, mapping,
review guidance and synthetic tests are versioned together here, on one
tt-metal branch. The operation being migrated is a separate change.

Start with [PREPARE_EVALUATED_BRANCH.md](PREPARE_EVALUATED_BRANCH.md) for the
branch-based command and validation config, then [PORT_FLOW.md](PORT_FLOW.md)
for the validation/review contracts.

Use [COMPARISON_GATE.md](COMPARISON_GATE.md) for the detailed migration-acceptance
protocol: host contract/planner comparison, cache transitions, paired host/device
profiling, and explicit performance budgets. It distinguishes current automated
checks from the additional measurement gates that still need tooling.

The private local `FLOW.html`, if present, is an older visualization, not the
branch-input specification or a live status dashboard.

1. Checkpoint the **final evaluated tree**, including supporting changes and submodule pins.
2. [Prepare an isolated worktree from the supplied branch](PREPARE_EVALUATED_BRANCH.md).
3. [Map the operation](MAPPING.md) and author its native
   [ProgramDescriptor factory](FACTORY_CONTRACT.md) and cache tests.
4. [Validate source/native parity and cache behavior](PORT_FLOW.md), then complete
   the [independent review gate](REVIEW.md).

Run entry points from this tt-metal repository root, for example:

```bash
python3 -m tools.generic_op_to_factory.export_run --help
python3 -m tools.generic_op_to_factory.prepare_branch --help
python3 -m tools.generic_op_to_factory.validate_port --help
```

The orchestration modules use the Python standard library. Live PostgreSQL
export additionally needs `psycopg2`; offline verification does not. Runtime
builds and golden tests need the selected checkout's own built environment.

The evaluated branch's **own evaluator/gitlinks** supply golden suites, helpers
and pytest plugins. Preparation never overwrites them with the DB's evaluator
revision, installs exported operation files, or reconstructs missing APIs.
Source and native golden suites share the final target worktree and build.
DB exports remain optional provenance/results artifacts; branch validation does
not read them or claim to reproduce their outcomes. The older
[historical-replay diagnostic](MIGRATION_WORKFLOW.md) remains separately available.

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

An incomplete evaluated branch must be checkpointed/recovered upstream rather
than silently repaired from DB exports. [Dependency substitutions](DEPENDENCY_SUBSTITUTIONS.md)
belong to the legacy reconstruction path, not branch-based post-processing.
