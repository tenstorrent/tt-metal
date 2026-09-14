# Optional historical-runtime replay

`tools.generic_op_to_factory.migration_workflow` is the historical-runtime
diagnostic, not a prerequisite of the default [two-golden migration flow](PORT_FLOW.md).
Use it when target-source execution cannot run, differs from recorded DB outcomes,
or historical reproduction is explicitly requested. It consumes a
[verified frozen export](EXPORT_RUN.md); it does not ask an LLM to reconstruct
source, query the live DB, or contain run/operation-specific recipes.

The implemented flow is:

```text
frozen export + recorded Git revisions
  → prepare source/dependencies
  → isolated recorded-runtime checkout + pinned evaluator
  → install exported operation → environment → build
  → collect → optional smoke → full golden suite
  → compare case identities and statuses with the selected DB phase
```

This is **not yet the complete migration**. Running on the target revision,
translating to C++, numerical equivalence checks, and program-cache validation
are explicit `not_implemented` gates. `migration_ready` always stays false.
The target revision is pinned now so that a later gate has an explicit input;
this driver does not check it out or modify the target branch.

The separate `tools.generic_op_to_factory.prepare_target` and `tools.generic_op_to_factory.validate_port` entry points now
cover target preparation and validation of an authored C++ port. Run
`python3 -m tools.generic_op_to_factory.validate_port --help` for its checkpointed entry point and see
[PORT_FLOW.md](PORT_FLOW.md) for
configuration and translation requirements. The recorded-baseline driver's
future-gate flags remain unchanged: they do not certify those separate runs.

## Configuration

An optional `dependency_substitutions` list supports user-approved, DB-backed
additions of missing canonical headers. See
[the substitution contract](DEPENDENCY_SUBSTITUTIONS.md). The default is empty;
using it changes the recorded baseline scope and does not relax outcome gates.

Run these commands from the tt-metal repository root containing this tool.
`eval_repository` supplies recorded Git objects only; it needs no migration-tool changes.
Supply absolute paths and full, locally available Git commit IDs. The workspace
must be new and disjoint from the repositories and export. Place tt-metal
workspaces under `/localdev/astancov` in this environment.

Example `workflow.json` (replace the paths and commit placeholder):

```json
{
  "export": "/localdev/astancov/frozen-export",
  "metal_repository": "/localdev/astancov/tt-metal-genop-v2",
  "eval_repository": "/localdev/astancov/tt_ops_code_gen",
  "workspace": "/localdev/astancov/migration-workspace",
  "target_revision": "<full 40-character target commit>",
  "phase": "initial",
  "precompile": false,
  "capture_metrics": false,
  "build_argv": ["./build_metal.sh", "--enable-ccache"]
}
```

All fields above are required. Choose `phase` from the exported results; use
`null` only for an unphased run. Empty phases and ambiguous case identities are
rejected. Choose precompile and profiler settings deliberately: the historical
DB may not contain enough information to recover the original runtime settings.
See [baseline provenance limitations](PREPARE_BASELINE.md).

Optional fields:

| Field | Meaning |
| --- | --- |
| `smoke_nodeid` | One explicit pytest node ID within the recorded golden suite. Default: no smoke test; the full suite is still required. |
| `precompile_workers` | Positive worker count when precompile is enabled; default `8`. |
| `command_timeout_seconds` | Positive timeout for each launched command; default no timeout. This does not replace safe-runner hang detection. |
| `environment` | Explicit string values for the allowlisted build/cache variables below. Default `{}`. |
| `configure_argv` | Optional CMake configuration argument array, executed before the build. Default `[]`. |
| `sfpi_directory` | Optional existing SFPI directory; requires explicit `configure_argv`. Default `null`. |
| `evaluator_path` | Nested evaluator location below `tt_metal/third_party/`; default `tt_metal/third_party/tt_ops_code_gen`. |

Allowed environment keys are `CMAKE_BUILD_PARALLEL_LEVEL`, `CPM_SOURCE_CACHE`,
`CMAKE_PREFIX_PATH`, `CCACHE_DIR`, `CCACHE_BASEDIR`, `CCACHE_CONFIGPATH`, and
`LD_LIBRARY_PATH`. Ambient DB, pytest, TT runtime and profiler variables are not
forwarded. Basic host variables such as `PATH`, `HOME`, and `SSH_AUTH_SOCK` are
inherited; this is environment hygiene, **not a security sandbox**. Only run
trusted source and reviewed configuration. Do not put credentials in argument
arrays or environment values; those explicit values are saved as evidence, and
subprocess logs can contain anything emitted by source/build scripts.

Argument arrays support literal `{runtime}` and `{workspace}` substitution,
without shell interpolation. `build_argv` must start with `./build_metal.sh` or
`.github/scripts/copilot-build.sh`. Clean, configure-only, and relocated builds
are not supported. The driver creates the environment using `./create_venv.sh`
and activates `./python_env/bin/activate` for build and test commands. It never
runs `install_dependencies.sh` or installs a host compiler. The CI wrapper still
requires its own Docker/cache prerequisites; artifacts must be importable from
the isolated runtime for the build checkpoint to complete.

If an old revision needs an existing compatible SFPI, configure its absolute
directory explicitly and supply the matching CMake configuration. The driver
hashes `compiler/bin/riscv-tt-elf-g++` and links the existing directory at
`{runtime}/runtime/sfpi` only if that location is absent or already matches.
This is not a full toolchain-distribution attestation. Configuration arguments
are trusted inputs, not a sandbox for arbitrary CMake behavior. A different
revision/layout may need a reviewed adapter; the driver does not automatically
patch sources or disable version checks to make a build pass.

## Plan, initialize, and run

```bash
# Read-only validation: no checkout, build, DB access, or source execution.
python3 -m tools.generic_op_to_factory.migration_workflow plan --config workflow.json

# Creates only the new workspace and its immutable plan / initial state.
python3 -m tools.generic_op_to_factory.migration_workflow init --config workflow.json

# Stops before collection or device tests; runs setup/build/import verification.
python3 -m tools.generic_op_to_factory.migration_workflow run \
  --workspace /localdev/astancov/migration-workspace --through build

# Read-only checkpoint summary; this does not revalidate artifacts.
python3 -m tools.generic_op_to_factory.migration_workflow status \
  --workspace /localdev/astancov/migration-workspace

# Continues through the full golden suite and comparison. Requires hardware.
python3 -m tools.generic_op_to_factory.migration_workflow run \
  --workspace /localdev/astancov/migration-workspace
```

`--through` accepts `prepare`, `checkout`, `evaluator`, `install`, `environment`,
`build`, `collect`, `smoke`, `baseline`, or `compare`. Default: `compare`.
Collection imports suite code and should not be treated as a sandboxed static
check. Every pytest invocation uses `./scripts/run_safe_pytest.sh`; the full
baseline has no case filters. Collection and smoke explicitly disable
precompile; the baseline uses the configured mode. A smoke case must pass to
continue, even if the historical full suite contains expected failures.

Checkout creates a detached Git worktree at the recorded `starting_commit`,
then runs `git submodule update --init --recursive`. The evaluator is selected
at the separately recorded `eval_commit`, which can differ from the parent
gitlink; the resulting submodule state is recorded. Worktree registration and
submodule initialization do update Git metadata, but the source repositories'
checked-out branches/files are not switched. No push, rebase, reset, or
automatic cleanup is performed. The operation installer refuses to overwrite
an existing operation directory.

## Checkpoints and retries

Each stage has a status and append-only attempt directories. Successful stages
are skipped on resume, after validating the plan, frozen export, implementation
hashes, source revisions, selected dependencies, submodule state, receipts,
logs and recorded build outputs. After a completed build, a live import probe
also checks the loaded runtime paths and Python package names/versions. A
workspace lock prevents concurrent driver runs against the same workspace;
it does not reserve the accelerator against unrelated processes.

On failure or interruption, inspect the stage's command and log files before
explicitly retrying:

```bash
python3 -m tools.generic_op_to_factory.migration_workflow run \
  --workspace /localdev/astancov/migration-workspace --retry
```

Retry creates a new attempt and retains previous evidence. It refuses to start
if an unfinished recorded command may still be alive or its process identity
was not saved; it never kills an old or unrelated process. An interrupted
command launched by this invocation receives
termination signals only through its own process group. Inspect safe-runner
device-health output before another device run; there is no automatic reset or
repair here.

Retry is conservative, not rollback: partial preparations, installed operation
directories or Python environments may require a **new workspace** rather than
overwriting them. A mismatched comparison remains blocked; `--retry` repeats
that comparison against the same completed baseline, not a fresh device run.
Use a new workspace to change configuration/source/toolchain/driver version or
repeat a completed baseline. Do not edit receipts to force progress.

The principal artifacts are:

```text
workspace/
  plan.json                   inputs, implementation hashes, selected revisions
  state.json                  checkpoints and future gates
  prepared/                   verified deterministic source/dependency package
  runtime/                    isolated tt-metal checkout and Python environment
  device-cache/               workspace-scoped kernel cache
  attempts/<stage>/<attempt>/ command metadata, combined logs, stage receipt
                             build: runtime.json
                             smoke/baseline: junit.xml and emitted sidecars
                             compare: comparison.json (also kept on mismatch)
```

A complete comparison means the same case identities and statuses as the
selected DB phase, including historical failures. It does **not** mean all
tests passed, failure messages match, numerical metrics match, performance is
unchanged, the historical hardware/environment was reproduced exactly, or the
operation has been migrated to C++. Those are separate claims and later gates.

## Verification

The workflow tests use temporary synthetic repositories, a synthetic frozen DB
export, fake build artifacts and fake safe-runner output. They exercise
orchestration, checkpoint reuse, drift detection, explicit configuration,
interruption, and comparison failures without DB credentials or an accelerator.
They contain no production run IDs or operation-specific test recipes.

```bash
./scripts/run_safe_pytest.sh --run-all --no-precompile tools/generic_op_to_factory/tests/test_export_run.py \
  tools/generic_op_to_factory/tests/test_prepare_baseline.py tools/generic_op_to_factory/tests/test_compare_baseline.py \
  tools/generic_op_to_factory/tests/test_migration_workflow.py -q
```

These tests do not validate real hardware execution, compiler integration, or
the CI wrapper. A supervised real-runtime run is still needed before treating
the new orchestration itself as device-validated.
