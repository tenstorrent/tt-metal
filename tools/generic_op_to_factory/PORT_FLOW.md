# From a completed evaluated branch to a validated C++ port

## Default input: the final evaluated branch

Migration is a post-processing step of evaluation. Pass the branch that contains
the **whole evaluated operation**, including supporting tt-metal APIs, helpers
and evaluator changes. Use [PREPARE_EVALUATED_BRANCH.md](PREPARE_EVALUATED_BRANCH.md)
and `prepare_branch --branch BRANCH`; it pins the final commit and initializes
its recursive gitlinks in one fresh worktree. Dirty run changes must first be
checkpointed by the caller. The tool does not commit, reset, rebase or push them.

In `validate_port`, use `evaluated_branch` and `migration_paths`
instead of `preparation`, `export`, `phase` and
`allow_recorded_failures`. These are mutually exclusive input modes. The source
baseline comes from running the unchanged operation and golden suite already
on that branch; DB retrieval and a historical checkout are not involved.

The validation sequence is **one parity-enabled correctness build → factory contract → source golden →
source-baseline check → native golden → source/native comparison → native acceptance tests
→ independent review → scoped receipt**. Smokes and precompile remain optional.
`source_compare` records the source baseline; it does not require a green source
suite or claim a DB comparison. Ordinary source failures/errors proceed to native
execution. `native_compare` enforces the same case set and exact outcome classes;
missing, added or changed outcomes block completion. Matching failures remain
failures. The obsolete `allow_source_failures` key is rejected.
Reusing an existing eval report instead of running the source baseline once is
not implemented: evidence-to-checkpoint/environment binding needs its own gate.

The factory, routing, evidence and review contracts below apply to both modes.
The old preparation commands/config below are retained only for historical
reconstruction, **not the starting point for new post-evaluation migrations**.

## Native acceptance tests: more than cache tests

The pre-migration skill authors ordinary Python pytest files and exercises them
on the unchanged source before checkpointing. They encode operation-specific
API, validation, output metadata, memory/alias and illegal-write, numerical, branch-boundary and
cache requirements. Keep source-only planner inspection separate.

Configure a nonempty, unique `acceptance_tests` list of repository-relative `.py`
files below `tests/`. During migration, the `acceptance` stage runs that list
**once, on C++ only**, after golden parity. It is not a second source run of the
skill's tests. Its route adapter supplies `TT_PRE_MIGRATION_MODE=native` and
`TT_PRE_MIGRATION_ENTRY=native_entry` inside the pytest process before collection.
Suites must resolve that entry at setup, use the same assertions on both routes,
and never fall back to Python. The driver observes direct native calls, blocks
generic-op fallback during those calls, and requires each selected file to
collect tests. Setup/readback helpers are not covered by that fallback guard.
Pre-imported generic aliases in the operation path still require suite guards
and review; call counting alone cannot establish assertion quality or full coverage.

All selected acceptance tests must pass; failure/error/skip/xfail blocks the gate.
This does not require all original golden tests to pass. Known source defects
need explicit disposition during acceptance-test authoring, not silent deletion
or weaker assertions. `PRE_MIGRATION.md` explains the cases and source evidence;
it is not a second executable gate format or an automatically verified receipt.

The old `cache_test` key and `cache` stage name are no longer accepted. Update
the reviewed adapter in the target, select the actual acceptance suite, and
initialize a new plan. Do not relabel an old cache-only suite as complete
acceptance coverage or rewrite old receipts. Performance measurement and direct
descriptor comparison still require the additional protocol below.

Illegal-write acceptance combines protected-memory assertions with supported
Watcher checks; neither alone is a complete allocation-ownership sanitizer. See
the skill's [case guide](skills/pre-migration-tests/references/cases.md#illegal-writes-check-ownership-as-well-as-addresses).
The current driver runs the Python assertions but does **not** enable or attest
Watcher. A recorded, native safety execution profile and its evidence checks
remain to be integrated before claiming that layer passed. Reuse acceptance
cases, not an additional full golden sweep, and do not use instrumented timing
as production-performance evidence.

Memory-poisoning cases also belong in the acceptance suite: vary permitted input
padding/verified guards and prefill overwrite-only outputs to expose unintended
reads, missing stores and missing zero-initialization. See the skill's
[poisoning guide](skills/pre-migration-tests/references/cases.md#memory-poisoning-and-missing-initialization).
The driver has no generic allocation/scratch-poisoning hook. Operation-specific
safe helpers and physical coverage evidence are required; prior-work sequences
alone do not establish direct scratch poisoning. No extra full golden stage is
needed for these parametrized acceptance cases.

### Descriptor cache-hit parity: required native configuration

The native acceptance suite runs with `ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON`.
This is a build option, not a pytest flag. The existing `factory_contract` gate
requires an enabled `build_Release/CMakeCache.txt` entry and compiles a probe
requiring `TT_DESCRIPTOR_PATCHING_PARITY_CHECK` using the actual factory TU flags.
Both the configuration and compile evidence are fingerprinted; OFF, missing
configuration, missing compile definition or later drift blocks validation.

Prepare this configuration in the target's normal build environment before
initializing validation. For a local Release build:

```bash
./build_metal.sh --enable-ccache --configure-only
cmake -S . -B build_Release -DENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON
# validate_port's normal build_argv then compiles/installs this configuration.
```

Use the same toolchain/build options as the eventual build command. Container
builds must perform configuration inside their build image as well. The driver
does not install toolchains or automatically change this CMake option, and
`build_metal.sh` at this revision does not accept arbitrary `-D` arguments.

On covered native cache hits, the adapter compares the refreshed program's
runtime arguments and tensor-backed CB addresses with a fresh native construction.
Keep the existing acceptance assertions and include proven non-no-op cache hits,
fresh retained buffers, and supported runtime-value/alias transitions. The
`descriptor_cache_hit_parity` review topic must identify those cases, their
cache-delta assertions and the binding-to-instrumented-operation trace. A flag
or positive operation-call count alone does not prove that a hit occurred.

This adds no extra acceptance invocation, clone or full golden sweep. It is a
correctness configuration, separate from production-performance measurement.
The checker does not establish numerical correctness, complete structural/key
equivalence or absence of unnecessary misses. It has no successful-hit counter;
execution coverage still depends on test assertions and review. Performance
measurements need the option OFF in a separately verified build; instrumented
cache-hit timings are not accepted as normal host-performance evidence.

### Required performance evidence before PR readiness

Profile the migrated C++ operation's **cache-hit path versus its cache-miss path**
using [the native hot/cold protocol](NATIVE_CACHE_PERFORMANCE.md). Use the same
operation and inputs, keep caching enabled and compiled kernels warm, and require
a statistically supported hot-path improvement. Truly cold kernel compilation is
a separate diagnostic, not the baseline used to make hits look faster.

This is an additional focused benchmark, not another full golden run. The driver
does not yet execute or statistically gate it. Its behavior-only `complete`
receipt (which permits `not_measured`) must not be presented as PR readiness when
this performance evidence is missing, failing or inconclusive.

## Legacy input: reconstructing a historical export

The flow is general; the C++ translation itself is authored and reviewed using
[MAPPING.md](MAPPING.md), not guessed by a retrieval script. Run IDs are inputs
to the exporter, never baked into migration code or tests.

For the stronger migration-acceptance decision, follow
[COMPARISON_GATE.md](COMPARISON_GATE.md). It specifies host behavior/planner
comparisons and measured host/device performance budgets in addition to this
driver's existing checks. Those extra gates are a review protocol, not current
automated stages; driver completion alone does not satisfy them.

If original shared dependencies cannot be recovered, the optional
[audited substitution contract](DEPENDENCY_SUBSTITUTIONS.md) requires explicit
user approval. Pass the same substitution list through source and target
validation and retain its changed-baseline scope in completion evidence.

1. Export exact candidate source and evidence with `tools.generic_op_to_factory.export_run`.
2. Freeze the recorded Git inputs using `tools.generic_op_to_factory.prepare_baseline`.
   This reads Git objects; it does not build or test a historical checkout.
3. Select one isolated target worktree and its compatible existing toolchain. Keep the
   recorded evaluator revision. `tools.generic_op_to_factory.prepare_target` verifies the frozen
   preparation, inventories changed target references, and installs only the
   absent source operation. It never replaces canonical helpers with old copies.
4. Author a named C++ operation: native validation/output allocation, reflected
   structural attributes, tensor arguments, a **ProgramDescriptor factory** with
   typed address bindings and an explicit per-Program cache-refresh hook,
   binding and source registration. Follow [the factory contract](FACTORY_CONTRACT.md):
   return `ProgramDescriptor` from `create_descriptor`, not `CachedProgram` from
   `create`. Regular factories, `ProgramSpec` and `WorkloadDescriptor` are not
   outputs of this flow. Fetch hardware geometry/format sizes from canonical APIs; obtain
   resource IDs from allocation and share named argument schemas with kernels.
   Inventory kernels/helpers and preserve the algorithm; record formatter-only
   changes separately from independently justified ABI/API adaptations. No Python planner dispatch or
   fallback to `generic_op` is permitted in the native path.
5. Supply a native entry-point mapping and operation-specific cache regression
   tests to `tools.generic_op_to_factory.validate_port`. That driver checkpoints build → compiled factory contract → optional source smoke
   → target-source golden suite → DB comparison → optional native smoke → native golden
   suite → source/native comparison → native acceptance tests → independent code-quality /
   host-performance review → scoped completion evidence. Use the independent
   agent task and receipt contract in [REVIEW.md](REVIEW.md). Confirmed findings
   must be fixed and revalidated before completion; changed source needs a new
   validation workspace and a fresh review bound to that plan.

The default flow runs **two full golden suites on one final target build**:
frozen Python source, then native C++. The DB comparison and source/native
comparison read existing result files; they do not launch tests. Cache tests
are a separate, focused check of repeated invocation and address changes.
Historical-runtime replay with `migration_workflow` is an optional diagnostic
when target-source execution fails, differs from the DB, or historical
reproduction is explicitly requested. It is not a prerequisite for translation
parity and does not trigger a third default golden run.

All migration tools live under `tools/generic_op_to_factory/` in tt-metal.
Commands below run from this repository root; runtime build/tests run in the
chosen isolated tt-metal checkout. The evaluator is a pinned historical input,
not a second repository that needs tool changes. The target checkout must already have its environment, submodules and
compatible build configuration. The driver does not install a host toolchain.
The target's `tools/generic_op_to_factory/native_adapter.py` and safe runner must
match the reviewed hashes pinned in `test_evidence.py`, even when the driver and
target share a worktree. If the target revision predates a flow update, bring those
tool files into the target worktree before initializing validation; their diff/hash
is then part of the validation plan. Do not change it during a resumed run.

```bash
python3 -m tools.generic_op_to_factory.prepare_target --preparation /absolute/prepared \
  --runtime /absolute/target --target-revision FULL_TARGET_COMMIT
# After reviewing the reference differences, add --output /absolute/new-evidence
# to install the absent operation and save its input inventory.

python3 -m tools.generic_op_to_factory.validate_port plan --config /absolute/port.json
python3 -m tools.generic_op_to_factory.validate_port init --config /absolute/port.json
python3 -m tools.generic_op_to_factory.validate_port run --workspace /absolute/new-validation --through acceptance
# Independent agent reviews the final source and evidence; reconcile findings,
# then write /absolute/new-validation/review.json as described in REVIEW.md.
python3 -m tools.generic_op_to_factory.validate_port run --workspace /absolute/new-validation
python3 -m tools.generic_op_to_factory.validate_port status --workspace /absolute/new-validation
```

Example config (replace placeholders, including the complete commit):

```json
{
  "runtime": "/absolute/target",
  "target_revision": "FULL_TARGET_COMMIT",
  "preparation": "/absolute/prepared",
  "export": "/absolute/frozen-export",
  "phase": "initial",
  "workspace": "/absolute/new-validation",
  "source_entry": "ttnn.operations.sample_op:sample_op",
  "native_entry": "ttnn:sample_op_native",
  "source_aliases": [],
  "factory_contract": {
    "operation_header": "ttnn/cpp/ttnn/operations/sample/device/sample_device_operation.hpp",
    "operation_type": "ttnn::prim::SampleDeviceOperation",
    "factory_source": "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp"
  },
  "smoke_nodeid": null,
  "acceptance_tests": ["tests/ttnn/unit_tests/operations/test_sample_native.py"],
  "build_argv": ["./build_metal.sh", "--enable-ccache"],
  "precompile": false,
  "precompile_workers": 6,
  "allow_recorded_failures": false,
  "environment": {"CMAKE_BUILD_PARALLEL_LEVEL": "6"}
}
```

The factory contract is required, including when resuming a future port: its
generated C++ probe checks every factory alternative using the target build's
actual compiler flags. The independent reviewer must also trace `native_entry`
to that exact operation type; a header-only type check cannot prove dispatch.
Historical validation receipts remain historical evidence, not proof of this
new contract. Do not rewrite receipts or convert an already completed operation
merely to adopt the flow update.

For Ninja builds the factory gate extracts the actual compiler commands with
`ninja -t compdb`, preserving the normal unity build. It fingerprints the Ninja
metadata and relevant unity source. Other generators need a complete
`build_Release/compile_commands.json` containing the factory translation unit.
A partial UMD-only database is not accepted. Do not switch off unity merely to
obtain this evidence.

Smokes are disabled when `smoke_nodeid` is omitted or null. Precompilation is
optional warmup, not an additional golden result. If enabled, the target safe
runner must write warmup JUnit to a separate file. The real invocation reports
its raw pytest exit code: crashes, interruptions and incomplete runs cannot be
accepted as ordinary failing tests. Source/native evidence also requires the
expected adapter route and a positive observed operation-call count, including
when the suite has failures. Bring the updated safe runner into an older target
before freezing the validation plan.
Target preparation permits only this driver's exact safe-runner update as an
uncommitted reference change, recording both target-commit and actual hashes.
This tooling update is not historical-runtime reproduction; arbitrary helper
or harness changes still fail preparation.

The source module must be the frozen operation package. The adapter replaces
its exported function before golden-suite collection, retaining the recorded
registry metadata, golden inputs and tolerances. Native mode rejects calls to
`ttnn.generic_op`; the adapter is test scaffolding, not a production dispatcher.
If the frozen package also re-exports itself at another entry point used by the
suite, declare it explicitly in `source_aliases`, for example
`["ttnn:sample_op"]`. Before changing any routing symbol, the adapter verifies
each alias is the exact frozen callable; it refuses to replace an unrelated
existing operation with the same name. All replacements are restored at teardown.
Keep the native binding distinct from source and alias names. This handles
known pre-collection aliases, not arbitrary later rebinding inside tests.
An explicitly selected smoke case must pass. Acceptance tests must all pass, with no skips or xfails.
Their assertions must cover the port's actual address/scalar/optional/alias
transitions; a green generic test count is not a substitute for that review.

In branch mode, preparation and validation evidence must be separate directories
under `RUNTIME/generated/generic_op_to_factory/`. Its generated local ignore marker keeps
logs/caches out of source snapshots; tracked source there is forbidden. The legacy
historical mode still requires its validation workspace outside the target repo.
In either mode, validation outputs cannot be placed inside frozen inputs.
It records tracked diffs, untracked non-ignored source hashes, input/reference
hashes, commands, logs, JUnit and build libraries. Source/configuration drift
blocks resume. `--through STAGE` stops at a checkpoint. Failed/interrupted
stages require inspection followed by explicit `--retry`; old evidence is
retained, and potentially live unfinished commands block retry. Changes to a
port require a new validation workspace, not edited receipts. A completed
comparison is not a request to rerun a completed device stage.

Completion requires a recorded independent review and is scoped to the recorded
golden outcomes/tolerances and supplied native acceptance tests. It does not claim universal support, tracing compatibility,
performance improvement or production readiness. A failing historical baseline
requires explicit acceptance and remains reported as failing after migration.
