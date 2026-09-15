# Post-process an evaluated branch

The migration input is the branch **after evaluation**, with the complete Python
operation, kernels, supporting C++ APIs and evaluator gitlinks. A DB run's
`starting_commit` is not that checkpoint. DB exports can be retained as audit
evidence but are not used to install or overwrite source in this mode.

## 1. Checkpoint and select the branch

The caller must first preserve every in-run modification in Git, including
submodule changes (commit those in the submodule, then record its gitlink in the
parent). Preparation rejects dirty checked-out copies of the selected branch
or commit, including non-ignored untracked files. It does not commit or stash
them automatically. Ignored files, external dependencies, virtual environments
and device state are not captured by a Git checkpoint; audit required runtime
inputs separately. Missing ignored source must be explicitly checkpointed.

Branches resolve from local refs only. Supply a local branch, an explicit
remote-tracking name such as `origin/evaluated-candidate`, or a full
`refs/heads/...` / `refs/remotes/...` ref. Missing or ambiguous refs fail; the tool
never falls back to main, HEAD, a DB revision or another operation. Fetch the
desired branch explicitly beforehand when necessary.

Run from the checkout containing this flow:

```bash
python3 -m tools.generic_op_to_factory.prepare_branch \
  --repository /localdev/astancov/tt-metal \
  --branch evaluated-candidate \
  --runtime /localdev/astancov/sample-migration-target \
  --operation sample_op --golden-suite sample_suite \
  --output /localdev/astancov/sample-migration-target/generated/generic_op_to_factory/inputs \
  --with-validation-tools
```

For the discussed Tilize example the supplied branch would be
`2026_09_09_1410_run1_tilize`, with operation/suite `tilize`. This is an input
example, not a special case in the code; this change does not run that migration.

Preparation resolves the branch once to a full SHA and creates a **new detached
worktree** at that SHA. It runs `git submodule update --init --recursive --checkout`
to initialize the branch's own gitlinks, including submodules marked `update =
none`. It never replaces the evaluator pin with a DB value. The complete
checkpoint is preserved, not just files under the Python operation directory.
No branch checkout, reset or edits occur in the source worktree.

The target worktree must be new. Preparation output defaults to
`RUNTIME/generated/generic_op_to_factory/inputs` and must be a new directory inside that
worktree-local artifact namespace. Validation uses a separate sibling, such as
`RUNTIME/generated/generic_op_to_factory/validation`. Parent-directory outputs and paths
inside operation source are rejected. The namespace has its own generated
`.gitignore`; neither the evaluated `.gitignore` nor shared Git configuration is
modified. Tracked source, migration allowlist entries and redirected paths are
forbidden there. Logs/caches therefore do not contaminate source hashes.

After worktree creation, command logs and failures are retained inside it; a
failed attempt does not delete its partial worktree. If Git cannot create the
worktree, its diagnostic is returned without making a fake checkout or loose
parent-directory logs. `branch.json`
records branch name, immutable source SHA, runtime path, operation/suite and
recursive submodule SHAs with a checksum. Later movement of the original branch
does not move an already prepared migration.

`--with-validation-tools` installs exactly two reviewed tooling files into the
new worktree: the safe pytest runner and test-only route adapter. Their changes
are hash-checked and recorded separately. It does not install operation source,
rewrite tests or provide SDK compatibility shims. Omit the flag if those exact
tool versions are already in the branch, or install them before validation.

## 2. Bootstrap and author the native port

Use the new target's normal `./build_metal.sh` configuration and its own
environment. If `python_env` is absent, create it with `./create_venv.sh` before
`source ./python_env/bin/activate`. Preparation does not install toolchains or
copy environments. Configure any required existing toolchain explicitly.

Author a distinct native entry using [MAPPING.md](MAPPING.md) and
[FACTORY_CONTRACT.md](FACTORY_CONTRACT.md). Keep the original Python operation
callable unchanged for source/native comparison on the same build.

List the exact native files and integration files in `migration_paths`.
Directories, source-package edits, evaluator/test edits and runtime-setup edits
are refused. All other changes since the evaluated checkpoint must be declared;
the reviewed runner/adapter are the only implicit exceptions. Native commits
are allowed if they descend from the checkpoint; initialization then freezes
the current HEAD plus tracked/untracked changes. No changes may occur on resume.

## 3. Validate through the existing driver

Example `RUNTIME/generated/generic_op_to_factory/inputs/port.json` (replace placeholders):

```json
{
  "runtime": "/localdev/astancov/sample-migration-target",
  "evaluated_branch": "/localdev/astancov/sample-migration-target/generated/generic_op_to_factory/inputs",
  "workspace": "/localdev/astancov/sample-migration-target/generated/generic_op_to_factory/validation",
  "migration_paths": [
    "ttnn/cpp/ttnn/operations/sample/device/sample_device_operation.hpp",
    "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp",
    "tests/ttnn/unit_tests/operations/test_sample_native.py"
  ],
  "source_entry": "ttnn.operations.sample_op:sample_op",
  "native_entry": "ttnn:sample_op_native",
  "source_aliases": [],
  "factory_contract": {
    "operation_header": "ttnn/cpp/ttnn/operations/sample/device/sample_device_operation.hpp",
    "operation_type": "ttnn::prim::SampleDeviceOperation",
    "factory_source": "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp"
  },
  "acceptance_tests": ["tests/ttnn/unit_tests/operations/test_sample_native.py"],
  "build_argv": ["./build_metal.sh", "--enable-ccache"],
  "precompile": false,
  "environment": {"CMAKE_BUILD_PARALLEL_LEVEL": "6"}
}
```

The illustrative `migration_paths` is deliberately incomplete for a real op:
also list bindings, source registration, kernel copies and every other authored
file. Do not add the original source or original golden tests to this list.

```bash
python3 -m tools.generic_op_to_factory.validate_port plan --config /localdev/astancov/sample-migration-target/generated/generic_op_to_factory/inputs/port.json
python3 -m tools.generic_op_to_factory.validate_port init --config /localdev/astancov/sample-migration-target/generated/generic_op_to_factory/inputs/port.json
python3 -m tools.generic_op_to_factory.validate_port run \
  --workspace /localdev/astancov/sample-migration-target/generated/generic_op_to_factory/validation --through acceptance
# Obtain independent review and supply review.json using REVIEW.md.
python3 -m tools.generic_op_to_factory.validate_port run \
  --workspace /localdev/astancov/sample-migration-target/generated/generic_op_to_factory/validation
```

There is one build, one full source golden, one full native golden and a separate
focused native acceptance suite. The source/native route is checked and native generic-op
fallback is forbidden. `source_compare` records the fresh source outcomes, not
a DB comparison. **The source suite does not have to be green.** Ordinary source
failures/errors remain visible and do not prevent the native suite from running.
The obsolete `allow_source_failures` configuration key is no longer accepted.
`native_compare` is mandatory: changed, missing or additional case outcomes block
completion, including a source failure becoming a native pass. Matching failures
are never relabeled passes. Hangs, missing/untrusted execution evidence and broken
test routing still block execution. All selected native acceptance tests must pass.
Existing DB outcomes can still be examined
separately with `compare_baseline`; they are not a prerequisite for this mode.

Existing evaluation results are not automatically reused: proving their binding
to this exact tree, suite and environment is a separate future capability.
Independent review remains mandatory. Native descriptor cache-hit instrumentation
is now required by the [factory/acceptance flow](PORT_FLOW.md#descriptor-cache-hit-parity-required-native-configuration).
Direct Python/native descriptor comparison and measured host/device performance
remain the additional protocol in [COMPARISON_GATE.md](COMPARISON_GATE.md), not
automated gates. No performance claim follows from branch preparation.

Old external evidence may be archived under the target's artifact namespace.
Preserve its bytes and record the old/new paths separately; absolute paths and
hashes in old receipts are historical facts, not fields to rewrite for resuming.
A moved old workspace is an archive, not a resumable validation. New policy or
source changes require a new plan; cross-workspace result import is not implemented.
