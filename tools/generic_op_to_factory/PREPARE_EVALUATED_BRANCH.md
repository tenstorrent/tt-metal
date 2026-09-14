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
  --output /localdev/astancov/sample-branch-inputs \
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

Both destination paths must be new and separate. Command logs and failures are
retained; a failed attempt does not delete its partial worktree. `branch.json`
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

Example `/absolute/port.json` (replace all placeholder files with the actual port):

```json
{
  "runtime": "/localdev/astancov/sample-migration-target",
  "evaluated_branch": "/localdev/astancov/sample-branch-inputs",
  "workspace": "/localdev/astancov/sample-validation",
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
  "cache_test": "tests/ttnn/unit_tests/operations/test_sample_native.py",
  "build_argv": ["./build_metal.sh", "--enable-ccache"],
  "precompile": false,
  "allow_source_failures": false,
  "environment": {"CMAKE_BUILD_PARALLEL_LEVEL": "6"}
}
```

The illustrative `migration_paths` is deliberately incomplete for a real op:
also list bindings, source registration, kernel copies and every other authored
file. Do not add the original source or original golden tests to this list.

```bash
python3 -m tools.generic_op_to_factory.validate_port plan --config /absolute/port.json
python3 -m tools.generic_op_to_factory.validate_port init --config /absolute/port.json
python3 -m tools.generic_op_to_factory.validate_port run \
  --workspace /localdev/astancov/sample-validation --through cache
# Obtain independent review and supply review.json using REVIEW.md.
python3 -m tools.generic_op_to_factory.validate_port run \
  --workspace /localdev/astancov/sample-validation
```

There is one build, one full source golden, one full native golden and a separate
focused cache suite. The source/native route is checked and native generic-op
fallback is forbidden. `source_compare` records the fresh source outcomes, not
a DB comparison. Source failures block by default; explicitly setting
`allow_source_failures` allows outcome-parity validation while retaining those
failures in the receipt. It never turns them into passes. Changed or missing
native outcomes fail comparison. Existing DB outcomes can still be examined
separately with `compare_baseline`; they are not a prerequisite for this mode.

Existing evaluation results are not automatically reused: proving their binding
to this exact tree, suite and environment is a separate future capability.
Independent review remains mandatory; descriptor parity and measured host/device
performance remain the additional protocol in [COMPARISON_GATE.md](COMPARISON_GATE.md),
not newly implemented gates. No performance claim follows from branch preparation.
