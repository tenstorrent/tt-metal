# Prepare a baseline from a frozen run export

This is the Git-input preparation step of the default [migration flow](PORT_FLOW.md).
After preparation, use `prepare_target` for the shared source/native target.
The historical checkout/install instructions below apply only to the optional
[historical-runtime diagnostic](MIGRATION_WORKFLOW.md).

This stage reads Git objects, not an LLM or the current checkout's files. It
does not fix generated code or turn a failed run into a passing operation.

```bash
python3 -m tools.generic_op_to_factory.prepare_baseline \
  --export "$EXPORT_DIRECTORY" \
  --eval-repository "$EVAL_REPOSITORY" \
  --metal-repository "$METAL_REPOSITORY" \
  --output "$PREPARATION_DIRECTORY"
python3 -m tools.generic_op_to_factory.prepare_baseline --verify "$PREPARATION_DIRECTORY"
```

The export must pass offline verification, both full recorded commits must be
available, and the output must not exist. No Git fetch, branch changes or
alteration of the DB package occur.

## Pinned inputs

- `overlay/ttnn/ttnn/operations/<prompt_name>/`: unchanged DB host/kernel text,
  placed in the operation package implied by the recorded prompt name.
- `overlay/eval/`: the complete selected golden directory, parent package
  initializers/conftest, three eval pytest plugins, and recursively resolved
  static `eval` imports, all from `eval_commit`.
- `reference/metal/`: the entire canonical `ttnn/cpp/ttnn/kernel_lib` family,
  support-contract exceptions, root conftest, pytest configuration, safe runner
  and `.gitmodules`, from `starting_commit`. These are reference bytes, not
  replacements to overlay onto a different runtime revision.
- `reference/evaluator/`: recorded orchestration and golden-runner scripts.
- `baseline.json`: file hashes and origins, input export digests, parent
  gitlinks, recorded status/configuration, external Python imports, recognized
  dynamic import/evaluation calls, and remaining verification gates.

Missing required modules, unsafe paths, symlinks and submodules inside selected
file trees fail. Python is parsed but never imported. Static analysis is not a
proof of dependency closure: dynamic aliases, runtime file reads, platform/API
headers, Python distributions and compiled libraries still depend on the full
pinned runtime. The whole canonical helper family is preserved; this is not
advertised as a minimal transitive include closure.

`eval_commit` and the parent's eval gitlink are distinct provenance fields.
Inspect the pinned setup script and `.gitmodules`: some revisions leave a
populated submodule untouched; others populate a skipped (`update = none`)
submodule from the orchestrator revision. Preparation records both fields,
without silently substituting one. The DB does not prove that a run never
changed a helper or harness outside its ingested source tables.

## Optional historical runtime and installation

Create an isolated tt-metal checkout at `starting_commit` under the user's
tt-metal workspace. Run `git submodule update --init --recursive`. Materialize
the evaluator at the selected `eval_commit`, preserving the root `eval` link
and documenting any intentional difference from the parent gitlink.

```bash
python3 -m tools.generic_op_to_factory.prepare_baseline \
  --install "$PREPARATION_DIRECTORY" --runtime "$RUNTIME_DIRECTORY"
```

Installation verifies the preparation, runtime/evaluator HEADs, clean tracked
runtime files (excluding submodule differences), and exact selected golden
files/canonical references. It creates only a previously absent operation
directory. It never overwrites a candidate or changes Git revisions. It does
not certify submodule cleanliness, untracked dependencies or the loaded build.

Build through applicable repository instructions (`build_metal.sh`, or its CI
wrapper where required). Use `./create_venv.sh` when no environment exists;
activate `./python_env/bin/activate` only after creation. Use a verified
available compiler of the required version, or report the mismatch. Do not
bypass version checks or substitute another checkout's compiled operation.

## Execution gates

Record runtime HEAD/submodules, build command/configuration, package versions,
loaded `ttnn`/extension paths, source hashes, device architecture and selected
environment settings. Never dump credentials or an unrestricted environment.

Run tests through `./scripts/run_safe_pytest.sh` from the runtime root. Start
with collection and a narrow smoke case, then the complete pinned suite. Use
`--run-all` for the full outcome inventory; retain locking and hang handling.
Keep results outside immutable preparation packages.

The historical golden runner uses `eval.hang_plugin`, `eval.metrics_plugin`
and `eval.axes_plugin`. Preserve their behavior; inspect environment, profiler
and precompile defaults at the recorded revision. Document deliberate changes,
such as disabling warm precompilation for smoke tests, as new settings rather
than historical facts. Clear unrelated pytest filters and isolate
`TT_METAL_HOME`, imports and device caches to the selected runtime.

Also inspect the frozen `records/device_timings.jsonl`: when populated, its
precompile mode, reason and program count provide evidence of actual historical
routing beyond script defaults. Verify the loaded build's collector bindings
before requesting precompilation. Keep warm-pass logs separate and never
interpret compile-only results as correctness results.

Compare case identities and outcome classes, not only aggregate passes.
Distinguish failures, expected refusals, skips, collection/setup errors and
incomplete/hung execution. Preserve golden tolerances. Select the historical
DB phase explicitly: current recorded source is not versioned by phase. A new
baseline can be useful without proving exact historical replay; keep that
distinction explicit.

Preparation and installation never mark `baseline_reproduced` or
`migration_ready` true. Device execution and the remaining gates need their
own evidence before C++ translation.

## Compare a completed run

The comparison CLI requires one explicit historical phase:

```bash
python3 -m tools.generic_op_to_factory.compare_baseline --export "$EXPORT_DIRECTORY" \
  --junit "$JUNIT_REPORT" --phase "$RECORDED_PHASE"
```

Use `--unphased` instead of `--phase` to select SQL NULL phase rows. There is
no implicit final/best phase and no pooling across phases. It verifies the
export and compares `(test_file, test_name)` identities and outcome classes
using the existing JUnit classifier. Empty, duplicate or ambiguous identities
fail rather than being silently dropped. Output includes both input hashes,
counts, missing/added cases and changed outcomes. An interrupted run with
missing cases cannot count as equivalent.

Exit codes: 0 means outcome parity (which may include recorded failures),
1 means differing outcomes/case sets, and 2 means invalid inputs. This is not
a metrics/performance comparison, failure-message equivalence check, or
attestation that runtime provenance matches. `migration_ready` stays false.

## Host-side tests

```bash
./scripts/run_safe_pytest.sh --run-all --no-precompile tools/generic_op_to_factory/tests/test_export_run.py tools/generic_op_to_factory/tests/test_prepare_baseline.py tools/generic_op_to_factory/tests/test_compare_baseline.py -q
```

Run from this tt-metal repository. Fixtures use synthetic DB rows and temporary
Git repositories, never named production runs, network or accelerator access.
