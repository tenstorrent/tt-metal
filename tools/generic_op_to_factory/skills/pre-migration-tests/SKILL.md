---
name: pre-migration-tests
description: Write and run operation-specific tests against an existing Python generic TTNN operation before migrating its host code to C++. Cover the current API, outputs, planner branches and cache behavior, then record a source baseline for the later port. Use when preparing an evaluated operation for migration, not when implementing or tuning the operation.
---

# Write Python acceptance tests before the port

Create runnable tests for the **existing generic operation**. Run them on that
unchanged operation and hand off the tests, case matrix and source results.
Writing C++ or changing the operation is outside this skill's scope.
These Python tests are the executable acceptance gates: API, validation,
outputs, memory effects and illegal writes, poisoned-memory/initialization
checks, numerical behavior, supported branch boundaries and cache transitions.
The later native correctness build must enable descriptor-patching parity:
cache-hit runtime state is checked against a fresh native reconstruction.
The migration driver later runs them against C++ only.
Existing golden-suite Python/C++ parity is a separate check.

For a concise review overview, see
[what we test and which bugs it catches](references/coverage-summary.md).

Invoke in Claude with `/pre-migration-tests OPERATION`, optionally naming the
evaluated branch, checkout or exact `module.path:callable`. Use the conversation
and supplied arguments to resolve the operation. Ask only if the target or a
required behavior cannot be determined from the source and existing tests.

## 1. Find the source and its contract

Work in the requested tt-metal checkout. Read its applicable `AGENTS.md` and
`CLAUDE.md`. Inspect Git status before editing; preserve existing work.

Trace the public Python callable through validation, output creation, descriptor
planning and `ttnn.generic_op`. Read the kernels and shared helpers needed to
understand argument roles, work splits, buffer formats and padding. Record the
exact callable and files; do not substitute a similarly named built-in TTNN op.

Read the branch's existing golden suite, reference function, tolerances and test
helpers. The default evaluator location is
`tt_metal/third_party/tt_ops_code_gen/eval/golden_tests/OPERATION/`; discover the
actual suite name. Use the **final evaluated tree and its evaluator revision**,
not a DB run's starting commit or a different checkout's tests.

Separate these three things in the case matrix:

- What the operation currently promises: signature, supported configurations,
  outputs, mutation and errors.
- What it does today, including source bugs discovered by the tests.
- Future support from `TARGET` or similar metadata. This is not current support.

Do not regenerate golden suites or widen `SUPPORTED`, change `EXCLUSIONS`, invent
`INVALID` cells, or loosen tolerances to make this baseline pass. Reuse existing
test helpers when they really reach this source callable. This skill complements
golden-suite creation and reference-test translation; it does not run those
broader workflows automatically.

## 2. Write a small case matrix, then the tests

Use [references/cases.md](references/cases.md) to choose cases and assertions.
Map each current validation rule and meaningful planner branch to a named case.
For each varying field, record whether it needs a different program, changes on
a cache hit, or is safely omitted from the key. Cite the source reason.

Prefer a new, additive suite under
`tests/ttnn/unit_tests/operations/OPERATION/`, following the local test layout.
Keep the existing golden suite intact. Produce:

- `test_acceptance.py`: shared observable-contract tests, including cache tests.
  Split into additional Python files when useful; the driver's `acceptance_tests`
  input accepts an explicit list of files. Keep source-only planner inspection
  outside this list.
- Small operation-specific fixtures/helpers as needed, including one callable
  selection point shared by the tests.
- `PRE_MIGRATION.md`: source identity, cases, commands, outcomes and the handoff,
  using [references/handoff.md](references/handoff.md).

Write real inputs and assertions, not empty scaffolds, skipped placeholders or
only a plan. Favor table-driven cases over a full Cartesian product. Give cases
stable, descriptive pytest IDs. Cover the supported branches and their boundary
neighbors; list untested supported cases explicitly.

Run mathematical checks against an independent PyTorch/CPU reference. The source
operation's output cannot be its own oracle. Use the agreed dtype/quantization
convention and existing justified tolerances. Check metadata and memory effects
as well as values.

Do not use TTNN comparison mode as acceptance evidence. Write explicit assertions
against the independent reference so each exercised case, output, tolerance and
failure remains attributable. Comparison mode being enabled incidentally does not
satisfy any requirement in this skill; disable it for the recorded runs.

Include illegal-write checks using the layered approach in
[references/cases.md](references/cases.md#illegal-writes-check-ownership-as-well-as-addresses):
assert protected memory is unchanged and specify a Watcher-instrumented run.
Record unsupported instrumentation/guard coverage explicitly; a correct output
or a clean Watcher log alone does not prove absence of illegal writes. The
current migration driver does not yet enforce a Watcher execution profile.

Also select [poisoning and missing-initialization cases](references/cases.md#memory-poisoning-and-missing-initialization).
Vary only safely controlled memory the contract says must not affect the result;
check independent expected values, including promised zeros. Distinguish padding
dependence, actual allocation overreads and reads of uninitialized state. Do not
claim internal scratch/CB poisoning without a verified test hook.

Use the [advanced-check escalation guide](references/advanced-checks.md) to keep
required acceptance evidence separate from risk-triggered diagnostics and checks
that only apply after native C++ exists. Do not enable incompatible instrumentation
in one process or substitute a diagnostic run for the ordinary correctness run.

## 3. Make the suite usable before and after migration

All operation calls in the new suite go through one fixture/helper. It defaults
to the exact source callable and works with **no native binding installed**.
Include test-only selectors `TT_PRE_MIGRATION_MODE=source|native` (default:
`source`) and `TT_PRE_MIGRATION_ENTRY=module.path:callable` for later runs. Source
mode must select the declared source callable. Native mode requires an explicit,
different, registered C++ callable (`is_cpp_operation` in the current binding
API). Fail on an invalid selection; never silently fall back. Keep cases,
references and tolerances independent of these selectors.

Implement the native-mode checks now, but activate them only when that mode is
requested. In native mode, forbid `ttnn.generic_op` through a scoped test patch
that raises, and account for any pre-imported generic aliases in the tested call
path. Scope this guard to the selected operation call so setup/readback helpers
are not mistaken for native fallback. Restore patches afterward. Do not import
or require a native symbol in source mode. These checks can be authored now;
their native execution remains unverified until a real port exists.

Resolve the callable at test setup, so later routing hooks can replace it before
collection/setup. Log the selected entry, loaded module path and positive call
count for a real run; collection-only is not execution evidence. Observe the
generic dispatch on representative successful source cases when practical, using
a scoped spy that forwards the real call. Do not bypass the source planner or
require dispatch for legitimate no-op/refusal cases.

The current `tools/generic_op_to_factory/native_adapter.py` resolves a registered
native entry **even in source mode**. Do not load it for this pre-port run and do
not create a fake native function to satisfy it.

The driver's native-only `acceptance` stage loads that adapter with
`--migration-acceptance-route`. It verifies the registered C++ entry, temporarily
sets the two selectors above before collection/setup, and observes direct native
calls without replacing the source callable. It guards `ttnn.generic_op` only
during the selected native call, not setup/readback. Do not supply the selectors
as global driver environment overrides; the driver controls this stage's route.
Every selected file must collect tests, the native call count must be positive,
and every selected test must pass (no skips/xfails). Record the exact test-file
list and command. Native execution remains deferred until a real port exists.

Author cache-hit sequences now for the later required native descriptor-parity
check. Include at least one successful non-no-op miss followed by a proven hit;
cover fresh live buffers and supported runtime-value/alias transitions. Assert
the expected selected-operation cache deltas with caching enabled, outside
setup/readback, and still check outputs independently. Repeated calls alone
do not prove hits. List the exact cases in the handoff. Native build evidence
and limits are described in [references/advanced-checks.md](references/advanced-checks.md).
This native-only instrumentation is not required or claimed during the source run.

Keep source-planner/descriptor inspection tests separate from the shared behavior
tests. They can freeze the source plan now, but must not run the Python planner as
a supposed test of native planning later. Likewise, C++ factory concepts, GIL
behavior and native-hit planning cost cannot be verified before the port exists.

## 4. Run the source baseline

Use this checkout's environment. If `python_env` is missing, run
`./create_venv.sh` before `source ./python_env/bin/activate`. Python-only tests
need no rebuild; an incompatible or missing native runtime may require the normal
repository build. Follow `AGENTS.md` for builds; do not install host toolchains.

Run Python tests through `./scripts/run_safe_pytest.sh` in the foreground. For
example, after replacing the operation and evidence paths:

```bash
./scripts/run_safe_pytest.sh --run-all --no-precompile \
  tests/ttnn/unit_tests/operations/OPERATION/test_acceptance.py \
  -q --junitxml=generated/generic_op_to_factory/pre-migration/source.junit.xml
```

After the ordinary correctness run, rerun the focused cases that exercise tails,
partial writes, aliases, fresh live buffers, cache refresh and protected-memory
assertions with the wrapper's `--dev` profile. This enables the repository's
Watcher/NoC and CB instrumentation, lightweight kernel assertions, LLK assertions,
timeout triage and reset handling. Save separate logs and JUnit evidence. Record
the effective settings and architecture; on simulator the wrapper disables the
NoC sanitizer because of known false positives. A clean `--dev` run only establishes
that no supported instrumented violation was detected in those cases.

```bash
./scripts/run_safe_pytest.sh --dev --no-precompile \
  tests/ttnn/unit_tests/operations/OPERATION/test_acceptance.py \
  -q --junitxml=generated/generic_op_to_factory/pre-migration/source-dev.junit.xml
```

Set or clear the suite's selector explicitly for this source run. Save the
complete log and underlying pytest status as well as JUnit. Keep large runtime
artifacts inside this worktree's `generated/generic_op_to_factory/` directory,
separate from test source. Create the evidence directory before running. Preserve wrapper exit status when piping
output. Use required existing evaluator plugins/import paths when applicable;
do not invent runner flags or a second environment.

Start with a small supported case, then run the required new cases and the
existing operation golden suite. Existing evidence can avoid redundant golden
execution only if its exact source, suite, environment and completion are
verifiable; record that decision. It does not exempt the later migration driver
from its own source run.

Check actual pass/fail/skip counts. Expected support errors are passing negative
tests, not skipped coverage. A required skipped/xfail case, empty collection,
crash, hang or incomplete run cannot establish readiness. A profile run or a
precompile warmup is not correctness evidence. Do not reset devices manually.

If a test exposes a source bug, keep the failing case and report it. Fix mistakes
in the test when the contract proves them wrong; do not change the operation or
redefine success. If hardware/runtime is unavailable, finish all possible test
authoring and host-only checks, and report the device baseline as unverified.

## 5. Hand off before migration starts

Record one outcome: **acceptance suite verified on source**, **source failures**,
or **unverified**. Verification requires executed passing evidence for the
proposed acceptance requirements, not an entirely green existing golden suite.
Record golden failures separately; they do not by themselves block migration.
If a proposed acceptance requirement fails on the source, preserve the failing
case and request an explicit scope/defect disposition before presenting that
requirement as a verified gate. Never silently exclude it, mark it xfail or weaken
it to make acceptance pass. The driver does not interpret this report as a receipt
or automatically resolve source defects. No status here proves C++ correctness.

Record source/helper/kernel/test identities and relevant dirty state before and
after execution. Unexpected changes during the run invalidate that evidence.
Do not change tests after recording a pass without rerunning affected checks.

The tests and their helpers must be included in the evaluated checkpoint before
`prepare_branch` freezes it. Existing tests in that checkpoint cannot be edited
through `migration_paths`. Report what remains to be checkpointed; do not commit
or alter Git history as a side effect of writing the suite. Do not rebase, push
or reset without the user's permission.

Finish with links to the written tests and handoff, what ran, failures or missing
evidence, and the later native command. Do not begin the migration in this skill.

## Repository references

Paths below are relative to this skill's migration-tools location. For another
checkout, use that checkout's copies and actual APIs.

- [Testing survey](../../HOST_CODE_TESTING_SURVEY.md): detailed examples by family.
  The optional [visual guide](../../HOST_CODE_TESTING.html) explains the test patterns.
- [Branch preparation](../../PREPARE_EVALUATED_BRANCH.md): what the evaluated
  checkpoint must contain and which files are protected afterward.
- [Comparison protocol](../../COMPARISON_GATE.md): host-contract and cache tests
  that the future port must also pass. Performance gates belong to later work
  unless the user requests baseline measurements now.
- [Advanced-check escalation](references/advanced-checks.md): when to use the
  safety profile, memory reports, NoC diagnostics, Emule ASAN and later native-only
  checks without turning every diagnostic into a mandatory source gate.
