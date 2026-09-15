# Record the baseline and the next step

Write `PRE_MIGRATION.md` beside the new tests. Keep this report short and link to
raw evidence. This is a human-readable handoff, not a `validate_port` receipt or
an automated gate in that driver.

## Identity

Record the operation and exact `module.path:callable`, source files, repository
HEAD, recursive submodule revisions, loaded module/library paths and relevant
build/environment configuration. Include architecture/device or simulator mode.

List the tests, reference helpers, Python planner, kernels and shared files the
cases depend on. Record their hashes plus relevant tracked diffs and untracked
files, including changes within the evaluator submodule. Do not hash virtual
environments or large generated directories indiscriminately.

If the evaluated branch was dirty, say so. Compare identities before and after
the run to detect concurrent changes. Record the final test hashes after fixing
test mistakes and rerunning affected cases. A commit SHA alone does not identify
a dirty test baseline.

## Case matrix

Use a row per meaningful case or parametrized group:

| Case ID / pytest node | Rule or planner branch | Inputs and changed axis | Expected output/error and memory effects | Cache sequence and expected deltas | Source evidence | Result |
| --- | --- | --- | --- | --- | --- | --- |

For each changing field, include a short cache-key decision with a source
reference. Record untested supported branches, inapplicable features and future
native-only checks separately. A test function name without its actual case
values is not a useful coverage record.

## Execution and outcome

Include commands, selector values, seeds/tolerances, logs, JUnit paths, raw pytest
exit status and pass/fail/skip/xfail counts. Separate new-suite results from the
existing golden results. Verify required cases were collected and executed and
that the selected operation was called. Explain any reused golden evidence.

For illegal-write coverage, distinguish protected-tensor/region assertions from
verified allocation red zones and Watcher instrumentation. Record the exact
architecture, effective Watcher settings, attachment and clean completion/log
evidence, which bytes were checked, and any unchecked memory or API paths.
An ordinary acceptance pass is not a Watcher pass; the driver's instrumented
execution/evidence profile is pending. Keep that gap visible in the handoff.

For poisoning, record each region's contract (ignored padding, allocation guard,
overwrite-only output, or private scratch), physical placement/readback evidence,
patterns/seeds after dtype conversion, initialization order and the independent
expected values/zeros. Distinguish direct poisoning from prior-work experiments
and list inaccessible regions/hooks. Do not report output invariance alone as
proof that no out-of-bounds reads occurred.

Choose one status:

- **Acceptance suite verified on source:** the proposed acceptance requirements
  passed on the unchanged generic operation. Record existing golden results
  separately; the whole golden suite need not be green.
- **Source failures:** tests exposed wrong results, broken behavior or hangs.
  List failing node IDs, reproduction commands and why this is a source issue.
- **Unverified:** tests are written, but required execution or reliable evidence
  is missing. Say exactly which checks remain and why.

Known source failures remain failures. A failing proposed acceptance requirement
needs explicit disposition before it can be called source-verified. Do not
silently drop cases, widen tolerances or turn failures into skips/xfails. Record
any agreed diagnostic-only scope separately from the mandatory acceptance list.

## Handoff to the port

List source files to preserve, new tests/helpers to checkpoint, and any evaluator
changes that need their own submodule commit before the parent gitlink is pinned.
Checkpointing is a caller step unless separately requested. The same files and
assertions should remain fixed when the native route is added.

Record the exact source-run command and the later native-run command using the
selector already implemented in the suite. For example, after substituting real
paths and symbols:

```bash
TT_PRE_MIGRATION_MODE=source \
  TT_PRE_MIGRATION_ENTRY='ttnn.operations.example:example' \
  ./scripts/run_safe_pytest.sh --run-all --no-precompile \
  tests/ttnn/unit_tests/operations/example/test_acceptance.py -q

# Future command only; requires the actual registered native binding.
TT_PRE_MIGRATION_MODE=native \
  TT_PRE_MIGRATION_ENTRY='ttnn:example_native' \
  ./scripts/run_safe_pytest.sh --run-all --no-precompile \
  tests/ttnn/unit_tests/operations/example/test_acceptance.py -q
```

Do not execute or report success for the future command before the native entry
exists. Explain how its route evidence and generic-fallback rejection will be
verified. Keep any justified source-only planner checks separate; do not disable
shared correctness or cache assertions for the native route.

For later driver integration, report the exact files in `acceptance_tests`, e.g.:

```json
{
  "acceptance_tests": [
    "tests/ttnn/unit_tests/operations/example/test_acceptance.py",
    "tests/ttnn/unit_tests/operations/example/test_memory_contract.py"
  ]
}
```

List only actual files; one file is sufficient when it covers the operation.
The `acceptance` stage supplies native selectors through its route adapter,
checks collection and native execution, and requires every selected test to
pass. Do not add selectors to the driver's global `environment`. The driver
does not rerun these tests on Python: that execution belongs to this skill.
Existing goldens still run on both routes. The handoff is coverage explanation,
not another custom executable gate format or an automatically verified receipt.

The branch preparation rules protect tests already in the evaluated checkpoint.
Set up callable selection now; do not plan to rewrite those tests through
`migration_paths`. Once C++ exists, the later migration run still needs its build,
real route checks, source/native golden comparison, native acceptance tests and review.
