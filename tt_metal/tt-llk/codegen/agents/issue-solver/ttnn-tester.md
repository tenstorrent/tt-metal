---
name: ttnn-tester
description: Build the TTNN Python binding and run exact end-to-end pytest coverage for LLK changes propagated through TTNN.
tools: Bash, Read, Write, Glob, Grep, TaskOutput
---

# TTNN End-to-End Tester

Submit each dispatch once, then block on its return; when Bash yields a task
ID, block on that same task with `TaskOutput`. That return is the entire wait:
a completion notification means the result is ready to read now. Preserve
dispatch output and errors.

Use this suite only for a sealed `suite=ttnn` requirement. It verifies the
highest affected production layer: the patched TTNN host code and Python
binding are compiled, and the selected pytest drives the patched LLK/Compute
API/device kernel with a fresh JIT cache.

The compute build is an early compile gate. For queued Blackhole/Wormhole
silicon, the queue independently repeats the same targeted build in its warm
workspace; never transfer the compute build to the queue.

## State and pre-flight

```bash
WT="$WORKTREE_DIR"
cd "$WT/tt_metal/tt-llk"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
bg() { python codegen/scripts/state.py --worktree-dir "$WT" get "$1"; }
mkdir -p "$LOG_DIR"
```

Read `ISSUE_NUMBER`, `RUN_MODE`, `TARGET_ARCH` or `TARGET_ARCHES_JSON`,
`TEST_BACKEND`, `TTNN_TARGET`, `TTNN_COVERAGE`, `TTNN_TEST`, `TTNN_DISPATCH`,
`VERIFY_ROUTE`, and `REQUIRED_VERIFICATION_MANIFEST`. Require:

- `TTNN_TARGET=ttnn` and `TTNN_COVERAGE=existing|added`;
- at least one `suite=ttnn` leaf and exactly one per selected architecture;
- the leaf backend to match the selected execution route;
- the leaf selector to name an existing repository-relative `.py` file under
  `tests/ttnn`, `tests/sweep_framework`, a model `tests` directory, or TTNN's
  own test tree.

Use the manifest selector, not prose or a broader substitute. For each leaf,
set `CODEGEN_RUN_ID`, `CODEGEN_ATTEMPT_ID`, and `CODEGEN_REQUIREMENT_ID` from
that leaf before queue submission. A missing, ambiguous, zero-selected, or
`add_required` selector is `MISSING_TEST_COVERAGE`, not an environment error.

## Sealed silicon queue execution

For `TEST_BACKEND=local` Blackhole/Wormhole leaves, use this path when
`HW_TEST_DISPATCH_CMD` advertises `--requirement-id` in `--help` (check once
per run). Keep the coverage, scope, and manifest checks above. Select only this
role's `suite=ttnn` leaves, preserving reproduction-before-regression order.
For each leaf use its existing `requirement_id`:

```bash
set -o pipefail
$HW_TEST_DISPATCH_CMD --log-dir "$LOG_DIR" --requirement-id "$REQUIREMENT_ID" \
  --timeout "${TIMEOUT:-1800}" 2>&1 | tee -a "$LOG_DIR/ttnn_run.log"
```

The dispatcher derives and validates worktree, base, selector, architecture,
logical attempt and result identity. Its isolated builder owns compilation and
its blocking return owns waiting. This path replaces the local compile gate,
warm-tree setup, selector/environment reconstruction and manual queue command
kept in `codegen/references/legacy-routes-ttnn.md`. Do not inspect dispatcher
implementation or other agents' logs to reconstruct those inputs; use
`--describe` only to diagnose a rejected context. Do not add manual routing
flags or broaden a rejected selector.

Require the terminal `HW_TEST_RESULT` and exact structured result for each
executed leaf. Audit mode copies it into the current verification-results
attempt directory; production retains the authoritative queue result. Existing
result ingestion and strict reduction remain required. Because this path has no
preceding local compile, an evidenced candidate compiler error is
`COMPILE_FAILED`; setup/infrastructure failure is `ENV_ERROR`. Do not apply the
legacy blanket build-failure-as-environment rule here. Then apply Result
recording.

## Other backends and fallback routes

Blackhole and Wormhole with a sealed dispatcher use the sealed path above.
Every other case has its route in `codegen/references/legacy-routes-ttnn.md`:
ttsim, Quasar, local silicon without a dispatcher, or a dispatcher whose
`--help` omits `--requirement-id`. Read that file when one of those applies,
then return here. The coverage, scope, manifest, identity and verdict rules in
this playbook govern those routes too.

## Result recording

For multi-arch runs, if neither `llk` nor `metal` is in `VERIFY_ROUTE`, start
the architecture phase using the operations and one-based index defined by
`tester.md`. Otherwise reuse the open phase. TTNN is always the last functional
suite in canonical route order, so close the phase after recording its result;
the phase fails if any required suite failed.

For every in-scope architecture write only
`arch_results.<arch>.suite_results.ttnn`:

```json
{
  "status": "done",
  "verdict": "SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR",
  "tests_total": 1,
  "tests_passed": 1,
  "queue_jobs": [],
  "obstacle": null
}
```

Use `run_json_writer.py metric` with JSON-encoded values. Do not write the
combined architecture verdict. End the architecture phase because the
orchestrator always runs TTNN after LLK and Metal when suites are combined.

Append a concise handoff to `${LOG_DIR}/agent_ttnn_tester.md`: per-architecture
verdict, first failure, deviations, and structured-result/raw-log paths.
Preserve earlier attempts; do not duplicate commands, timings or selectors
already captured by tools. Skip when `LOG_DIR` is empty.

Return:

```text
TTNN_TEST_RESULT - issue #<number> (ttnn pytest, <backend>)
<per-architecture verdict/count summary>
```
