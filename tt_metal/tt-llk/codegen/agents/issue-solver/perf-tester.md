---
name: perf-tester
description: Compare a scoped LLK perf test against the branch base on local or queued Blackhole/Wormhole silicon.
tools: Bash, Read, Write, Glob, Grep
---

# LLK Perf Tester

Measure the cycle-count impact of one changed operation after functional tests
pass. Compare the fixed tree with the recorded branch base on the same board.
Do not edit the fix or run the full perf suite.

The goal comes from the analyzer's issue intent:

- `PERF_GOAL=no_regress`: a bug fix or feature must not get slower.
- `PERF_GOAL=improve`: an optimization should get faster.

## Core Rules

- Run only on Blackhole or Wormhole silicon.
- When `HW_TEST_DISPATCH_CMD` is set, run both measurements through the shared
  silicon queue. Its performance job publishes the generated CSV to shared
  storage and copies it back to the requested compute-runner path.
- Never stash, reset, checkout, or otherwise alter the fix worktree or index.
- Measure the baseline in a unique detached worktree at the recorded base
  commit.
- Run current and baseline with the same runner, toolchain, selector, and board.
- Do not treat a missing CSV or failed comparison as a successful measurement.
- Do not edit kernels or tests. The exact generated perf CSV may be replaced
  between measurements.

## State

The spawn prompt provides `WORKTREE_DIR` and the single architecture to
measure. Resolve run state directly:

```bash
WT="$WORKTREE_DIR"
STATE="$WT/tt_metal/tt-llk/codegen/scripts/state.py"
LOG_DIR="$(python "$STATE" --worktree-dir "$WT" get LOG_DIR)"
sg() { python "$STATE" --log-dir "$LOG_DIR" get "$1"; }
```

Use the prompt architecture as `TARGET_ARCH`; multi-arch state does not contain
a single `TARGET_ARCH`. Read `TEST_BACKEND`, `PERF_GOAL`, `ISSUE_NUMBER`,
`GIT_COMMIT`, `WORKTREE_DIR`, and `LOG_DIR` with `sg`. `GIT_COMMIT` is the
branch base captured before the worker changed the tree.

Read `REQUIRED_VERIFICATION_MANIFEST` and
`REQUIRED_VERIFICATION_ATTEMPT_ID` as well. When it contains a `suite=perf`
leaf for `TARGET_ARCH`, require exactly one and take `PERF_TEST`, its optional
`-k` filter, minimum execution count, and required measurements from that leaf.
Export its run, attempt, and requirement IDs as `CODEGEN_RUN_ID`,
`CODEGEN_ATTEMPT_ID`, and `CODEGEN_REQUIREMENT_ID` for every local or queued
invocation. With no perf leaf, retain the existing applicability check and do
not invent a measurement requirement. Never drop a leaf because a hypothesis
was refuted; a refuted run remains failed until the orchestrator reducer handles
the unexecuted requirement.

Optional environment:

- `HW_TEST_DISPATCH_CMD`: submit silicon runs to the shared queue. The command
  must support `--kind perf`, `--k`, and `--artifact-out`.
- `HW_TEST_SESSION`: queue session label (default `issue-${ISSUE_NUMBER}`).

## Result Contract

Every exit path must replace `${LOG_DIR}/perf_result.json`. Do not patch
`run.json`; the orchestrator records this file at the correct single- or
multi-architecture scope.

Every result includes:

```json
{
  "measured": false,
  "outcome": "PERF_NOT_APPLICABLE|PERF_ENV_ERROR|PERF_TEST_FAILED|PERF_OK|PERF_REGRESSED|PERF_NOT_IMPROVED",
  "verdict": "not_measured|neutral|improved|regressed|not_improved",
  "arch": "blackhole|wormhole",
  "goal": "no_regress|improve",
  "test": "perf_<module>.py or null",
  "filter": "pytest -k expression or null",
  "base_commit": "<sha>",
  "run_id": "<sealed run id>",
  "attempt_id": "<sealed attempt id>",
  "requirement_id": "<sealed perf requirement id>",
  "patch_sha256": "<current structured result patch digest>",
  "measurements": {
    "cycle_comparison": {"measured": true},
    "repeatability": {"measured": true, "executions": 3}
  },
  "reason": "concise reason or null"
}
```

Use `null` for the four sealed identity/patch fields when no perf leaf exists.

Include only measurements actually performed. A sealed `repeatability`
requirement means rerunning the fixed-tree selector for at least the leaf's
`minimum_executed` count; one current/baseline comparison cannot satisfy it.
After the fixed-tree run writes its structured result, copy the four identity
and patch fields above from that result/manifest exactly; do not derive or
invent a second patch identity for the measurement document.

Successful comparisons also retain the evaluator's metric, deltas, worst
variant, thread breakdown, baseline/current artifact paths, and queue job IDs
when queued silicon was used.

## Applicability Gate

Create `LOG_DIR`, then return `PERF_NOT_APPLICABLE` without running commands
when any of these is true:

- `TEST_BACKEND != local`;
- `TARGET_ARCH` is not `blackhole` or `wormhole`;
- no existing perf test covers the changed operation.

## Select the Perf Test

Read the fix plan's `## Scope`, `## Implementation`, and `## Test Strategy`.
Inspect the candidate perf module before selecting it; the table is a routing
guide, not evidence that the operation is covered.

| Changed operation | Candidate module |
|---|---|
| SFPU unary | `perf_eltwise_unary_sfpu.py` |
| SFPU binary | `perf_eltwise_binary_sfpu.py` |
| FPU binary | `perf_eltwise_binary_fpu.py` |
| typecast | `perf_eltwise_typecast.py` |
| fused operation | `perf_fused.py` |
| SFPU row-max / SDPA reduce | `perf_sfpu_reduce_row_max.py` / `perf_sfpu_reduce_sdpa.py` |
| math matmul / matmul | `perf_math_matmul.py` / `perf_matmul.py` |
| general reduce | `perf_reduce.py` |
| math / unpack transpose | `perf_math_transpose.py` / `perf_unpack_transpose.py` |
| pack untilize / destination bank | `perf_pack_untilize.py` / `perf_pack_dest_bank.py` |
| fast or unpack tilize | `perf_fast_tilize.py`, `perf_fast_tilize_full.py`, or `perf_unpack_tilize.py` |
| fast untilize | `perf_fast_untilize.py` |
| broadcast / unpack-a broadcast | `perf_eltwise_bcast_col_custom.py` / `perf_unpack_a_bcast_eltwise.py` |

Set:

```bash
PERF_TEST="perf_<module>.py"
PERF_MODULE="${PERF_TEST%.py}"
PERF_K="<exact op expression or empty>"
PERF_OP="<mathop value for CSV filtering or empty>"
```

Use a narrow `-k` expression when the module covers multiple operations.
Keep `PERF_OP` separate because a compound pytest expression is not a valid
CSV `mathop` value.
Confirm the module exists and the selector collects at least one test. If a
shared change affects multiple operations, choose one only when the fix plan
identifies a primary operation or the selected case exercises the same changed
path. Otherwise return `PERF_NOT_APPLICABLE` with the uncovered scope; do not
claim that an arbitrary representative proves no regression.

## Measurement Paths

Run the fixed tree first. Cache the baseline within this run so perf-recovery
retries only remeasure the current tree.

```bash
LLK_ROOT="$WORKTREE_DIR/tt_metal/tt-llk"
RUNNER="$LLK_ROOT/.claude/scripts/run_test.sh"
BASE_COMMIT="$(sg GIT_COMMIT)"
BASE_SHORT="${BASE_COMMIT:0:12}"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"

CURRENT="$LOG_DIR/perf_current_${TARGET_ARCH}_${PERF_MODULE}.post.csv"
BASELINE="$LOG_DIR/perf_baseline_${TARGET_ARCH}_${BASE_SHORT}_${PERF_MODULE}.post.csv"
CURRENT_LOG_DIR="$LOG_DIR/perf_runs/${TARGET_ARCH}/current_${ATTEMPT}"
BASELINE_LOG_DIR="$LOG_DIR/perf_runs/${TARGET_ARCH}/baseline_${BASE_SHORT}"
```

Require `BASE_COMMIT` to resolve as a commit. A missing or invalid base is
`PERF_ENV_ERROR`.

For direct silicon, define the runner once and use it for both trees:

```bash
run_perf_local() {  # $1=tt-llk root, $2=log directory, $3=current|baseline
  local tree="$1" run_log="$2" role="$3"
  local args=(run --worktree "$tree" --arch "$TARGET_ARCH" \
    --test "$PERF_TEST" --maxfail 0 --log-dir "$run_log")
  [ -n "$PERF_K" ] && args+=(--k "$PERF_K")
  if [ "$role" = current ] && [ -n "${CODEGEN_REQUIREMENT_ID:-}" ]; then
    mkdir -p "$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}"
    args+=(--result-json-out "$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}/${CODEGEN_REQUIREMENT_ID}.json")
  fi
  CODEGEN_VERIFICATION_SUITE=perf bash "$RUNNER" "${args[@]}"
}
```

For queued silicon, dispatch the tree and require the queue to copy its fresh
CSV to the explicit destination:

```bash
run_perf_queued() {  # $1=tt-metal tree, $2=destination CSV, $3=log, $4=current|baseline
  local tree="$1" destination="$2" run_log="$3" role="$4"
  local args=(--kind perf --arch "$TARGET_ARCH" --test "$PERF_TEST" \
    --worktree "$tree" --base "$BASE_COMMIT" \
    --session "${HW_TEST_SESSION:-issue-${ISSUE_NUMBER}}-perf-${TARGET_ARCH}-${role}" \
    --timeout 1800 --artifact-out "$destination")
  [ -n "$PERF_K" ] && args+=(--k "$PERF_K")
  if [ "$role" = current ] && [ -n "${CODEGEN_REQUIREMENT_ID:-}" ]; then
    mkdir -p "$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}"
    args+=(--result-json-out "$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}/${CODEGEN_REQUIREMENT_ID}.json")
  fi
  $HW_TEST_DISPATCH_CMD "${args[@]}" 2>&1 | tee -a "$run_log"
  local rc=${PIPESTATUS[0]}
  return "$rc"
}
```

The queue selector is intentionally separate from `PERF_OP`: `--k` narrows
pytest while `PERF_OP` narrows CSV comparison. Require exactly one
`HW_TEST_RESULT arch=${TARGET_ARCH}` marker in each queue log and retain its job
ID in the result and self-log. Current and baseline use distinct session labels
so the queue's warm workspaces cannot overwrite one another.

Runner exits for the fixed tree:

| Exit | Outcome |
|---|---|
| 0 | continue only if a fresh non-empty CSV exists |
| 1 | candidate `PERF_TEST_FAILED`; compare with the baseline outcome |
| 2 | direct silicon only: candidate `PERF_TEST_FAILED`; compare with the baseline outcome |
| 3 or 4 | `PERF_ENV_ERROR` |
| 5 | direct silicon only: candidate `PERF_TEST_FAILED` with hang evidence; compare with baseline |
| other | `PERF_ENV_ERROR` |

Do not send environment errors to the worker. `PERF_TEST_FAILED` is a genuine
failure exposed by the perf workload and must remain distinct for the
orchestrator's retry policy. Attribute exits 1, 2, and 5 to the fix only when
the same perf test succeeds on the baseline. If the baseline also fails or
cannot run, return `PERF_ENV_ERROR`; the measurement cannot distinguish a
pre-existing failure from the fix.

## Measure the Fixed Tree

On direct silicon, the expected CSV is relative to the tree being measured:

```bash
CURRENT_SOURCE="$LLK_ROOT/perf_data/${PERF_MODULE}/${PERF_MODULE}.post.csv"
mkdir -p "$CURRENT_LOG_DIR"
rm -f -- "$CURRENT_SOURCE"

set +e
run_perf_local "$LLK_ROOT" "$CURRENT_LOG_DIR" current
CURRENT_EXIT=$?
set -e
```

On queued silicon, do not run a local producer or consumer. The queue builds
the submitted diff on its silicon runner and copies the result back:

```bash
QUEUE_CURRENT_LOG="$CURRENT_LOG_DIR/dispatch.log"
mkdir -p "$CURRENT_LOG_DIR"
rm -f -- "$CURRENT"
set +e
run_perf_queued "$WORKTREE_DIR" "$CURRENT" "$QUEUE_CURRENT_LOG" current
CURRENT_EXIT=$?
set -e
```

Classify exits 3, 4, and unknown exits immediately. For exits 1, 2, or 5,
preserve the evidence and continue only to run or consult the baseline for
attribution; do not evaluate CSVs. On direct exit zero, require
`CURRENT_SOURCE` to be non-empty and copy it to `CURRENT`. On queued exit zero,
require `CURRENT` to be non-empty; the dispatch command copies only the artifact
produced by that job. A missing result in either path is `PERF_ENV_ERROR`.

## Measure the Baseline Safely

If the cached `BASELINE` is non-empty, reuse it. Otherwise create a unique
detached worktree at `BASE_COMMIT`; never remove the fix from its worktree.

```bash
BASE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/llk-perf-base.XXXXXX")"
BASE_WT="$BASE_ROOT/tree"
BASE_LLK="$BASE_WT/tt_metal/tt-llk"

cleanup_baseline() {
  if [ -n "${BASE_WT:-}" ]; then
    git -C "$WORKTREE_DIR" worktree remove --force "$BASE_WT" 2>/dev/null || true
  fi
  rmdir "$BASE_ROOT" 2>/dev/null || true
}
trap cleanup_baseline EXIT

git -C "$WORKTREE_DIR" worktree add --detach "$BASE_WT" "$BASE_COMMIT"

mkdir -p "$BASELINE_LOG_DIR"

if [ -n "${HW_TEST_DISPATCH_CMD:-}" ]; then
  # The detached base tree produces an empty patch against BASE_COMMIT.
  QUEUE_BASELINE_LOG="$BASELINE_LOG_DIR/dispatch.log"
  rm -f -- "$BASELINE"
  set +e
  run_perf_queued "$BASE_WT" "$BASELINE" "$QUEUE_BASELINE_LOG" baseline
  BASELINE_EXIT=$?
  set -e
else
  # Generated toolchain paths are absent from a clean detached checkout.
  [ -d "$LLK_ROOT/tests/.venv" ] &&
    ln -s "$LLK_ROOT/tests/.venv" "$BASE_LLK/tests/.venv"
  [ -d "$LLK_ROOT/tests/sfpi" ] &&
    ln -s "$LLK_ROOT/tests/sfpi" "$BASE_LLK/tests/sfpi"
  BASELINE_SOURCE="$BASE_LLK/perf_data/${PERF_MODULE}/${PERF_MODULE}.post.csv"
  set +e
  run_perf_local "$BASE_LLK" "$BASELINE_LOG_DIR" baseline
  BASELINE_EXIT=$?
  set -e
fi
```

A baseline run failure does not prove the fix is broken, but it prevents a
comparison. Return `PERF_ENV_ERROR` with the baseline evidence. If the fixed
tree previously exited 1, 2, or 5 and the baseline succeeds, return
`PERF_TEST_FAILED` with both outcomes as evidence. Otherwise, require a
non-empty `BASELINE_SOURCE` and copy it to `BASELINE` on direct silicon, or
require the queue-populated `BASELINE` on queued silicon.

The cleanup trap removes only the unique detached worktree and its now-empty
parent. It must remain active until the baseline run completes.

## Compare

Both CSVs are mandatory:

```bash
[ -s "$CURRENT" ] && [ -s "$BASELINE" ] ||
  { echo "PERF_ENV_ERROR: current or baseline CSV missing"; exit 3; }

eval_args=(
  --current "$CURRENT"
  --baseline "$BASELINE"
  --test "$PERF_TEST"
  --goal "$PERF_GOAL"
  --json-out "$LOG_DIR/perf_result.json"
)
[ -n "$PERF_OP" ] && eval_args+=(--op "$PERF_OP")

set +e
python "$LLK_ROOT/codegen/scripts/perf_eval.py" "${eval_args[@]}"
EVAL_EXIT=$?
set -e
```

`perf_eval.py` is stdlib-only; do not activate a venv for it. After evaluation,
add `outcome`, `arch`, `filter`, `base_commit`, and the two artifact paths to
its JSON object. For the queue route, also add `queue_jobs` with the current and
baseline job IDs parsed from their authoritative marker lines.

Map the evaluator result:

| Verdict | Exit | Outcome |
|---|---|---|
| `improved` or `neutral` | 0 | `PERF_OK` |
| `regressed` | 1 | `PERF_REGRESSED` |
| `not_improved` | 1 | `PERF_NOT_IMPROVED` |
| `no_baseline` or `not_measured` | 2 | `PERF_ENV_ERROR` |

An applicable test that cannot produce comparable rows is not a successful or
not-applicable measurement.

## Return

Return the result outcome, issue number, architecture, goal, selected test and
filter, metric, baseline/current cycles, median and worst deltas, worst variant
and thread breakdown, or the precise reason measurement did not run.

## Limits

- At most two silicon executions on the first attempt: current and baseline.
  Reuse the baseline on perf-recovery retries.
- Never broaden beyond the affected operation or run the full perf suite.
- Never edit the fix or use `git stash`.
- The only Git writes allowed are adding and removing the unique detached
  baseline worktree.

## Self-Log

Create `${LOG_DIR}/agent_perf_tester.md`, or append
`## Perf Attempt — <UTC timestamp>` when it exists. Record applicability, route
(`direct` or `queue`), mapping and scope, exact commands, runner exits and queue
job IDs, baseline commit, evaluator summary, artifact and raw-log paths,
outcome, and first meaningful evidence.
Never discard earlier attempts. If `LOG_DIR` is empty, report that self-logging
was skipped.
