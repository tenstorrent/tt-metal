# Bug Escape Detection Runbook

This file is the complete operating procedure for BrAIn's bug escape detection pipeline.
No GitHub Actions scaffolding needed beyond the pre-existing workflows that run hardware tests.

---

## Definitions

**Vertical bug escape**: A test at layer X fails, and the commit that fixed it touches a layer
LOWER than X. The bug "escaped" downward past CI gates.

**Layer hierarchy** (lower number = more foundational):
```
1  tt-llk       tt_metal/hw/ckernels/
                tt_metal/hw/firmware/
                tt_metal/hw/inc/
2  tt-metalium  tt_metal/impl/
                tt_metal/core_bindings/
                tt_metal/distributed/
                tt_metal/api/
3  ttnn         ttnn/cpp/ttnn/
                ttnn/ttnn/
4  models       models/
```

A **test's layer** is determined by its file path in pytest output.
A **fix's layer** is the lowest-numbered layer among files changed in the fix commit.

An escape requires: fix_layer < test_layer.

---

## State Files (all in /workspace/group/bug-escapes-data/)

| File | Purpose |
|------|---------|
| `seen-escapes.json` | Set of escape IDs already analyzed. Prevents re-analysis. |
| `confirmed-escapes.json` | All confirmed escapes (source of truth for Confluence). |
| `campaign-state.json` | Active campaign: candidates, bisect run IDs, current step. |
| `RUNBOOK.md` | This file. |

**Escape ID format**: `{test_case_id}__{last_failing_sha[:8]}`
Example: `42847291__1ee8c3ca`

Before touching any candidate, check `seen-escapes.json`. If ID is present, skip entirely.
After any verdict (confirmed, refuted, unverifiable), write the ID to `seen-escapes.json`.

---

## Two Modes

### Mode A — Backfill
Run once manually. Scans last 60 days. Full analysis pipeline.

### Mode B — Incremental
Scheduled daily at 00:15Z via nanoclaw. Scans last 1 day only.
Lighter threshold (2+ consecutive failures → 3+ consecutive passes).
Skip Opus pre-classification if layer pre-filter already rules it out.

---

## Pipeline Steps

### Step 1: Detection (Snowflake)

Run the following SQL. Adjust `DATEADD` for backfill (60 days) vs incremental (1 day).

```sql
WITH ranked AS (
  SELECT
    tc.CICD_TEST_CASE_ID,
    tc.NAME            AS test_name,
    tc.FILEPATH        AS test_filepath,
    p.GIT_COMMIT_SHA   AS commit_sha,
    p.PIPELINE_START_TS,
    t.TEST_SUCCESS,
    -- Normalize failure signature: strip timestamps, addresses, runner paths
    REGEXP_REPLACE(
      REGEXP_REPLACE(t.FAILURE_REASON, '0x[0-9a-fA-F]+', 'ADDR'),
      '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:Z.]+', 'TS'
    ) AS norm_signature,
    ROW_NUMBER() OVER (
      PARTITION BY tc.CICD_TEST_CASE_ID
      ORDER BY p.PIPELINE_START_TS DESC
    ) AS rn
  FROM TTDATASF.SW_TEST.CICD_TEST t
  JOIN TTDATASF.SW_TEST.CICD_TEST_CASE tc ON t.TEST_CASE_ID = tc.CICD_TEST_CASE_ID
  JOIN TTDATASF.SW_TEST.CICD_JOB j ON t.CICD_JOB_ID = j.CICD_JOB_ID
  JOIN TTDATASF.SW_TEST.CICD_PIPELINE p ON j.CICD_PIPELINE_ID = p.CICD_PIPELINE_ID
  WHERE p.PROJECT = 'tt-metal'
    AND p.GIT_BRANCH_NAME = 'main'
    AND p.PIPELINE_START_TS >= DATEADD('day', -1, CURRENT_TIMESTAMP())  -- adjust for mode
    AND j.FAILURE_SIGNATURE NOT LIKE 'InfraErrorV1%'
),
-- Find transitions: last N failures followed by M passes
transitions AS (
  SELECT
    CICD_TEST_CASE_ID,
    test_name,
    test_filepath,
    norm_signature,
    -- Collect consecutive pass/fail blocks
    MAX(CASE WHEN TEST_SUCCESS = FALSE THEN commit_sha END)
      OVER (PARTITION BY CICD_TEST_CASE_ID ORDER BY rn
            ROWS BETWEEN CURRENT ROW AND 4 FOLLOWING) AS last_failing_sha,
    MIN(CASE WHEN TEST_SUCCESS = TRUE THEN commit_sha END)
      OVER (PARTITION BY CICD_TEST_CASE_ID ORDER BY rn
            ROWS BETWEEN CURRENT ROW AND 4 FOLLOWING) AS first_passing_sha
  FROM ranked
)
SELECT DISTINCT
  CICD_TEST_CASE_ID,
  test_name,
  test_filepath,
  last_failing_sha,
  first_passing_sha,
  norm_signature
FROM transitions
WHERE last_failing_sha IS NOT NULL
  AND first_passing_sha IS NOT NULL
LIMIT 50;
```

NOTE: This SQL is a starting sketch. The consecutive-run counting logic may need refinement
after seeing real data. Adjust the window size and consecutive count as needed.

For each row:
- Compute escape ID: `{CICD_TEST_CASE_ID}__{last_failing_sha[:8]}`
- Check `seen-escapes.json`. If present, skip.
- Otherwise, add to candidate list.

---

### Step 2: Layer Pre-filter (GitHub API)

For each candidate, get commits between `last_failing_sha` and `first_passing_sha`:

```
GET /repos/tenstorrent/tt-metal/compare/{last_failing_sha}...{first_passing_sha}
```

For each commit in the response, check files changed. Determine the lowest layer touched.

Determine test layer from `test_filepath`:
- starts with `models/` → layer 4
- starts with `ttnn/` → layer 3
- starts with `tt_metal/impl/` or `tt_metal/api/` or `tt_metal/distributed/` → layer 2
- starts with `tt_metal/hw/` → layer 1
- other (CI, docs, cmake) → skip (not a meaningful test layer)

If NO commit in the range touches a layer lower than the test layer → **discard**. Not an escape.
If any commit does → proceed to Step 3.

---

### Step 3: Opus Pre-classification

Fetch the raw GHA job log for the `last_failing_sha` run:
```
GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs
```
(Get `job_id` from the pipeline data in Snowflake or via the runs API for the relevant commit.)

Provide Opus with:
1. The raw failure log (or first 200 lines of the relevant step)
2. The list of cross-layer commits from Step 2 (commit SHA, message, files changed)
3. The test filepath and name

Ask Opus to answer:
```
You are analyzing a potential vertical bug escape in tenstorrent/tt-metal CI.

A test at layer {test_layer} ({test_filepath}) failed consistently then started passing.
The commits between last failure and first pass include these cross-layer changes:
{commit_list_with_diffs}

Raw failure log:
{failure_log}

Answer these questions:
1. Is the failure signature consistent with a genuine code regression (not infra flake,
   not resource exhaustion, not a known intermittent issue)?
2. Is there a plausible causal connection between any of the cross-layer commits and
   this specific test failure? Explain which commit and why.
3. Verdict: PROCEED_TO_BISECT / SKIP_LIKELY_NOISE / SKIP_UNRELATED

If PROCEED_TO_BISECT, name the most likely fix commit SHA.
```

If verdict is not PROCEED_TO_BISECT → write escape ID to `seen-escapes.json` with
`status: "skipped_prefilter"` and stop.

---

### Step 4: Bisect (after midnight, ~00:15Z)

Use the existing bisect workflow:
https://github.com/tenstorrent/tt-metal/actions/workflows/bisect-dispatch.yaml

Dispatch via GitHub API:
```
POST /repos/tenstorrent/tt-metal/actions/workflows/bisect-dispatch.yaml/dispatches
{
  "ref": "main",
  "inputs": {
    "good_commit": "{first_passing_sha}",
    "bad_commit":  "{last_failing_sha}",
    "arch":        "blackhole",
    "command":     "{test command, e.g. pytest tests/foo.py::test_bar -x}",
    "timeout":     "60",
    "retries":     "2",
    "download_artifacts": "true"
  }
}
```

Extract the test command from the failure log or from known workflow-to-test mappings.

Save bisect run ID to `campaign-state.json`.

Poll every 15 minutes until terminal state (succeeded / failed / timed_out).
Parse the final log message from the "Run Git Bisect" step — it contains the culprit commit SHA.

If timed out → write `status: "inconclusive_bisect_timeout"` to `seen-escapes.json`, stop.

---

### Step 5: Verification

Create two branches in the worktree:
- `brain/escape-before-{escape_id}` at `git checkout {bisect_result}^` (parent)
- `brain/escape-after-{escape_id}` at `git checkout {bisect_result}` (fix commit)

Dispatch the appropriate test workflow (blackhole-e2e-tests.yaml or blackhole-demo-tests.yaml)
for each branch. Run them in parallel.

Wait for both to complete.

Verdict logic:
- BEFORE=FAIL + AFTER=PASS → **confirmed escape** ✅
- BEFORE=FAIL + AFTER=FAIL → **refuted** (fix didn't address this test)
- BEFORE=PASS + AFTER=anything → **refuted** (test wasn't failing at that commit)
- Either times out → **inconclusive_timeout** (retry next night)

Write verdict + all run IDs to `campaign-state.json`.

---

### Step 6: Record Confirmed Escape

Append to `confirmed-escapes.json`:
```json
{
  "escape_id": "...",
  "test_name": "...",
  "test_filepath": "...",
  "test_layer": "...",
  "fix_commit_sha": "...",
  "fix_commit_message": "...",
  "fix_layer": "...",
  "last_failure_run_id": "...",
  "first_success_run_id": "...",
  "bisect_run_id": "...",
  "before_run_id": "...",
  "after_run_id": "...",
  "reasoning": "...",  // Opus's explanation from Step 3
  "confirmed_at": "2026-05-14T..."
}
```

Write escape ID to `seen-escapes.json` with `status: "confirmed"`.

Delete BEFORE/AFTER branches from GitHub.

---

### Step 7: Confluence Updates (happens at MULTIPLE points, not just at the end)

Page ID: 2424012846
URL: https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/2424012846/bug+escapes

The page has four sections. Move each escape between sections as its status changes.
Never delete an entry — only move it and update its status field.

**After Step 3 (Opus pre-classification passes) → add to "🔍 Under Investigation":**
```
### Candidate: {test_name[:60]}

- *Test*: {test_name}
- *Test layer*: {test_layer}
- *Fix layer (suspected)*: {suspected_fix_layer}
- *Last failure*: [Run {last_failure_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{last_failure_run_id})
- *First success*: [Run {first_success_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{first_success_run_id})
- *Opus reasoning*: {opus_reasoning}
- *Status*: Under Investigation — bisect dispatched {date}
```

**After Step 4 (bisect completes) → update the entry, still in "Under Investigation":**
Add fix commit and bisect proof link. Update status to "Under Investigation — verification dispatched {date}".

**After Step 5 (verification complete):**
- CONFIRMED → move entry to "✅ Confirmed Escapes", update status, add before/after run links
- REFUTED → move entry to "❌ Refuted", update status, add explanation of why it was refuted
- INCONCLUSIVE_TIMEOUT → move entry to "⏳ Inconclusive", update status, note retry scheduled

**Entry format once confirmed:**
```
### Escape #{N}: {fix_commit_message[:60]}

- *Test*: {test_name}
- *Test layer*: {test_layer}
- *Fix layer*: {fix_layer}
- *Type*: {fix_layer} → {test_layer} escape
- *Last failure*: [Run {last_failure_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{last_failure_run_id})
- *First success*: [Run {first_success_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{first_success_run_id})
- *Fix commit*: [{fix_commit_sha[:8]}](https://github.com/tenstorrent/tt-metal/commit/{fix_commit_sha})
- *Bisect proof*: [Run {bisect_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{bisect_run_id})
- *Verification*: BEFORE [Run {before_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{before_run_id}) | AFTER [Run {after_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{after_run_id})
- *Reasoning*: {reasoning}
- *Status*: Confirmed {date}
```

Always fetch the current page content before updating so you don't clobber concurrent writes.

---

## Incremental Mode Schedule

```
00:15Z  Step 1 (Snowflake, last 1 day)
00:20Z  Steps 2-3 (layer filter + Opus pre-classification, parallelized)
00:30Z  Step 4 (bisect dispatch for survivors)
03:30Z  Step 4 poll / parse bisect results
04:00Z  Step 5 (verification dispatch)
06:30Z  Step 5 poll / parse verification results
07:00Z  Steps 6-7 (record + Confluence update)
07:05Z  Post summary to Slack #ai-sw-infra
```

All dispatches happen in the nightly window when P300-viommu, LoudBox, and LLMBox runners
are online. The watchdog task at 07:00Z checks for stalled campaigns (no progress in 3+ hours)
and posts an alert.

---

## Watchdog Task

Run at 07:00Z. Check `campaign-state.json`:
- Any candidate stuck in `step: "bisect"` with `dispatch_time` > 4 hours ago → log as `inconclusive_timeout`
- Any candidate stuck in `step: "verification"` with `dispatch_time` > 3 hours ago → log as `inconclusive_timeout`
- Post campaign summary to Slack regardless

---

## Subagent Rules (MANDATORY)

These rules exist so the campaign can always be interrupted by a human at any time.

- ALL subagents MUST be launched with `run_in_background: true`. No blocking subagent calls.
- Subagents MUST NOT commit or push code to any repository.
- Subagents MUST NOT dispatch GitHub Actions workflow runs (no `workflow_dispatch` API calls).
- Subagents MUST NOT cancel or modify in-progress workflow runs.

Only the main BrAIn session may: commit/push code, dispatch workflows, cancel runs.
Subagents may only: read data (Snowflake, GitHub API, logs), analyze, and return findings.

---

## What NOT to Do

- Do NOT dispatch two bisects simultaneously for the same hardware type (contends for runners)
- Do NOT start verification without a completed bisect result
- Do NOT skip writing to `seen-escapes.json` even for fast refutals — prevents re-analysis
- Do NOT delete branches until verification is complete
- Do NOT update Confluence with unconfirmed candidates

---

## Backfill Mode Notes

For the initial 60-day scan:
1. Run Step 1 with 60-day window. Expect 30-80 candidates.
2. Layer pre-filter eliminates ~50%. Run in one pass (no hardware needed).
3. Opus pre-classification: batch process, 8 candidates at a time max (rate limit).
4. Bisect: queue candidates by hardware type, dispatch in nightly windows over multiple nights.
   - Stagger: 3 bisects per night max to avoid runner contention.
   - Multi-night campaign: 5-10 nights to process all survivors.
5. Verification: same nightly cadence.
6. Confluence page seeded with all confirmed escapes from the backfill.

After backfill, switch to incremental mode. The `seen-escapes.json` file prevents re-analysis.
