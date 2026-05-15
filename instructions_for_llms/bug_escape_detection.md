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
    tc.TEST_CASE_NAME  AS test_name,
    tc.FILEPATH        AS test_filepath,
    p.GIT_COMMIT_HASH  AS commit_sha,
    p.PIPELINE_START_TS,
    t.SUCCESS          AS test_success,
    -- Normalize failure signature: strip timestamps, addresses, runner paths
    REGEXP_REPLACE(
      REGEXP_REPLACE(t.ERROR_MESSAGE, '0x[0-9a-fA-F]+', 'ADDR'),
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
    MAX(CASE WHEN test_success = FALSE THEN commit_sha END)
      OVER (PARTITION BY CICD_TEST_CASE_ID ORDER BY rn
            ROWS BETWEEN CURRENT ROW AND 4 FOLLOWING) AS last_failing_sha,
    MIN(CASE WHEN test_success = TRUE THEN commit_sha END)
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

### Step 4: Find the Fix Commit

**WARNING: Do NOT use `bisect-dispatch.yaml` for this step.**
That workflow finds *breaking* commits (good=old passing, bad=new failing). We want the *fix*
commit. Using it here will either error ("good is not ancestor of bad") or find the wrong commit.

**Step 4a — Snowflake oracle (free, try this first):**

Query for all runs of the test between the last failure and first pass timestamps:

```sql
SELECT p.GIT_COMMIT_HASH, p.PIPELINE_START_TS, t.SUCCESS
FROM TTDATASF.SW_TEST.CICD_TEST t
JOIN TTDATASF.SW_TEST.CICD_JOB j ON t.CICD_JOB_ID = j.CICD_JOB_ID
JOIN TTDATASF.SW_TEST.CICD_PIPELINE p ON j.CICD_PIPELINE_ID = p.CICD_PIPELINE_ID
WHERE t.TEST_CASE_ID = {test_case_id}
  AND p.PIPELINE_START_TS BETWEEN '{last_fail_ts}' AND '{first_pass_ts}'
ORDER BY p.PIPELINE_START_TS ASC;
```

If intermediate runs exist, the last FAIL → first PASS transition identifies the fix commit.
No hardware needed. Proceed directly to Step 5 with that commit.

**Step 4b — Hardware search when Snowflake has no intermediate data:**

Choose between Method A (binary search) and Method B (hypothesis testing):

- If Opus named a **specific commit with HIGH confidence** → try **Method B first** (fewer dispatches)
- If Opus confidence is LOW/MEDIUM, or Method B fails → use **Method A** (guaranteed to find it)
- If there are ≤ 4 commits in the range → use **Method B** with all cross-layer commits as candidates

---

#### Method A — Binary Search

Binary search dispatches O(log₂ N) single-commit runs to narrow the range until the fix commit
is identified. Each dispatch is a **single branch at a midpoint** — NOT a BEFORE/AFTER pair.
Do NOT create BEFORE/AFTER branches during binary search; that is only done in Step 5.

**Algorithm:**

1. Get commit list: `GET /repos/tenstorrent/tt-metal/compare/{last_failing}...{first_passing}` → N commits (chronological order).
2. If N ≤ 1: that commit is the fix. Save to `campaign-state.json` and proceed to Step 5.
3. Pick midpoint M = commit at index N//2.
4. Create a branch at M. Dispatch a **pruned** run (see Step 5 pruning instructions). Save run ID to `campaign-state.json`.
5. Wait for completion:
   - **PASS** → fix is in the earlier half. New range = commits 0…N//2 (inclusive of M).
   - **FAIL** → fix is in the later half. New range = commits N//2…N-1.
   - **TIMEOUT** → write `inconclusive_timeout` to `campaign-state.json`. Stop for tonight. Resume from saved state next night.
6. If new range has 1 commit remaining: that is the fix commit F. Proceed to Step 5.
7. Otherwise go to step 3 with the narrowed range.

Save the identified fix commit SHA and full bisect history to `campaign-state.json`.

---

#### Method B — Hypothesis Testing

Use when Opus has a strong candidate or the range is small (≤ 4 cross-layer commits).

**Algorithm:**

Build a hypothesis list: all cross-layer commits in `[last_failing...first_passing]` that touch
a layer lower than the test layer. Order them with Opus's top candidate first.

For each hypothesis H in order:
1. Create `brain/escape-before-{escape_id}` at `H^` (parent of H).
2. Create `brain/escape-after-{escape_id}` at `H`.
3. Dispatch **pruned** runs on both branches in parallel (see Step 5 pruning instructions). Save run IDs.
4. Wait for both to complete. Evaluate:
   - **BEFORE=FAIL + AFTER=PASS** → H is the fix commit. **Confirmed. Proceed to Step 6** (skip Step 5).
   - **BEFORE=FAIL + AFTER=FAIL** → H is not the fix. Delete branches. Continue to next H.
   - **BEFORE=PASS + AFTER=anything** → The test was already passing before H; the fix boundary is earlier than the assumed window. Do NOT refute yet — re-examine the Snowflake data to find an earlier `last_failing_sha`, then restart Method B with the corrected range.
   - **Either TIMEOUT** → write `inconclusive_timeout` to `campaign-state.json`. Stop for tonight. Resume next night.

**End condition — refute:** When ALL commits in the hypothesis list have been tested (BEFORE=FAIL,
AFTER=FAIL for each) and no BEFORE=PASS occurred, the escape cannot be attributed to any
cross-layer change in this window. Write `status: "refuted_hypothesis_exhausted"` to
`seen-escapes.json`.

**Do NOT refute after a single failing hypothesis** unless it was the only cross-layer commit
in the range. Test every plausible candidate before concluding nothing fixed it.

---

### Step 5: Final Confirmation (Method A only) / Pruning Instructions (all dispatches)

**If you used Method A (binary search),** you now have a single identified fix commit F.
Run the two-branch confirmation:

1. Create `brain/escape-before-{escape_id}` at `F^` (parent of F).
2. Create `brain/escape-after-{escape_id}` at `F`.
3. Dispatch both in parallel with pruning (see below). Save run IDs.
4. Wait for completion. Verdict:
   - **BEFORE=FAIL + AFTER=PASS** → **confirmed escape** ✅ Proceed to Step 6.
   - **BEFORE=FAIL + AFTER=FAIL** → Binary search identified the wrong commit. Delete branches. Restart binary search excluding F. If this happens twice, log as `inconclusive_bisect_error` and escalate to human.
   - **BEFORE=PASS + AFTER=anything** → Unexpected: the test was already fixed before F. Do NOT auto-refute. Check `campaign-state.json` for bisect history and re-examine the starting range. Escalate to human if range was correct.
   - **Either TIMEOUT** → `inconclusive_timeout`. Retry next night.

**If you used Method B (hypothesis testing),** confirmation is built in — there is no separate Step 5. Skip directly to Step 6 on confirmation.

---

**Pruning instructions (mandatory for ALL dispatches in Steps 4b and 5):**

Before dispatching any verification run:

1. Look up the correct workflow. Do NOT guess from memory. Query Snowflake:
```sql
SELECT DISTINCT j.NAME, j.GITHUB_JOB_LINK
FROM TTDATASF.SW_TEST.CICD_TEST t
JOIN TTDATASF.SW_TEST.CICD_JOB j ON t.CICD_JOB_ID = j.CICD_JOB_ID
WHERE t.TEST_CASE_ID = {test_case_id}
  AND j.JOB_START_TS >= DATEADD('day', -30, CURRENT_TIMESTAMP())
LIMIT 3;
```
`j.NAME` gives the workflow and SKU (e.g. `models-unit-tests / Qwen3-32B unit tests (Galaxy) [wh_galaxy_perf]`).
Cross-reference with `GET /repos/tenstorrent/tt-metal/actions/workflows` to get the workflow file and ID.

2. Edit the `TESTS_YAML_PATH` file on the branch to contain only the failing test group,
narrowed to the specific test function. Commit and push. Then dispatch.

Dispatching without pruning wastes hardware time on unrelated tests and is prohibited.

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

**This page is the single source of truth for campaign status and run links.**
Every GHA run URL dispatched during this campaign must be recorded on this page immediately
after dispatch — not just at the end. If someone asks "where are we?", the answer is always
"check Confluence" — not a Slack status report.

**Page format: one flat table** containing ONLY candidates that reached hardware.
A candidate appears in this table if and only if Opus returned `PROCEED_TO_BISECT`.
Do NOT include layer-filtered candidates — they were eliminated before any hardware ran.
Do NOT include Opus-dismissed candidates (`SKIP_LIKELY_NOISE`, `SKIP_UNRELATED`).
Never use separate sections per status.

**Use the native Confluence storage format table** with `data-layout="full-width"` and
`data-table-display-mode="default"`. Set explicit column widths in `<colgroup>` (130–240px each)
to prevent columns from being squished. Do NOT use the HTML macro — it is not enabled.

**Table columns (required, in this order):**

| Escape ID | Test File | Job | HW | Last Fail | Last Fail Commit | First Pass | First Pass Commit | Fail Rate | Runs Dispatched | Opus Verdict | Opus Reasoning | Status |

Column notes:
- `Escape ID`: `{test_case_id}__{last_failing_sha[:8]}`
- `Test File`: basename + layer number, e.g. `test_foo.py (layer 4)`
- `Job`: full GHA job name from Snowflake `CICD_JOB.NAME`
- `HW`: runner type, e.g. `Galaxy (WH)`, `LLMBox (WH)`, `LoudBox (BH)`, `T3000 (WH)`
- `Last Fail` / `First Pass`: date only (YYYY-MM-DD), linked commit SHA
- `Runs Dispatched`: all GHA run links with result, e.g. `BEFORE [12345](url) → PASS; AFTER [12346](url) → FAIL`
- `Opus Verdict`: `PROCEED_TO_BISECT` / `N/A` (layer filtered — Opus was not run)
- `Opus Reasoning`: 1–2 sentence summary (empty/`—` for layer-filtered rows)
- `Status`: one of `⏳ Pending`, `🔍 Under Investigation`, `✅ Confirmed`, `❌ Refuted`, `⏸ Inconclusive`

**Update the table after EVERY state transition:**
- Candidate detected → add row immediately (even before Opus, with status TBD)
- After layer filter → update Status to `❌ Layer filter`, fill Opus columns as N/A
- After Opus → update Opus Verdict/Reasoning and Status
- After any dispatch → add run link to Runs Dispatched column immediately
- After run completes → append result (PASS/FAIL) to that run link inline
- After final verdict → update Status

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

---

## Known Pitfalls

### 1. Using the bisect workflow to find fix commits

**Error**: `bisect-dispatch.yaml` finds breaking commits (good=old, bad=new). The campaign
needs fix commits, which are in the opposite direction. The workflow will error or find the
wrong commit if used for fix commit search.

**Fix**: Use the Snowflake oracle (Step 4a), binary search (Step 4b Method A), or hypothesis
testing (Step 4b Method B).

### 4. Refuting a bug escape after testing only one hypothesis

**Error**: Agent tests Opus's top candidate commit, gets BEFORE=FAIL+AFTER=FAIL, and marks the
escape as `refuted`. The other cross-layer commits in the range were never tested.

**Fix**: Method B has a defined end condition — refute ONLY after ALL plausible cross-layer
commits have been tested. A single failing hypothesis does not refute the escape.

### 2. Guessing the verification workflow from memory

**Error**: Assuming which GitHub Actions workflow runs a given test without checking.
Different tests run in different workflows — guessing leads to dispatching a workflow that
doesn't contain the test at all.

**Fix**: Always query `CICD_JOB.NAME` from Snowflake for a recent run of the test, then
find the corresponding workflow file via the GitHub workflows API. See Step 5.

### 3. Skipping test matrix pruning before verification dispatch

**Error**: Dispatching the full workflow instead of narrowing to the one failing test.
This wastes Galaxy hardware on dozens of unrelated tests and makes results harder to interpret.

**Fix**: Always edit the `TESTS_YAML_PATH` file on both branches to contain only the failing
test group before dispatching. This is mandatory, not optional.
