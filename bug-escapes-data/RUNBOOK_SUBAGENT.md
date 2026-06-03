# Bug Escape Detection — Subagent Runbook

**Role:** You are an Opus analysis subagent. Your only job is to find and classify bug escape candidates. You return a structured JSON findings report. You DO NOT dispatch workflows, commit code, push branches, or write to state files.

---

## What You Can Do

- Query Snowflake
- Call GitHub API (read-only: runs, jobs, logs, commits, compare, workflows)
- Analyze CI log content
- Reason about code diffs

## What You Must NOT Do

- Dispatch any GitHub Actions workflow run
- Create, push, or delete any branch or commit
- Write to any state file (campaign-state.json, seen-escapes.json, confirmed-escapes.json)
- Cancel or modify any in-progress run

---

## Definitions

**Vertical bug escape**: A test at layer X fails, and the commit that fixed it touches a layer LOWER than X. fix_layer < test_layer.

**Horizontal bug escape**: A test at layer X fails, and the commit that fixed it touches the SAME layer X. fix_layer == test_layer.

**Layer hierarchy:**
```
1  tt-llk       tt_metal/hw/ckernels/
                tt_metal/hw/firmware/
                tt_metal/hw/inc/
                tt_metal/third_party/umd/
                tt_metal/llrt/hal/
2  tt-metalium  tt_metal/impl/
                tt_metal/core_bindings/
                tt_metal/distributed/
                tt_metal/api/
                tt_metal/fabric/
                tests/tt_metal/
3  ttnn         ttnn/cpp/ttnn/
                ttnn/ttnn/
                tests/ttnn/
4  models       models/
                demos/
```

A **test's layer** is determined by its file path.
A **fix's layer** is the lowest-numbered layer among files changed in the fix commit.

- fix_layer < test_layer → vertical escape
- fix_layer == test_layer → horizontal escape
- fix_layer > test_layer → inverse (not an escape, discard)

---

## Inputs

You will be given:
- The path to `seen-escapes.json` — check this first; skip any escape ID already present
- The path to `confirmed-escapes.json` — for reference
- The campaign mode: **backfill** (90-day window) or **incremental** (2-day window)
- A target count: how many findings to return (default: 1). **When target is 1: stop as soon as you find one qualifying candidate (Opus verdict PROCEED_TO_BISECT). Do not continue scanning further candidates.**

---

## Step 1: Detection (Snowflake)

**⚠️ Snowflake commit boundaries are unreliable — always verify against CI run logs.**

**Unified SQL** — run this to get candidates. Use `DATEADD('day', -90, ...)` for backfill, `DATEADD('day', -2, ...)` for incremental.

```sql
WITH runs AS (
  SELECT
    tc.CICD_TEST_CASE_ID,
    tc.TEST_CASE_NAME AS test_name,
    tc.FILEPATH AS test_filepath,
    p.GIT_COMMIT_HASH AS commit_sha,
    p.PIPELINE_START_TS,
    t.SUCCESS,
    REGEXP_REPLACE(
      REGEXP_REPLACE(t.ERROR_MESSAGE, '0x[0-9a-fA-F]+', 'ADDR'),
      '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:Z.]+', 'TS'
    ) AS norm_signature,
    ROW_NUMBER() OVER (
      PARTITION BY tc.CICD_TEST_CASE_ID
      ORDER BY p.PIPELINE_START_TS ASC
    ) AS rn_asc,
    ROW_NUMBER() OVER (
      PARTITION BY tc.CICD_TEST_CASE_ID, t.SUCCESS
      ORDER BY p.PIPELINE_START_TS ASC
    ) AS rn_by_success
  FROM TTDATASF.SW_TEST.CICD_TEST t
  JOIN TTDATASF.SW_TEST.CICD_TEST_CASE tc ON t.TEST_CASE_ID = tc.CICD_TEST_CASE_ID
  JOIN TTDATASF.SW_TEST.CICD_JOB j ON t.CICD_JOB_ID = j.CICD_JOB_ID
  JOIN TTDATASF.SW_TEST.CICD_PIPELINE p ON j.CICD_PIPELINE_ID = p.CICD_PIPELINE_ID
  WHERE p.PROJECT = 'tt-metal'
    AND p.GIT_BRANCH_NAME = 'main'
    AND p.PIPELINE_START_TS >= DATEADD('day', -90, CURRENT_TIMESTAMP())
    AND j.FAILURE_SIGNATURE NOT LIKE 'InfraErrorV1%'
),
islands AS (
  SELECT *, (rn_asc - rn_by_success) AS island_key
  FROM runs
),
island_summary AS (
  SELECT
    CICD_TEST_CASE_ID,
    SUCCESS,
    island_key,
    COUNT(*) AS streak_length,
    COUNT(DISTINCT norm_signature) AS distinct_sig_count,
    MIN(PIPELINE_START_TS) AS streak_start_ts,
    MAX(PIPELINE_START_TS) AS streak_end_ts,
    MIN_BY(commit_sha, PIPELINE_START_TS) AS streak_first_sha,
    MAX_BY(commit_sha, PIPELINE_START_TS) AS streak_last_sha,
    MAX_BY(norm_signature, PIPELINE_START_TS) AS norm_signature,
    MIN(rn_asc) AS streak_first_rn,
    MAX(rn_asc) AS streak_last_rn
  FROM islands
  GROUP BY 1, 2, 3
),
fail_streaks AS (
  SELECT
    CICD_TEST_CASE_ID, island_key,
    streak_length AS consecutive_fail_count,
    streak_end_ts AS last_fail_ts,
    streak_last_sha AS last_failing_sha,
    streak_last_rn AS last_fail_rn,
    norm_signature
  FROM island_summary
  WHERE SUCCESS = FALSE
    AND streak_length >= 5
    AND distinct_sig_count <= 3
),
pass_streaks AS (
  SELECT
    CICD_TEST_CASE_ID, island_key,
    streak_length AS consecutive_pass_count,
    streak_start_ts AS first_pass_ts,
    streak_first_sha AS first_passing_sha,
    streak_first_rn AS first_pass_rn
  FROM island_summary
  WHERE SUCCESS = TRUE
    AND streak_length >= 5
    AND (streak_length <= 1 OR DATEDIFF('hour', streak_start_ts, streak_end_ts) / NULLIF(streak_length - 1, 0) <= 24)
),
transitions AS (
  SELECT
    fs.CICD_TEST_CASE_ID,
    fs.last_failing_sha,
    ps.first_passing_sha,
    fs.consecutive_fail_count,
    ps.consecutive_pass_count,
    fs.norm_signature,
    fs.last_fail_ts,
    ps.first_pass_ts
  FROM fail_streaks fs
  JOIN pass_streaks ps
    ON fs.CICD_TEST_CASE_ID = ps.CICD_TEST_CASE_ID
    AND ps.first_pass_rn BETWEEN fs.last_fail_rn + 1 AND fs.last_fail_rn + 4
  WHERE DATEDIFF('hour', fs.last_fail_ts, ps.first_pass_ts) <= 168
),
result AS (
  SELECT
    t.CICD_TEST_CASE_ID,
    r.test_name,
    r.test_filepath,
    CASE
      WHEN r.test_filepath LIKE '%models/%' OR r.test_filepath LIKE '%demos/%' THEN 4
      WHEN r.test_filepath LIKE '%ttnn/%' OR r.test_filepath LIKE '%tests/ttnn/%' THEN 3
      WHEN r.test_filepath LIKE '%tt_metal/%' AND r.test_filepath NOT LIKE '%tt_metal/hw/%' THEN 2
      WHEN r.test_filepath LIKE '%tt_metal/hw/%' THEN 1
      ELSE 0
    END AS test_layer,
    t.last_failing_sha,
    t.first_passing_sha,
    t.consecutive_fail_count,
    t.consecutive_pass_count,
    t.last_fail_ts,
    t.first_pass_ts,
    DATEDIFF('hour', t.last_fail_ts, t.first_pass_ts) AS gap_hours,
    t.norm_signature
  FROM transitions t
  JOIN (SELECT DISTINCT CICD_TEST_CASE_ID, test_name, test_filepath FROM runs) r
    ON t.CICD_TEST_CASE_ID = r.CICD_TEST_CASE_ID
  QUALIFY ROW_NUMBER() OVER (PARTITION BY t.CICD_TEST_CASE_ID ORDER BY t.last_fail_ts DESC) = 1
)
SELECT *
FROM result
WHERE test_layer >= 1
ORDER BY last_fail_ts DESC
LIMIT 100;
```

**Escape expiry:** If a candidate's `last_fail_ts` is older than 90 days from today, mark it `expired` in campaign-state.json and skip — CI logs are gone and probes cannot be set up for commits that old.

**Snowflake gotchas:**
- `CICD_TEST` has 5.8B rows — ALWAYS filter by date and LIMIT
- `CICD_PIPELINE.PROJECT = 'tt-metal'` (not `tenstorrent/tt-metal`)
- `CICD_PIPELINE.GIT_COMMIT_HASH` (not `GIT_SHA`)
- LLK assert nightly runs (`J.NAME LIKE '%LLK asserts%'`) — exclude from island analysis
- `j.FAILURE_SIGNATURE NOT LIKE 'InfraErrorV1%'` filters infra failures

For each row: compute escape ID = `{CICD_TEST_CASE_ID}__{last_failing_sha[:8]}`. Check `seen-escapes.json`:

- Status `confirmed*`, `refuted`, `expired`, `skipped_*`: **skip entirely**.
- Status `refuted_wrong_fix`: add to a **retry queue** — process only after all new candidates are exhausted (see priority rule below).
- Not present: new candidate — always prioritize these.

**Priority rule (STRICTLY ENFORCED):**
New candidates come first. Only attempt a `refuted_wrong_fix` retry if the new-candidate queue is completely empty. When you do retry, you MUST attempt different fix commits than the ones previously tried (check the existing candidate entry in campaign-state.json for what was already attempted). If no new commits remain to try, skip.

**⚠️ Retrying `refuted_wrong_fix` is heavily discouraged.** A prior probe already showed BEFORE=PASS for the leading fix hypothesis, meaning the correct commit is non-obvious. New escapes almost always represent more actionable, lower-effort findings. Only retry if explicitly instructed or the new-candidate queue has been fully exhausted for the session.

---

## Step 2: Verify Commit Range (GitHub API)

Snowflake SHAs are unreliable. Before doing any analysis, verify the actual transition:

1. Find the workflow that runs the test (match job name from Snowflake to workflow name):
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows
   ```
2. List runs on main for that workflow near the Snowflake window:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows/{id}/runs?branch=main&per_page=100
   ```
3. For each run whose `head_sha` falls in the Snowflake range, find the relevant job and check its log for explicit PASSED or FAILED for the test name.
4. Establish true `last_failing_sha` and `first_passing_sha`.
5. Record the `run_id` and `job_id` for the last failing run and first passing run — main BrAIn needs these.
6. Also record: `workflow_id`, `workflow_name`, and the **hardware** (runner label: N150, N300, P150, P100, T3K, Galaxy) from the failing job name.

Get the commit range:
```
GET /repos/tenstorrent/tt-metal/compare/{true_last_failing}...{true_first_passing}
```

---

## Step 3: Layer Pre-filter

For each commit in the range:
- Determine `fix_layer` = lowest-numbered layer among files changed
- Classify:
  - `fix_layer < test_layer` → vertical candidate
  - `fix_layer == test_layer` → horizontal candidate
  - `fix_layer > test_layer` → discard

Fetch full diff for qualifying commits:
```
GET /repos/tenstorrent/tt-metal/commits/{sha}
```
Pass the full diff to Opus in Step 4. If >500 lines, pass first 500 and note truncated.

**Commit intent signal**: Note if any candidate commit message contains "fix", "bug", "revert", "restore", "correct". If ZERO fix-intent commits → set `low_confidence_prior: true`.

---

## Step 4: Opus Classification

### 4a: Noise blocklist (skip Step 4 if matched)

Skip to `status: skipped_prefilter` if the failure log contains:
- `TracyAlloc|TracyFree|FrameMarkNamed|FrameMarkStart`
- `tenstorrent_pcie_ioctl|ioctl.*pcie.*fail`
- `FileNotFoundError.*No such file or directory.*libtt_`
- Zero pytest/gtest identifiers AND no `TT_FATAL`/`TT_THROW`

### 4b: Fetch failure log

```
GET /repos/tenstorrent/tt-metal/actions/jobs/{last_failing_job_id}/logs
```
Use first 200 lines of the relevant test step.

### 4c: Vertical Opus prompt

```
You are analyzing a potential vertical bug escape in tenstorrent/tt-metal CI.

A vertical escape means a bug at layer N caused a test at layer M (N < M) to fail.
Trace a SPECIFIC CAUSAL MECHANISM:
  → Which exact function/macro/register/ABI changed in the lower-layer commit?
  → What is the call path from that change to the test's execution path?
  → Why does that change produce EXACTLY the observed error?

If you cannot name all three with specificity, the verdict must be SKIP_UNRELATED.

Test: {test_name}
File: {test_filepath} (layer {test_layer})
Streak: {consecutive_fail_count} failures → {consecutive_pass_count} passes
Failure window: {last_fail_ts} to {first_pass_ts}
Low-confidence prior: {low_confidence_prior}

Cross-layer commits (with full diffs):
{commit_list_with_diffs}

Raw failure log:
{failure_log}

Work through these 5 checks:
CHECK 1 — Error classification: test code vs infrastructure?
CHECK 2 — Determinism: consistent error signature?
CHECK 3 — Causal analysis (HIGH BAR): specific_change + call_path + error_match, all three
CHECK 4 — Range size sanity: >50 commits → lower confidence unless single obvious fix
CHECK 5 — Cross-test scope: many unrelated tests failing → likely infra

Respond with JSON only:
{
  "verdict": "PROCEED_TO_BISECT" | "SKIP_LIKELY_NOISE" | "SKIP_UNRELATED",
  "most_likely_fix_sha": "<sha or null>",
  "confidence": "high" | "medium" | "low",
  "check1_error_type": "test_code" | "infrastructure",
  "check2_determinism": "deterministic" | "likely_flaky" | "unknown",
  "check3_causal_mechanism": "<one sentence or null>",
  "vertical_escape_justification": {
    "specific_change": "<exact function/macro/register changed, or null>",
    "call_path": "<how test reaches changed code, or null>",
    "error_match": "<why this change produces exactly the observed error, or null>"
  },
  "reasoning": "<2-3 sentences>"
}
```

Post-Opus validation: If `specific_change`, `call_path`, or `error_match` is null, generic, or uses "might/possibly/could" → override to SKIP_UNRELATED.

MANDATORY layer check: Confirm `most_likely_fix_sha` has `fix_layer < test_layer`. Same-layer → horizontal. Higher-layer → discard.

### 4d: Horizontal Opus prompt

```
You are analyzing a potential horizontal bug escape in tenstorrent/tt-metal CI.

A horizontal escape means a bug at layer N caused a test at the SAME layer N to fail.
The fix and test are in the same codebase layer, so the causal link is typically direct.

Required analysis:
  (a) same_component: Does the commit modify code in the same component/module the test exercises?
  (b) plausible_fix: Does the diff address a failure mode consistent with the error signature?
  (c) temporal_fit: Is the commit within the failure-to-pass transition window?

Test: {test_name}
File: {test_filepath} (layer {test_layer})
Streak: {consecutive_fail_count} failures → {consecutive_pass_count} passes
Failure window: {last_fail_ts} to {first_pass_ts}

Same-layer commits in range (with diffs):
{commit_list_with_diffs}

Raw failure log:
{failure_log}

Respond with JSON only:
{
  "verdict": "PROCEED_TO_BISECT" | "SKIP_LIKELY_NOISE" | "SKIP_UNRELATED",
  "most_likely_fix_sha": "<sha or null>",
  "confidence": "high" | "medium" | "low",
  "same_component": "<what component/module the commit and test share, or null>",
  "plausible_fix": "<one sentence on why the diff addresses the failure mode, or null>",
  "temporal_fit": "yes" | "no",
  "reasoning": "<2-3 sentences>"
}
```

If `same_component` is null or diff has zero overlap with test's component → SKIP_UNRELATED.

---

## Step 5: PR-Linked CI Proof Check

For the `most_likely_fix_sha` from Opus, check for an associated PR:
```
gh pr list --search {sha} --state merged
```

If a PR exists, read it and check whether the PR body or comments contain a link to a CI run where the specific test explicitly passed. If yes, record:
- `pr_number`
- `ci_run_url` (the run link from the PR)
- `test_pass_evidence` (the exact line from the PR proving the test passed)
- Set `action_required: "pr_ci_proof_check"`

Note: You are providing the evidence — the verify agent will run `verify-pr-proof.py` to formally confirm it. Just record what you found.

---

## Step 6: Existing CI Log Check (Narrow the Range Without Probing)

Before concluding `action_required: "probe_verification"`, check if existing CI runs at key commits can narrow the range:

```
GET /repos/tenstorrent/tt-metal/actions/runs?head_sha={midpoint_sha}
```

For each midpoint in the binary search space, if a run exists at that SHA on the relevant workflow, check its job log for the test. Record in `bisect_log_reads`. If existing reads narrow to a single commit, set `fix_commit_sha` and reduce `candidate_fix_commits` accordingly.

---

## Output Format

Return a single JSON object. Do not wrap in markdown. Do not add prose outside the JSON.

```json
{
  "session_summary": "<1-2 sentences: how many candidates checked, how many found, how many skipped>",
  "findings": [
    {
      "escape_id": "{CICD_TEST_CASE_ID}__{last_failing_sha[:8]}",
      "escape_type": "vertical|horizontal",
      "test_name": "...",
      "test_filepath": "...",
      "test_layer": 3,
      "fix_layer": 3,
      "last_failing_sha": "...",
      "first_passing_sha": "...",
      "consecutive_fail_count": 18,
      "consecutive_pass_count": 5,
      "gap_hours": 19,
      "last_failure_run_id": "...",
      "last_failure_job_id": "...",
      "first_success_run_id": "...",
      "first_success_job_id": "...",
      "workflow_id": "...",
      "workflow_name": "...",
      "hardware": "N300",
      "architecture": "wormhole_b0",
      "fix_commit_sha": "...",
      "fix_commit_message": "...",
      "candidate_fix_commits": ["sha1", "sha2"],
      "fix_pr": "42760",
      "action_required": "probe_verification|pr_ci_proof_check",
      "confirmation_method": "bisect|pr_ci_proof|null",
      "confidence": "high|medium|low",
      "pr_proof_evidence": null,
      "bisect_log_reads": [],
      "opus_reasoning": "..."
    }
  ],
  "rejected": [
    {
      "escape_id": "...",
      "reason": "..."
    }
  ]
}
```

**Field notes:**
- `hardware`: the runner label from the failing job (N150, N300, P150, P100, T3K, Galaxy)
- `architecture`: `wormhole_b0` for N150/N300/T3K/Galaxy, `blackhole` for P150/P100/P150b
- `action_required`:
  - `probe_verification` → verify agent dispatches BEFORE/AFTER probes at `fix_commit_sha^` and `fix_commit_sha`
  - `pr_ci_proof_check` → verify agent runs `verify-pr-proof.py` first, then may run probes
- `fix_commit_sha`: single best guess (required if `action_required` is `probe_verification`); null if range can't be narrowed
- `candidate_fix_commits`: all same-layer or cross-layer commits in range (always populate)
