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

### seen-escapes.json — canonical status values

Every entry is a JSON object with a `status` field. Valid values (exhaustive — do NOT invent new ones):

| Status | Set when |
|--------|----------|
| `pending` | Candidate found but not yet processed |
| `skipped_layer_filter` | Step 2 layer pre-filter: no cross-layer commits in range |
| `skipped_same_layer` | Step 2: fix commit touches same layer as test (not vertical) |
| `skipped_same_layer_api_change` | Step 2: cross-layer commit is a non-bug API change |
| `skipped_flaky_same_sha` | Step 2: test flips pass/fail on the same commit SHA — flaky |
| `skipped_likely_noise` | Step 3a: noise blocklist match (infra pattern, no test output) |
| `skipped_likely_noise_perf` | Step 3a: failure is a perf regression, not a correctness bug |
| `skipped_prefilter` | Step 3a: pre-Opus quick filter caught it |
| `skipped_prefilter_opus` | Step 3c: Opus verdict was SKIP_LIKELY_NOISE |
| `skipped_unrelated` | Step 3c: Opus verdict was SKIP_UNRELATED |
| `skipped_vague_justification` | Step 3c: Opus returned PROCEED but justification fields were generic/hedged |
| `skipped_not_vertical` | Step 3c post-check: suspected fix layer ≥ test layer |
| `false_positive` | Bisect converged to [skip ci] or CODEOWNERS-only commit |
| `refuted_false_positive` | Same as above — use `false_positive` (legacy alias) |
| `inconclusive_bisect_timeout` | Step 4: bisect run timed out |
| `inconclusive_api_error` | Step 4 or 5: 3 consecutive GitHub API failures |
| `inconclusive` | Step 5: verification timed out or was cancelled |
| `refuted` | Step 5: BEFORE=PASS (wasn't failing) or BEFORE=FAIL+AFTER=FAIL (fix didn't fix it) |
| `confirmed` | Step 5: BEFORE=FAIL + AFTER=PASS |

### campaign-state.json schema

Top-level structure (array-based, NOT keyed by escape_id):

```json
{
  "campaign_start": "2026-05-15",
  "bisects_in_progress": [
    {
      "escape_id": "42847291__1ee8c3ca",
      "hw": "BH",
      "runner_label": "BH-llmbox",
      "test_name": "test_foo::test_bar",
      "test_filepath": "tests/models/...",
      "test_command": "pytest tests/models/... -v",
      "last_failing_sha": "...",
      "first_passing_sha": "...",
      "total_commits_in_range": 12,
      "manual_bisect": {
        "low": 0,
        "high": 11,
        "runs": [
          {"sha": "...", "run_id": 12345678, "result": "FAIL", "dispatched_at": "2026-05-14T02:00:00Z"}
        ]
      },
      "step": "bisect_mid1_dispatched",
      "fix_commit": "abc123",
      "fix_commit_full": "abc123def456...",
      "fix_pr": 12345,
      "fix_pr_title": "...",
      "fix_layer": 2,
      "test_layer": 4,
      "verification_runs": {
        "before": {"run_id": null, "result": null},
        "after": {"run_id": null, "result": null}
      },
      "status": "bisect_in_progress"
    }
  ],
  "verification_in_progress": [
    {
      "escape_id": "...",
      "purpose": "verify fix commit",
      "fix_commit": "...",
      "before_commit": "...",
      "hw": "BH",
      "test_command": "pytest ...",
      "before_run_id": 12345678,
      "before_run_url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345678/job/...",
      "before_dispatched_at": "2026-05-14T02:00:00Z",
      "before_result": "FAIL",
      "after_run_id": 12345679,
      "after_run_url": "https://github.com/tenstorrent/tt-metal/actions/runs/12345679/job/...",
      "after_dispatched_at": "2026-05-14T03:00:00Z",
      "after_result": null,
      "verification_conclusion": null,
      "verified_at": null
    }
  ],
  "resolved": [
    {
      "escape_id": "...",
      "hw": "BH",
      "method": "A",
      "test": "test_foo",
      "verdict": "confirmed",
      "before_run_id": 12345678,
      "before_result": "FAIL",
      "after_run_id": 12345679,
      "after_result": "PASS",
      "resolved_at": "2026-05-14T04:00:00Z",
      "note": "..."
    }
  ],
  "pending": [
    {
      "escape_id": "...",
      "hw": "BH",
      "runner_label": "...",
      "priority": 1,
      "test_name": "...",
      "test_filepath": "...",
      "suggested_test_command": "pytest ...",
      "last_failing_sha": "...",
      "first_passing_sha": "...",
      "commit_range_size": 12,
      "cluster_with": null,
      "note": "..."
    }
  ],
  "active_bisects": {}
}
```

**Recovery rule:** On any session start:
- Check every entry in `bisects_in_progress` — for each `manual_bisect.runs` entry with a `run_id`, poll its GitHub Actions status before dispatching a new probe
- Check every entry in `verification_in_progress` — for each `before_run_id`/`after_run_id`, poll status before dispatching
- See Context Reset Protocol in MEMORY.md for full procedure

---

## MANDATORY DISPATCH RULES (Evan's explicit requirements)

After **every** `workflow_dispatch` API call — no exceptions, no delays:

1. **Update Confluence page 2424012846 IMMEDIATELY** — record the run ID, URL, and set status to ⏳. Do this before any other action.
2. **DM @ebanerjeeTT** — send a Slack message mentioning @ebanerjeeTT with the run URL and a one-line description of what it is.

These two actions must happen before proceeding to any next pipeline step.

---

## Two Modes

### Mode A — Backfill
Run once manually. Scans last 60 days. Full analysis pipeline.

### Mode B — Incremental
Scheduled daily at 00:15Z via nanoclaw. Scans last 1 day only.
Same thresholds as backfill (≥10 consecutive failures, ≥10 consecutive passes).
Skip Opus pre-classification if layer pre-filter already rules it out.

---

## Pipeline Steps

### Step 1: Detection (Snowflake)

**⚠️ Snowflake commit boundaries are unreliable — always verify against CI run logs.**
Snowflake has ingestion gaps for any test type: jobs that upload no JUnit XML on success,
partial ingestion, infra aborts, etc. Do NOT treat Snowflake's `last_failing_sha` or
`first_passing_sha` as authoritative. Always correct the range first.

**Procedure: correct the commit range (do this for ALL escapes, before Step 2)**

1. From the failing job name in Snowflake, identify the CI workflow that runs the test.
   Find its workflow ID:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows
   ```
   Match on `name` or `path`.

2. List recent runs of that workflow on main:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows/{workflow_id}/runs?branch=main&per_page=100
   ```
   Each run has: `id`, `head_sha`, `created_at`, `conclusion`.

3. For each run whose `head_sha` falls between Snowflake's boundaries, find the relevant job
   and read its log:
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs   → filter for job name matching the Snowflake job name
   GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs   → search for {test_name}
   ```
   Look for:
   - `PASSED tests/.../{test_file}::{test_name}` → test passed at this commit
   - `FAILED tests/.../{test_file}::{test_name}` → test failed at this commit
   Skip runs where the log has no mention of the test (job aborted early).

4. Establish from actual log data:
   - **True `last_failing_sha`**: latest `head_sha` where the test explicitly FAILs.
   - **True `first_passing_sha`**: earliest `head_sha` after that where the test **explicitly PASSes**.

   ⚠️ **A pipeline existing at a SHA does NOT mean the job ran.**
   Jobs are sometimes omitted from a pipeline (scheduling changes, workflow restructuring, machine
   unavailability). If the job name is absent from the `first_passing_sha` pipeline's job list,
   that "pass" is a scheduling gap, not a real fix — treat the escape as unverifiable and skip it.

   Always confirm: the `first_passing_sha` pipeline contains a job matching the same name as the
   failing job, AND that job's log shows the test explicitly PASSED.

5. Recompute the range:
   ```
   GET /repos/tenstorrent/tt-metal/compare/{true_last_failing_sha}...{true_first_passing_sha}
   ```
   Use this corrected range for all subsequent steps.

Run the following SQL. Adjust `DATEADD` for backfill (60 days) vs incremental (1 day).

Thresholds:
- **Backfill**: `consecutive_fail_count >= 10`, `consecutive_pass_count >= 10`, lookback 60 days
- **Incremental**: `consecutive_fail_count >= 10`, `consecutive_pass_count >= 10`, lookback 1 day

**Signature requirement**: All failures in the streak must share the **exact same normalized error signature**
(addresses and timestamps stripped). A test that fails 5 times with 5 different errors is flaky noise,
not a regression. The SQL enforces this via `distinct_sig_count = 1`.

**Streak length requirement**: ≥ 10 consecutive failures followed by ≥ 10 consecutive passes, with
no gap between them. This rules out flaky tests — a 20% flaky test may produce a 5-run streak by
chance, but a 10-run streak is statistically unlikely without a genuine regression.

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
    -- rn_asc: 1 = oldest run; rn_by_success: for islands technique
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
    AND p.PIPELINE_START_TS >= DATEADD('day', -14, CURRENT_TIMESTAMP())  -- adjust for mode
    AND j.FAILURE_SIGNATURE NOT LIKE 'InfraErrorV1%'
),
-- Islands technique: (rn_asc - rn_by_success) is constant within a consecutive streak
islands AS (
  SELECT *, (rn_asc - rn_by_success) AS island_key
  FROM runs
),
island_summary AS (
  SELECT
    CICD_TEST_CASE_ID,
    SUCCESS,
    island_key,
    COUNT(*)                              AS streak_length,
    COUNT(DISTINCT norm_signature)        AS distinct_sig_count,  -- must be 1 for fail streaks
    MIN(PIPELINE_START_TS)                AS streak_start_ts,
    MAX(PIPELINE_START_TS)                AS streak_end_ts,
    MIN_BY(commit_sha, PIPELINE_START_TS) AS streak_first_sha,
    MAX_BY(commit_sha, PIPELINE_START_TS) AS streak_last_sha,
    MAX_BY(norm_signature, PIPELINE_START_TS) AS norm_signature,
    MIN(rn_asc)                           AS streak_first_rn,  -- for true adjacency check
    MAX(rn_asc)                           AS streak_last_rn
  FROM islands
  GROUP BY 1, 2, 3
),
fail_streaks AS (
  SELECT
    CICD_TEST_CASE_ID,
    island_key,
    streak_length AS consecutive_fail_count,
    streak_end_ts AS last_fail_ts,
    streak_last_sha AS last_failing_sha,
    streak_last_rn AS last_fail_rn,
    norm_signature
  FROM island_summary
  WHERE SUCCESS = FALSE
    AND streak_length >= 10        -- require 10+ consecutive failures
    AND distinct_sig_count = 1     -- all failures must share the same normalized error signature
),
pass_streaks AS (
  SELECT
    CICD_TEST_CASE_ID,
    island_key,
    streak_length AS consecutive_pass_count,
    streak_start_ts AS first_pass_ts,
    streak_first_sha AS first_passing_sha,
    streak_first_rn AS first_pass_rn
  FROM island_summary
  WHERE SUCCESS = TRUE
    AND streak_length >= 10  -- require 10+ consecutive passes after fix
    AND (streak_length <= 1 OR DATEDIFF('hour', streak_start_ts, streak_end_ts) / NULLIF(streak_length - 1, 0) <= 24)  -- passes must be dense (avg ≤24h between runs) — filters scheduling gaps
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
    -- True adjacency: pass streak begins on the very next run after the fail streak ends
    AND ps.first_pass_rn = fs.last_fail_rn + 1
  WHERE DATEDIFF('hour', fs.last_fail_ts, ps.first_pass_ts) <= 72  -- gap > 72h likely a scheduling pause, not a fix
),
-- Most recent transition per test, joined back to get test metadata
result AS (
  SELECT
    t.CICD_TEST_CASE_ID,
    r.test_name,
    r.test_filepath,
    CASE
      WHEN r.test_filepath LIKE '%models/%' OR r.test_filepath LIKE '%demos/%' THEN 4
      WHEN r.test_filepath LIKE '%ttnn/%' OR r.test_filepath LIKE '%tt_eager%' THEN 3
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
    t.norm_signature,
    CASE
      WHEN t.norm_signature LIKE '%tt_metal/hw/%' THEN 1
      WHEN t.norm_signature LIKE '%tt_metal/%' THEN 2
      WHEN t.norm_signature LIKE '%ttnn/%' THEN 3
      ELSE NULL
    END AS error_layer_hint  -- hints at fix layer from error message; if error_layer_hint < test_layer, strong vertical escape signal
  FROM transitions t
  JOIN (SELECT DISTINCT CICD_TEST_CASE_ID, test_name, test_filepath FROM runs) r
    ON t.CICD_TEST_CASE_ID = r.CICD_TEST_CASE_ID
  QUALIFY ROW_NUMBER() OVER (PARTITION BY t.CICD_TEST_CASE_ID ORDER BY t.last_fail_ts DESC) = 1
)
SELECT *
FROM result
WHERE test_layer >= 3  -- only L3 (ttnn) and L4 (models/demos) — L1/L2 tests that fail are unlikely to be vertical escapes
ORDER BY last_fail_ts DESC
LIMIT 50;
```

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
If any commit does → collect the cross-layer commits for Step 3.

**For each cross-layer commit, fetch its full diff now — do not skip this:**
```
GET /repos/tenstorrent/tt-metal/commits/{sha}
```
The `files[].patch` field contains the actual line-level diff. You MUST pass this to Opus in
Step 3c — not just the filename. Opus cannot reason about causality from filenames alone.

**Commit intent signal**: Note whether any cross-layer commit has explicit fix intent — commit
message containing "fix", "bug", "revert", "restore", "correct". If the range has ZERO fix-intent
cross-layer commits (only features, refactors, docs), set a `low_confidence_prior = true` flag.
Proceed to Step 3 but Opus should treat this as a strong indicator against PROCEED_TO_BISECT.

---

### Step 3: Opus Pre-classification

#### 3a: Noise blocklist (before calling Opus)

Before spending tokens on Opus, check the failure log for known infra noise patterns.
If ANY of the following match, mark `status: "skipped_prefilter"` in `seen-escapes.json` and stop:

- `TracyAlloc|TracyFree|FrameMarkNamed|FrameMarkStart` — Tracy profiler symbols leaking into logs
- `tenstorrent_pcie_ioctl|ioctl.*pcie.*fail` — PCIe driver errors, not test code
- `FileNotFoundError.*No such file or directory.*libtt_` — stale shared library path, deploy issue
- Error log contains zero pytest/gtest test identifiers (no `FAILED`, `PASSED`, `::test_`) and no
  `TT_FATAL`/`TT_THROW` — pure infra noise with no test output at all

#### 3b: Fetch failure log

```
GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs
```
(Get `job_id` from the pipeline data in Snowflake or via the runs API for the relevant commit.)
Use first 200 lines of the relevant test step.

#### 3c: Opus prompt

Provide Opus with:
1. The raw failure log excerpt
2. The cross-layer commits from Step 2 (SHA, message, files changed, **full line-level diffs** — NOT just filenames)
3. The test filepath, name, and streak stats from Step 1
4. The `low_confidence_prior` flag from Step 2 (if set)

> **⚠️ DO NOT skip the diffs.** Opus cannot reason about causality from filenames alone.
> Pass the `files[].patch` content from `GET /repos/tenstorrent/tt-metal/commits/{sha}` for every
> cross-layer commit. If a diff is too large (>500 lines), pass the first 500 lines and note it was truncated.

```
You are analyzing a potential vertical bug escape in tenstorrent/tt-metal CI.

════════════════════════════════════════════════════════════════
YOUR PRIMARY TASK: DETERMINE WHETHER THIS IS A VERTICAL ESCAPE
════════════════════════════════════════════════════════════════

A vertical escape means a bug at layer N caused a test at layer M (N < M) to fail.
This is NOT just "a lower-layer file was changed in this range."
You must trace a SPECIFIC CAUSAL MECHANISM:
  → Which exact function/macro/register/ABI changed in the lower-layer commit?
  → What is the call path from that change to the test's execution path?
  → Why does that change produce EXACTLY the observed error (wrong output, TT_FATAL, assertion)?

If you cannot name all three with specificity, the verdict must be SKIP_UNRELATED.
Vague causal links ("it's possible the change affected performance") do NOT qualify.

════════════════════════════════════════════════════════════════

Test: {test_name}
File: {test_filepath} (layer {test_layer})
Streak: {consecutive_fail_count} consecutive failures → {consecutive_pass_count} consecutive passes after fix
Failure window: {last_fail_ts} to {first_pass_ts}
Low-confidence prior: {low_confidence_prior}  ← if true, weight heavily against PROCEED_TO_BISECT

Cross-layer commits between last failure and first pass (with full diffs):
{commit_list_with_diffs}

Raw failure log:
{failure_log}

Work through these 5 checks in order:

CHECK 1 — Error classification:
  Is the error from TEST CODE (assertion, TT_FATAL/TT_THROW, wrong output, OOM in test, test timeout)?
  Or INFRASTRUCTURE (runner disconnect, docker pull fail, network timeout, SSH drop, "Lost communication")?
  → If infrastructure: verdict SKIP_LIKELY_NOISE. Stop.

CHECK 2 — Determinism:
  Does the SAME test fail with the SAME error signature consistently?
  A flaky test alternates pass/fail; a real regression fails steadily.
  Note: {consecutive_fail_count} consecutive failures is a strong determinism signal.
  → If no consistent test name or error signature across runs: verdict SKIP_LIKELY_NOISE. Stop.

CHECK 3 — Causal analysis (HIGH BAR — read carefully):
  You must establish ALL THREE of the following before this check passes:
  (a) **Specific change**: Name the exact function, macro, register, or ABI modified in the
      cross-layer commit. "Changed files in tt-llk" does NOT pass. Example of passing:
      "ckernel_sfpu_binary_bcast.h stopped restoring lreg11 to -1 after use."
  (b) **Call path**: Trace how the test reaches the changed code.
      Example: "test calls ttnn.adamw → SFPU binary bcast kernel → lreg11 read in step 2."
  (c) **Error match**: Explain why the specific change produces EXACTLY the observed failure,
      not a different error. Example: "lreg11 corrupted → cached kernel computes wrong
      gradient → AdamW weight update diverges → assertion fails on step 2 of 4."
  Red flags that should push you toward SKIP_UNRELATED:
    - Hardware-specific sensor errors (temp, PCI link), timing races, network/registry failures
    - The commit message says "feature", "refactor", "cleanup" with no "fix"/"bug"/"revert"
    - `low_confidence_prior = true` (no fix-intent commits in range)
    - You can only say "it's possible" or "might affect" — not "it does because..."
  → If you cannot satisfy all three: verdict SKIP_UNRELATED. Stop.

CHECK 4 — Range size sanity:
  How many commits are in the bisect range? If > 50, the range is likely too wide to bisect
  efficiently unless your causal candidate from Check 3 is a single specific commit.
  → If range > 50 AND no single obvious fix commit: lower confidence to "low".

CHECK 5 — Cross-test scope:
  Is exactly ONE specific test failing, or many unrelated tests?
  Many unrelated tests failing → likely infra outage, not a code regression.
  → If many unrelated tests: lower confidence, note in reasoning.

Respond with JSON only:
{
  "verdict": "PROCEED_TO_BISECT" | "SKIP_LIKELY_NOISE" | "SKIP_UNRELATED",
  "most_likely_fix_sha": "<commit SHA, or null>",
  "confidence": "high" | "medium" | "low",
  "check1_error_type": "test_code" | "infrastructure",
  "check2_determinism": "deterministic" | "likely_flaky" | "unknown",
  "check3_causal_mechanism": "<one sentence explaining the link, or null>",
  "vertical_escape_justification": {
    "specific_change": "<exact function/macro/register/ABI changed, or null>",
    "call_path": "<how the test reaches the changed code, or null>",
    "error_match": "<why this specific change produces exactly the observed error, or null>"
  },
  "reasoning": "<2-3 sentences: what the error is, which commit is likely the fix and why>"
}
```

**Post-Opus validation (BrAIn must check before dispatching):**
Inspect `vertical_escape_justification` in Opus's response.
- If `specific_change`, `call_path`, or `error_match` is null, generic, or uses hedging language
  ("might", "possibly", "could affect") → override verdict to `SKIP_UNRELATED`, regardless of
  what Opus returned. Mark `status: "skipped_vague_justification"` in `seen-escapes.json`.
- Only proceed if all three fields are concrete and specific.

If verdict is not PROCEED_TO_BISECT → write escape ID to `seen-escapes.json` with
`status: "skipped_prefilter"` and stop.

**MANDATORY layer check after Opus:** Before dispatching any hardware run, confirm that
`most_likely_fix_sha`'s layer is STRICTLY LOWER than the test layer (fix\_layer < test\_layer).
- Same layer (L3 test, L3 fix) → horizontal bug, NOT a vertical escape. Discard.
- Higher layer (L3 test, L4 fix) → inverse, NOT a vertical escape. Discard.
- Only fix\_layer < test\_layer qualifies. If Opus's suspected fix fails this check,
  mark `status: "skipped_not_vertical"` in `seen-escapes.json` and stop.

---

### Step 4: Manual Binary Search (find the fix commit)

**⚠️ DO NOT use `bisect-dispatch.yaml`** — that finds *breaking* commits, not *fixing* commits.
**⚠️ DO NOT use `test-dispatch.yaml`** — it runs the full job suite; use a probe branch instead.

#### The method (universal — applies to all test types)

**Step 4a — Read existing CI run logs before dispatching anything**

Every commit on main has at least one CI run. Reading logs costs nothing; dispatching
triggers builds that can take 60+ min. Before probing any midpoint:

1. From the workflow identified in the Step 1 range-correction procedure, list runs:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows/{workflow_id}/runs?branch=main&per_page=100
   ```
2. For each midpoint commit in the binary search, check if a run exists at that SHA
   (`run.head_sha == commits[mid].sha`).
3. If a run exists: read its job log for the specific test (same procedure as Step 1 range
   correction — find the right job, download log, grep for test_name PASSED/FAILED).
   This IS your probe result — do not dispatch a new run.
4. Only dispatch when no existing run covers the midpoint commit.

**Step 4b — Probe branch dispatch (only when no existing run covers a midpoint)**

**⚠️ DISPATCH SAFETY GUARD — check before EVERY workflow_dispatch call:**
1. Confirm the target `ref` starts with `brain/` — NEVER dispatch on `main`, `ebanerjee/*`, or any non-probe branch
2. Confirm the branch was just created or verified to contain the pruned YAML — not a stale branch from a previous run
3. If the branch already exists: check for an existing in-flight or completed run on it before dispatching:
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs?branch={branch_name}&per_page=5
   ```
   If a run is already queued or in_progress: DO NOT dispatch again — wait for it.
   If a completed run exists with the same workflow: read its result instead of re-dispatching.

Never modify `test-dispatch.yaml` or any shared workflow. Instead:

**⚠️ WORKFLOW PRUNING IS MANDATORY — NOT OPTIONAL**

Production CI workflows can have 47–51 jobs. Dispatching the full workflow unmodified ties up
shared hardware for hours and is a protocol violation. Every probe dispatch MUST use a
pruned workflow YAML — regardless of which workflow runs the failing test.

**Pruning rules (must result in ≤ 2 jobs):**
1. Open the workflow YAML for the failing test (identified in Step 1 — e.g. `blackhole-demo-tests.yaml`,
   `blackhole-e2e-tests.yaml`, `nightly-wh-models.yaml`, etc. — whatever workflow the CI run came from).
2. Find the single target test job (the job whose log shows the test failing).
3. Look at that job's `needs:` field — typically `[build-artifact]` or similar.
4. **Delete every other job from the YAML.** Keep ONLY:
   - The target test job (with its `run:` step pruned to just the specific failing test command)
   - The jobs listed in its `needs:` field (typically just `build-artifact`)
5. Result: 2 jobs max (build + test). If using artifact reuse (see below), 1 job (test only).

**Pruning the test command within the job:**
Inside the target job's `run:` step, replace the full test matrix invocation with just:
```bash
pytest {test_filepath}::{test_name} -v
```
Remove any `--timeout`, matrix loops, or parametrize expansions not needed for this specific test.

**⚠️ ARTIFACT REUSE IS MANDATORY when artifacts exist — you MUST NOT trigger a fresh build otherwise**

Building from scratch takes 60+ min and wastes shared hardware. Always check for existing
artifacts before creating a `build-artifact` job in the pruned workflow:

```
GET /repos/tenstorrent/tt-metal/actions/runs?head_sha={target_sha}
```
Look for a completed merge-gate or post-commit run at or near that SHA. Check its artifacts:
```
GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/artifacts
```
If build artifacts exist — pass all three artifact inputs in the workflow dispatch payload:
```json
{
  "ref": "{branch}",
  "inputs": {
    "dev-docker-image": "{dev_docker_image from prior run's outputs}",
    "build-artifact-name": "{build_artifact_name from prior run's outputs}",
    "wheel-artifact-name": "{wheel_artifact_name from prior run's outputs}"
  }
}
```
The `build-artifact` job has `if: ${{ inputs.dev-docker-image == '' }}` — it skips itself
when `dev-docker-image` is provided. The `resolve-artifacts` job then forwards the
pre-built artifact names to all downstream test jobs. **All three inputs must be set
together** — `resolve-artifacts` will fail with an error if only some are set.

To get the artifact names from a prior run:
```
GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs   → find `resolve-artifacts` job
GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs   → grep for `build-artifact-name=`, `dev-docker-image=`, `wheel-artifact-name=`
```
Or from the build-artifact job outputs via the API:
```
GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}   → check `jobs_url`
```

Do NOT remove the `build-artifact` job from the YAML — it self-skips via its `if:` condition.
Keep the YAML structure intact; just pass the three inputs at dispatch time.

Artifact reuse tests the *artifact commit's* code, not the probe branch's code. Only use when the
artifact commit is within ~5 commits of the target and those commits are irrelevant to the test.

**⚠️ Tracy/profiler mismatch when reusing merge-gate artifacts**

Merge-gate runs produce Release builds without the Tracy profiler. Their `build-artifact-name`
will NOT contain `_profiler_` in the artifact name. If you pass a profiler artifact name to a
workflow that expects a non-profiler artifact (or vice versa), the download step will fail:
`ERROR: Could not find build artifact matching expected pattern`.

**Check**: grep the `resolve-artifacts` log of the source run for `build-artifact-name=`. If
the value does NOT contain `profiler`, you're using a standard Release build — this is fine
for most test probes. Pass that exact artifact name as `build-artifact-name` input.

1. Create (or reuse) a branch `brain/{escape-id}-bisect-probe` in a worktree.

2. Attempt artifact reuse (MANDATORY first step):
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs?head_sha={target_sha}
   ```
   Prefer the nearest completed merge-gate run (`gh-readonly-queue/main/...` branch).
   If found: extract `dev-docker-image`, `build-artifact-name`, `wheel-artifact-name` from
   that run's `resolve-artifacts` job logs, then pass all three as dispatch inputs (see above).
   The `build-artifact` job will self-skip via its `if:` condition — do not edit the YAML for this.

3. If no reusable artifacts exist: keep `build-artifact` in the pruned YAML, then rebase the
   probe branch to the target commit with the pruned workflow on top:
   ```bash
   cd /workspace/group/worktrees/{probe-branch}
   WORKFLOW_COMMIT=$(git rev-parse HEAD)
   git reset --hard {target_sha}
   git cherry-pick $WORKFLOW_COMMIT
   git push origin brain/{escape-id}-bisect-probe --force
   ```

4. After dispatch, the overall run may report failure for unrelated reasons.
   Always read the specific job's log to get the actual test result:
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs   → find the job matching the failing test's job name
   GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs   → grep for {test_name} PASSED or FAILED
   ```

**Algorithm:**
1. Get the commit list between the corrected `last_failing_sha` (exclusive) and
   `first_passing_sha` (inclusive):
   ```
   GET /repos/tenstorrent/tt-metal/compare/{last_failing_sha}...{first_passing_sha}
   ```
   → `commits` array, index 0 = oldest (first after last_failing), last = first_passing
2. Pick midpoint index `mid = len(commits) // 2`
3. Probe `commits[mid].sha` using Step 4a (read existing log) or Step 4b (dispatch probe branch).
4. After MANDATORY post-dispatch actions (Confluence + DM Evan), wait for result.
5. If **PASS** → fix is in the lower half: set `high = mid - 1`
6. If **FAIL** → fix is in the upper half: set `low = mid + 1`
7. Repeat from step 2 with `commits[low..high]`
8. When `low == high` (or range is empty), the fix commit is `commits[low]`
   - Special case: if ALL midpoints PASS → fix commit is `commits[0]` (first after last_failing)

9. **Before accepting the bisect result** — inspect the fix commit:
   - If commit message contains `[skip ci]` OR only touches `CODEOWNERS`, `.github/`, docs, or
     CI config files → it is a **false positive**. These are no-op commits that don't affect
     test behavior. Look at the commit immediately before it in the range — that is the real candidate.
   - If the fix commit is a merge commit with no meaningful diff (auto-merge, version bump) → same rule.
   - Mark false positives as `refuted_false_positive` in campaign-state.json and do not dispatch verification.

Save each midpoint run ID to `campaign-state.json`.

**Arch determination:**
- BH-LoudBox / topology-6u → `blackhole`
- N150, N300, P150, P300, config-t3000 → `wormhole_b0`

If timed out → write `status: "inconclusive_bisect_timeout"` to `seen-escapes.json`, stop.

---

### Step 4.5: Build Variant Check (MANDATORY before any probe dispatch)

Before creating probe branches, check the original failing job's build variant:

```bash
# From Snowflake — get the job name for the failing test
SELECT J.NAME FROM TTDATASF.SW_TEST.CICD_JOB J
JOIN TTDATASF.SW_TEST.CICD_TEST T ON T.CICD_JOB_ID = J.CICD_JOB_ID
WHERE T.TEST_CASE_ID = {test_case_id}
AND J.JOB_SUCCESS = FALSE
LIMIT 5;
```

- If `J.NAME` contains **"LLK asserts"** → probe workflow must use the LLK-asserts build config
- If `J.NAME` contains **"watcher"** → probe workflow must use the watcher build config
- If `J.NAME` is the standard build → probe can use default config

**A probe in the wrong build variant will give an invalid result and must be discarded.**
Record the required variant in `campaign-state.json` before dispatching.

### Step 4.6: Cluster Representative Rule

When multiple escapes share the **exact same fix commit SHA**:
- One representative probe pair covers the entire cluster
- Do NOT dispatch individual probes for each escape in the cluster
- Record the shared run IDs in `confirmed-escapes.json` for all escapes in the cluster, noting it as a cluster representative
- If any escape in the cluster has a *different* fix commit SHA, it gets its own probe pair

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

**⚠️ Validate the entry before writing to confirmed-escapes.json or updating Confluence:**

```python
REQUIRED_FIELDS = [
    "escape_id", "test_name", "test_filepath", "test_layer",
    "fix_commit_sha", "fix_commit_message", "fix_layer",
    "before_run_id", "after_run_id", "reasoning", "confirmed_at"
]

def validate_confirmed_escape(entry):
    missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # fix_layer must be strictly less than test_layer
    if not (entry["fix_layer"] < entry["test_layer"]):
        raise ValueError(
            f"Layer invariant violated: fix_layer={entry['fix_layer']} "
            f"must be < test_layer={entry['test_layer']}"
        )

    # run IDs must be integers > 0
    for field in ("before_run_id", "after_run_id"):
        if not isinstance(entry[field], int) or entry[field] <= 0:
            raise ValueError(f"{field} must be a positive integer, got: {entry[field]!r}")

    # confirmed_at must be ISO 8601
    import re
    if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", entry["confirmed_at"]):
        raise ValueError(f"confirmed_at is not ISO 8601: {entry['confirmed_at']!r}")
```

If validation fails → do NOT write to `confirmed-escapes.json`, do NOT update Confluence.
Instead mark `status: "inconclusive_api_error"` in `seen-escapes.json` and DM Evan.

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

**Update Confluence confirmed escapes page** (ID 2428895268):
https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/2428895268/confirmed+bug+escapes

Add a new row to the table on that page using the same column format as the existing entries.
Fetch the current page content first to avoid clobbering concurrent writes.

Delete BEFORE/AFTER branches from GitHub.

---

### Step 7: Confluence Updates (happens at MULTIPLE points, not just at the end)

Page ID: 2424012846
URL: https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/2424012846/bug+escapes

The page has four sections. Move each escape between sections as its status changes.
Never delete an entry — only move it and update its status field.

**After Step 3 (Opus pre-classification passes) → add a row to the "Candidate Table" on page 2424012846:**

Table header (do not change):
```
| Escape ID | Test File | Job | HW | Last Fail Commit | First Pass Commit | Layers | Runs Dispatched | Opus Verdict | Opus Reasoning | Status |
```

**Layers column format:**
- Under Investigation: `Test L{X} → Suspected Fix L{Y}`
  - X = test layer number (from test filepath); Y = lowest layer touched by the suspected fix commit.
  - Example: "Test L4 → Suspected Fix L3" = failing test in models/ (L4), suspected fix in ttnn/ (L3).
- Leave blank (empty cell) for: abandoned, refuted, or skipped candidates.

Example row:
```
| 892871__a249d854 | test_wan_decoder.py (layer 4) | T3K WAN tests | T3K (WH) | [a249d854](https://github.com/tenstorrent/tt-metal/commit/a249d854) | [b3e8eb62](https://github.com/tenstorrent/tt-metal/commit/b3e8eb62) | Test L4 → Suspected Fix L3 | [Run 26130351088 ❌ FAIL](https://github.com/tenstorrent/tt-metal/actions/runs/26130351088/job/76855704319) | PROCEED_TO_BISECT (MEDIUM) | Commit c53fc338 fixes nlp_create_qkv_heads precision. | 🔍 Under Investigation — mid1 dispatched {date} |
```

**⚠️ URL format rules (non-negotiable):**
- Commit links: always `https://github.com/tenstorrent/tt-metal/commit/{full_sha}` — never bare SHAs
- Run links: always job-level `https://github.com/tenstorrent/tt-metal/actions/runs/{run_id}/job/{job_id}` — never bare run IDs, never run-level-only URLs
- Link text must include pass/fail emoji: `Run {run_id} ❌ FAIL` or `Run {run_id} ✅ PASS`
- To get `job_id`: `GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs` → find the job matching the test name → use its `id` field

**After Step 4 (bisect completes) → update the entry, still in "Under Investigation":**
Add fix commit and bisect proof link. Update status to "Under Investigation — verification dispatched {date}".

**After Step 5 (verification complete):**
- CONFIRMED → move entry to "✅ Confirmed Escapes", update status, add before/after run links
- REFUTED → move entry to "❌ Refuted", update status, add explanation of why it was refuted
- INCONCLUSIVE_TIMEOUT → move entry to "⏳ Inconclusive", update status, note retry scheduled

**Confirmed escapes table** (page 2428895268) uses this header:
```
| Escape ID | Test | HW | Fix PR | Fix Layer | Test Layer | Confirmed | Parent probe run (fix_commit^ → expect FAIL) | Fix commit probe run (fix_commit → expect PASS) |
```

The two probe columns have colored backgrounds (orange = parent/before, green = fix/after).
Each probe cell contains a job-level link: `[Run {run_id} ❌ FAIL](https://github.com/tenstorrent/tt-metal/actions/runs/{run_id}/job/{job_id})`

**Entry format once confirmed** (add row to 2428895268; update row in 2424012846):
- Fix PR cell: `[PR #{pr_number}](https://github.com/tenstorrent/tt-metal/pull/{pr_number}) {fix_commit_message}`
- Parent probe: `[Run {before_run_id} ❌ FAIL](https://github.com/tenstorrent/tt-metal/actions/runs/{before_run_id}/job/{before_job_id})`
- Fix probe: `[Run {after_run_id} ✅ PASS](https://github.com/tenstorrent/tt-metal/actions/runs/{after_run_id}/job/{after_job_id})`

Extended bullet-point format (for notes outside the table):
```
### Escape #{N}: {fix_commit_message[:60]}

- *Test*: {test_name}
- *Test layer*: {test_layer}
- *Fix layer*: {fix_layer}
- *Layers*: Test L{test_layer} → Verified Fix L{fix_layer}
- *Last failure*: [Run {last_failure_run_id} ❌ FAIL](https://github.com/tenstorrent/tt-metal/actions/runs/{last_failure_run_id}/job/{last_failure_job_id})
- *First success*: [Run {first_success_run_id} ✅ PASS](https://github.com/tenstorrent/tt-metal/actions/runs/{first_success_run_id}/job/{first_success_job_id})
- *Fix commit*: [{fix_commit_sha[:8]}](https://github.com/tenstorrent/tt-metal/commit/{fix_commit_sha})
- *Bisect proof*: [Run {bisect_run_id}](https://github.com/tenstorrent/tt-metal/actions/runs/{bisect_run_id}/job/{bisect_job_id})
- *Parent probe*: [Run {before_run_id} ❌ FAIL](https://github.com/tenstorrent/tt-metal/actions/runs/{before_run_id}/job/{before_job_id})
- *Fix probe*: [Run {after_run_id} ✅ PASS](https://github.com/tenstorrent/tt-metal/actions/runs/{after_run_id}/job/{after_job_id})
- *Reasoning*: {reasoning}
- *Status*: Confirmed {date}
```

To get job IDs: `GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs` — match on job name containing the test name fragment.

**Confluence update procedure (mandatory — do not skip version handling):**

```python
# 1. Fetch current page to get version number
page = confluence_get_page(page_id="{page_id}")
current_version = page["version"]["number"]

# 2. Apply your edit to the content

# 3. Update with version number — Confluence uses optimistic locking
confluence_update_page(
    page_id="{page_id}",
    title=page["title"],
    content=updated_content,
    # version number MUST be current_version + 1 (the API increments from current)
)

# 4. If you get HTTP 409 Conflict: re-fetch, re-apply edit, retry (max 3 attempts)
# 409 means someone else edited between your fetch and your update — re-fetch and try again
```

The `mcp__jira__confluence_update_page` tool handles versioning internally — it fetches the
current version before updating. Do NOT call `confluence_get_page` + `confluence_update_page`
as separate steps with a gap between them if you can avoid it; do the edit atomically.

---

## GitHub API Resilience (MANDATORY — applies to ALL GitHub API calls)

Every GitHub API call in this runbook can fail. Handle all failures:

```
Retry policy:
  - 429 (rate limited): read X-RateLimit-Reset header, sleep until reset, then retry
  - 500/502/503 (GitHub outage): exponential backoff — wait 30s, 60s, 120s — then mark step inconclusive
  - 404 (run/branch/artifact deleted): do NOT retry — treat as "no data at this SHA", move on
  - 403 (forbidden/token issue): stop campaign, DM Evan — this needs human intervention
  - Network timeout: retry up to 3 times with 10s delay between attempts

Rate limit budget:
  - Authenticated: 5000 requests/hour
  - Each escape bisect uses ~15-20 API calls; each verification uses ~10
  - Max safe throughput: ~30 escapes/hour — well within limits for nightly incremental mode
  - If X-RateLimit-Remaining < 200: stop dispatching new probes until reset

Hard stop:
  If 3 consecutive API calls to the same endpoint all fail → stop the current escape,
  mark it inconclusive_api_error in campaign-state.json, move to next candidate.
  Do NOT retry indefinitely.
```

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

## Continuous Campaign Rule (MANDATORY)

**When a campaign ends with no bisects_in_progress and no pending candidates, immediately start a new one.** Do not wait for the next scheduled cron. Query Snowflake for fresh candidates right away. Spending cron cycles posting "all done" without starting new campaigns is wasted time.

**If a status update contains only "campaign is done" with no new runs dispatched or candidates found, that update is a failure.** Always end each work session with either (a) active runs in flight or (b) a fresh Snowflake scan producing new candidates, or (c) an explicit explanation of why neither is possible.

---

## Hardware-Specific Dispatch Rules

### Single-card jobs (N150, N300, P150, P300)
- Verification (Step 5): BEFORE and AFTER can be dispatched **in parallel**, as long as no more than one set of verification per machine type is running at a time.
- Bisect (Step 4): always one probe at a time (inherent to binary search).

### Multi-card jobs (T3K, Galaxy)
- **Only 1 run at a time per machine type** — never dispatch two runs simultaneously on T3K or Galaxy.
- Verification (Step 5): dispatch BEFORE first, **wait for it to complete**, then dispatch AFTER. Strictly sequential.
- Bisect (Step 4): unchanged — already sequential by nature (one probe per midpoint).
- During **weekends**: T3K and Galaxy runs **may run during the day** (not just nightly windows). Still observe the 1-at-a-time-per-machine rule.
- During **weekends**: T3K and Galaxy runs **may run during the day** (not just nightly windows). Still observe the 1-at-a-time-per-machine rule.

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
