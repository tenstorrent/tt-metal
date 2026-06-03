# Bug Escape Detection Runbook

**Version:** 2.0 — Updated 2026-05-27 to include horizontal escapes, context reset protocol, 90-day lookback, and incremental mode.

**Local authoritative copy.** Do not rely on the GitHub copy mid-session. Update this file when the GitHub source (`tenstorrent/tt-metal` branch `ebanerjee/markdown-files`, path `instructions_for_llms/bug_escape_detection.md`) changes.

---

## Definitions

**Vertical bug escape**: A test at layer X fails, and the commit that fixed it touches a layer LOWER than X. The bug "escaped" downward past CI gates. fix_layer < test_layer.

**Horizontal bug escape**: A test at layer X fails, and the commit that fixed it touches the SAME layer X. The bug slipped through its own layer's CI. fix_layer == test_layer.

**Layer hierarchy** (lower number = more foundational):
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

A **test's layer** is determined by its file path in pytest output.
A **fix's layer** is the lowest-numbered layer among files changed in the fix commit.

- fix_layer < test_layer → **vertical** escape
- fix_layer == test_layer → **horizontal** escape
- fix_layer > test_layer → **inverse** (not an escape, discard)

---

## State Files (all in /workspace/group/bug-escapes-data/)

| File | Purpose |
|------|---------|
| `seen-escapes.json` | Set of escape IDs already analyzed. Prevents re-analysis. |
| `confirmed-escapes.json` | All confirmed escapes (source of truth for Confluence). |
| `campaign-state.json` | Active campaign: candidates, active run IDs, current step. |
| `RUNBOOK.md` | This file. |

**Escape ID format**: `{test_case_id}__{last_failing_sha[:8]}`
Example: `42847291__1ee8c3ca`

Before touching any candidate, check `seen-escapes.json`. If ID is present, skip entirely.
After any verdict (confirmed, refuted, unverifiable), write the ID to `seen-escapes.json`.

### campaign-state.json schema

```json
{
  "schema_version": 2,
  "mode": "backfill|incremental",
  "last_updated": "2026-05-27T14:30:00Z",
  "current_step": "sql|layer_filter|opus|pr_proof|bisect|verification|recording",
  "backfill_window_days": 90,
  "candidates": {
    "ESCAPE_ID": {
      "test_case_id": "...",
      "last_failing_sha": "...",
      "first_passing_sha": "...",
      "test_layer": 3,
      "candidate_type": "vertical|horizontal|both",
      "status": "sql_found|layer_filtered|opus_classified|bisecting|awaiting_probes|confirmed|rejected",
      "opus_verdict": "vertical_escape|horizontal_escape|inconclusive|not_escape|null",
      "active_runs": {
        "before_run_id": "12345|null",
        "after_run_id": "67890|null",
        "bisect_run_ids": []
      },
      "confirmation_method": "bisect|pr_ci_proof|assumed_horizontal|null",
      "confidence": "proven|assumed|null",
      "fix_commit": "sha|null",
      "candidate_fix_commits": [],
      "confluence_updated": false,
      "dm_sent": false
    }
  },
  "completed_escape_ids": [],
  "rejected_escape_ids": []
}
```

**Invariant:** NEVER dispatch a probe without reading campaign-state.json first. This prevents duplicate dispatches after a context reset.

### Session Locking Protocol

`campaign-state.json` contains a `lock` field to prevent cron agents from colliding with an active session.

**When starting work (main session or cron):**
1. Read campaign-state.json
2. If `lock.locked == true` AND `lock.expires_at` is in the future → **stand down immediately. Do not do any work. DM @ebanerjeeTT: "not working because another session is active."**
3. If unlocked (or lock is expired): set `lock.locked = true`, `lock.locked_at = now`, `lock.expires_at = now + 90 minutes`, `lock.held_by = "brief description"`. Write to file before proceeding.

**While working:** Refresh `lock.expires_at = now + 90 minutes` every 15 minutes.

**When done:** Set `lock.locked = false`, `lock.locked_at = null`, `lock.expires_at = null`, `lock.held_by = null`. Write to file.

**Expired lock:** If `lock.locked == true` but `lock.expires_at` is in the past, the previous session died without releasing the lock. Treat as unlocked — acquire it and proceed.

---

## MANDATORY DISPATCH RULES (Evan's explicit requirements)

After **every** `workflow_dispatch` API call — no exceptions, no delays:

0. **Write to `campaign-state.json` FIRST** — record the run ID, URL, candidate escape_id, and current step. This is the highest-priority action. Even if Confluence or Slack fail, the run ID is safe. Do this BEFORE anything else.
1. **Update Confluence page 2424012846 IMMEDIATELY** — record the run ID, URL, and set status to ⏳.
2. **DM @ebanerjeeTT** — send a Slack message mentioning @ebanerjeeTT with the run URL and a one-line description of what it is.

These three actions must happen before proceeding to any next pipeline step.

### General State Persistence Rule

**After EVERY significant action, write state before proceeding.** Significant actions include:
- Dispatching any probe (write run ID to campaign-state.json immediately)
- Receiving a probe result (write verdict to campaign-state.json before updating Confluence)
- Classifying an escape (write candidate status update to campaign-state.json)
- Confirming or refuting an escape (write to confirmed-escapes.json and seen-escapes.json before updating Confluence)

Context compression can happen at any moment. If state is not written to disk, it is lost. The rule is: **write first, then act on the result.**

---

## Messaging Rules (what to DM @ebanerjeeTT)

### Always DM

1. **Every `workflow_dispatch`** — mandatory (see above). Include run URL + one-line description of what it is testing.
2. **Every confirmed escape** — send one message when a verdict of `confirmed` is reached. Include: escape ID, test name, layer type (vertical/horizontal), fix PR, confirmation method (proven vs assumed).
3. **Blockers** — any of the following warrant an immediate DM:
   - Snowflake unreachable or returning no results unexpectedly
   - A probe timing out 2+ times on the same candidate
   - Structural issue with a candidate that can't be resolved without human input

### Daily summary at 08:00Z

Send a single digest message containing:
- N new escapes confirmed since last summary (vertical and horizontal counts)
- Running campaign totals (vertical total / horizontal total)
- Probe queue depth (candidates pending, candidates in-flight)
- Any candidates stuck or blocked

### Never DM

- Refuted or skipped candidates
- Routine poll cycles ("checked run 12345, still running")
- Between-session state saves or Confluence updates
- Snowflake scan results with zero new candidates

---

## Two Modes

**Current mode: BACKFILL** (as of 2026-05-29). Evan will signal when to switch to incremental.

### Mode A — Backfill (ACTIVE)
Cron runs **hourly**. Each session scans last **90 days** and continues the pipeline from `current_step` in campaign-state.json.
- Consecutive failure threshold: **≥ 5**
- Consecutive pass threshold: **≥ 5**
- Gap threshold: **≤ 168 hours** (1 week)
- Error signature count: **≤ 3**
- Adjacency tolerance: **up to 4 skipped runs between fail and pass streak**

### Mode B — Incremental (NOT YET ACTIVE)
Cron runs **hourly** (same schedule as backfill). Each session scans last **2 days** only.
- Consecutive failure threshold: **≥ 3**
- Consecutive pass threshold: **≥ 3**
- Gap threshold: **≤ 48 hours**
- Flake guard: if test has >30% failure rate over last 14 days, flag `likely_flaky`, deprioritize (do NOT auto-reject)

---

## Pipeline Steps

### Step 1: Detection (Snowflake)

**⚠️ Snowflake commit boundaries are unreliable — always verify against CI run logs.**
Snowflake has ingestion gaps. Do NOT treat `last_failing_sha` or `first_passing_sha` as authoritative until verified against actual CI logs.

**Procedure: correct the commit range (do this for ALL escapes before Step 2)**

1. From the failing job name in Snowflake, find the workflow that runs the test:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows
   ```
2. List recent runs on main:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows/{id}/runs?branch=main&per_page=100
   ```
3. For each run whose `head_sha` falls in the Snowflake range, find the relevant job and check its log for explicit `PASSED` or `FAILED` for the test name.
4. Establish true `last_failing_sha` (latest explicit FAIL) and `first_passing_sha` (earliest explicit PASS after that).
5. Recompute: `GET /repos/tenstorrent/tt-metal/compare/{true_last_failing}...{true_first_passing}`

**Unified SQL (both vertical and horizontal candidates):**

Use `DATEADD('day', -90, ...)` for backfill, `DATEADD('day', -2, ...)` for incremental. Adjust thresholds per mode (see Two Modes section).

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
    AND streak_length >= 5         -- 5 for backfill; 3 for incremental
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
    AND streak_length >= 5         -- 5 for backfill; 3 for incremental
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
  WHERE DATEDIFF('hour', fs.last_fail_ts, ps.first_pass_ts) <= 168  -- 168h backfill; 48h incremental
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
WHERE test_layer >= 1    -- ALL layers (vertical: test_layer >= 3; horizontal includes L1-L2)
ORDER BY last_fail_ts DESC
LIMIT 100;
```

For each row:
- Compute escape ID: `{CICD_TEST_CASE_ID}__{last_failing_sha[:8]}`
- Check `seen-escapes.json`. If present, skip.
- Otherwise, add to candidate list.

**NOTE on Snowflake gotchas:**
- Column is `H.HOST_NAME` (not `H.HOSTNAME`)
- Column is `J.JOB_SUCCESS` (not `J.SUCCESS`)
- CICD_TEST has 5.8B rows — ALWAYS filter by date and use LIMIT
- `CICD_PIPELINE.PROJECT = 'tt-metal'` (not `tenstorrent/tt-metal`)
- `CICD_PIPELINE.GIT_COMMIT_HASH` (not `GIT_SHA`)
- LLK assert nightly runs (`J.NAME LIKE '%LLK asserts%'`) — exclude from island analysis

---

### Step 2: Layer Pre-filter (GitHub API)

For each candidate, get commits between `last_failing_sha` and `first_passing_sha`:
```
GET /repos/tenstorrent/tt-metal/compare/{last_failing_sha}...{first_passing_sha}
```

Determine test layer from `test_filepath`:
- `models/` or `demos/` → layer 4
- `ttnn/` or `tests/ttnn/` → layer 3
- `tt_metal/impl/`, `tt_metal/api/`, `tt_metal/distributed/`, `tt_metal/fabric/`, `tests/tt_metal/` → layer 2
- `tt_metal/hw/` → layer 1
- Other (CI, docs, cmake) → skip

For each commit in the range, determine `fix_layer` = lowest-numbered layer among files changed.

**Classification:**
- If any commit has `fix_layer < test_layer` → **vertical candidate** (include in vertical pipeline)
- If any commit has `fix_layer == test_layer` → **horizontal candidate** (include in horizontal pipeline)
- If same commit qualifies for both → set `candidate_type: "both"`, run both pipelines, let evidence decide
- If no commit meets either condition → **discard** (not an escape)

**Fetch full diffs for cross-layer or same-layer commits:**
```
GET /repos/tenstorrent/tt-metal/commits/{sha}
```
The `files[].patch` content is required for Opus in Step 3. Pass the full diff; if >500 lines, pass first 500 and note truncated.

**Commit intent signal**: Note if any candidate commit has "fix", "bug", "revert", "restore", "correct" in message. If ZERO fix-intent commits → set `low_confidence_prior = true`. Proceed to Step 3 but lower confidence.

---

### Step 3: Opus Pre-classification

#### 3a: Noise blocklist (before calling Opus)

Skip to `status: "skipped_prefilter"` if any of these match the failure log:
- `TracyAlloc|TracyFree|FrameMarkNamed|FrameMarkStart`
- `tenstorrent_pcie_ioctl|ioctl.*pcie.*fail`
- `FileNotFoundError.*No such file or directory.*libtt_`
- Zero pytest/gtest identifiers AND no `TT_FATAL`/`TT_THROW`

#### 3b: Fetch failure log

```
GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs
```
Use first 200 lines of the relevant test step.

#### 3c: Vertical Opus prompt (high causal bar)

Provide: raw failure log, cross-layer commits with **full diffs**, test filepath/name, streak stats, `low_confidence_prior`.

```
You are analyzing a potential vertical bug escape in tenstorrent/tt-metal CI.

YOUR PRIMARY TASK: DETERMINE WHETHER THIS IS A VERTICAL ESCAPE

A vertical escape means a bug at layer N caused a test at layer M (N < M) to fail.
You must trace a SPECIFIC CAUSAL MECHANISM:
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

Work through these 5 checks in order:
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

**Post-Opus validation:** If `specific_change`, `call_path`, or `error_match` is null, generic, or uses "might/possibly/could" → override to `SKIP_UNRELATED`, mark `status: "skipped_vague_justification"`.

**MANDATORY layer check:** Before dispatching, confirm `most_likely_fix_sha` has `fix_layer < test_layer`. Same-layer → horizontal, not vertical. Higher-layer → inverse, discard.

#### 3d: Horizontal Opus prompt (lower causal bar)

For horizontal candidates, the causal bar is lower — same-layer means the commit is more obviously related.

```
You are analyzing a potential horizontal bug escape in tenstorrent/tt-metal CI.

A horizontal escape means a bug at layer N caused a test at the SAME layer N to fail.
The fix and the test are in the same codebase layer, so the causal link is typically direct.

Required analysis (lower bar than vertical):
  (a) same_component: Does the commit modify code in the same component/module/subsystem the test exercises?
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

If `same_component` is null or the diff has zero overlap with the test's component → `SKIP_UNRELATED`.

---

### Step 3.5: PR-Linked CI Proof Check

Before dispatching any probe, check if a PR already proves the fix. This step runs AFTER Opus classification passes and BEFORE bisect dispatch.

1. For each commit in the SHA range (or just the `most_likely_fix_sha` from Opus), check for an associated PR:
   ```
   gh pr list --search {sha} --state merged
   ```

2. If a PR is found, **you MUST run the verification script** — do NOT make this judgment yourself:
   ```
   python3 /workspace/group/bug-escapes-data/verify-pr-proof.py --pr {PR_NUMBER} --test {TEST_NAME}
   ```
   This script calls an isolated Opus agent (cold context, no campaign history) whose sole job is to determine whether the PR body or comments contain a link to a CI run where the specific test explicitly passed. Your own reading of the PR does not count.

3. **If the script exits 0 (CONFIRMED):**
   - Paste the stdout JSON into `campaign-state.json` as `pr_body_verification`.
   - Record `confirmation_method: "pr_ci_proof"`, `confidence: "proven"`.
   - You still need BEFORE=FAIL evidence. Check if existing nightly CI logs cover `last_failing_sha`. If yes, record those run IDs. If no, dispatch a BEFORE probe only.
   - **Do NOT write to Confluence without `pr_body_verification` present in campaign-state.json.**
   - Proceed to Step 6 (Record).

4. **If the script exits 1 (BLOCKED) or exits 2 (error):** Fall through to Step 4 (bisect). Do not record `pr_ci_proof`.

> **⚠️ DIFF INSPECTION IS NOT PROOF.** Reading a commit diff and deciding it "looks like" a fix is NOT sufficient to skip probing, for either vertical or horizontal escapes.
>
> **⚠️ YOUR OWN READING OF THE PR IS NOT PROOF.** Even if you believe the PR proves it, you must run `verify-pr-proof.py` and get exit 0 before recording `pr_ci_proof`. The isolated Opus agent's judgment is the gate, not yours.
>
> "The diff removes the failing call" is not proof. "The PR seems to mention the test" is not proof. If `verify-pr-proof.py` exits nonzero, fall through to Step 4 and probe.

---

### Step 4: Manual Binary Search (find the fix commit)

**⚠️ DO NOT use `bisect-dispatch.yaml`** — that finds *breaking* commits, not *fixing* commits.
**⚠️ Read existing CI logs FIRST before dispatching anything.** Every midpoint probe costs 60+ minutes.

#### Step 4a: Read existing CI logs

1. List runs of the relevant workflow on main:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows/{id}/runs?branch=main&per_page=100
   ```
2. For each midpoint commit, check if a run exists at that SHA.
3. If yes: read job log for the test — PASS or FAIL. This IS your probe result.
4. Only dispatch when no existing run covers the midpoint.

#### Step 4b: Probe branch dispatch (only when needed)

1. Create branch `brain/{escape-id}-bisect-probe` in a worktree.
2. Try artifact reuse first:
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs?head_sha={target_sha}
   ```
   If artifacts exist and target SHA is within ~5 commits, dispatch with artifact reuse.
3. If no artifacts: rebase probe branch to target SHA, dispatch.
4. After dispatch: MANDATORY post-dispatch actions (Confluence + DM Evan) BEFORE reading results.

**Algorithm:**
1. Get commits: `GET /repos/tenstorrent/tt-metal/compare/{last_failing}...{first_passing}` — `commits` array, index 0 = oldest
2. `mid = len(commits) // 2`
3. Probe `commits[mid].sha` (4a or 4b)
4. PASS → fix is in lower half: `high = mid - 1`
5. FAIL → fix is in upper half: `low = mid + 1`
6. Repeat until `low == high` — that is the fix commit
7. All midpoints PASS → fix commit is `commits[0]`

Save each midpoint run ID to `campaign-state.json`.

#### "Assume Horizontal" Rule (HORIZONTAL ONLY)

> **⚠️ This rule requires a genuine attempt at probing first.** Do NOT apply it because the diff looks convincing. "The commit modifies the failing test" is not sufficient — you must have actually tried and failed to narrow further via existing CI logs or dispatched probes.

For horizontal candidates, if bisect **genuinely cannot narrow** to a single commit (existing logs exhausted, probes inconclusive), apply this rule when ALL five conditions hold:

1. Confirmed failure-to-pass transition (thresholds met, single error signature, adjacency, gap).
2. At least one commit in the range has `fix_layer == test_layer`.
3. **Bisect was genuinely attempted**: existing CI logs were read for all midpoints AND/OR probes were dispatched. Diff inspection alone does NOT satisfy this condition.
4. The test is **NOT** currently failing on main.
5. No vertical escape was identified for the same candidate (vertical takes priority).

If all five hold: mark `status: "confirmed_horizontal"`, `confirmation_method: "assumed_horizontal"`, `confidence: "assumed"`. Record `candidate_fix_commits` as the list of same-layer commits in the range.

This rule does NOT apply to vertical candidates. Vertical always requires a proven fix commit.

---

### Step 5: Verification

**Vertical and Horizontal:** Create two branches:
- `brain/escape-before-{escape_id}` at `git checkout {fix_commit}^` (parent)
- `brain/escape-after-{escape_id}` at `git checkout {fix_commit}` (fix commit)

Dispatch the appropriate test workflow for each.

**After MANDATORY post-dispatch actions (Confluence + DM), wait for results.**

Verdict logic:
- BEFORE=FAIL + AFTER=PASS → **confirmed escape** ✅ (`confirmation_method: "bisect"`)
- BEFORE=FAIL + AFTER=FAIL → **refuted** (fix didn't address this test)
- BEFORE=PASS + AFTER=anything → **refuted** (test wasn't failing at that commit)
- Either times out → **inconclusive_timeout** (retry next night)

Write verdict + all run IDs to `campaign-state.json`.

Delete BEFORE/AFTER branches from GitHub after confirmation (or mark for deletion if refuted).

**Note:** For horizontal candidates that reached this step via `assumed_horizontal`, step 5 is SKIPPED. For horizontal candidates that were narrowed to a single commit via bisect, step 5 applies normally.

---

### Step 6: Record Confirmed Escape

Append to `confirmed-escapes.json`:
```json
{
  "escape_id": "...",
  "escape_type": "vertical|horizontal",
  "test_name": "...",
  "test_filepath": "...",
  "test_layer": 3,
  "fix_commit_sha": "...",
  "fix_commit_message": "...",
  "fix_layer": 2,
  "candidate_fix_commits": [],
  "confirmation_method": "bisect|pr_ci_proof|assumed_horizontal",
  "confidence": "proven|assumed",
  "last_failure_run_id": "...",
  "first_success_run_id": "...",
  "before_run_id": "...",
  "after_run_id": "...",
  "reasoning": "...",
  "confirmed_at": "2026-05-27T..."
}
```

Write escape ID to `seen-escapes.json` with `status: "confirmed"`.

---

### Step 7: Confluence Updates

**Page IDs:**
- `2424012846` — TT-Metal Vertical Bug Escapes (main tracking)
- `2428895268` — Vertical Bug Escapes Confirmed Table
- `2461597780` — Horizontal Bug Escapes (main tracking)
- `2460254399` — Confirmed Horizontal Bug Escapes
- `2440298505` — Bug Escape Chart

Always fetch the current page content before updating (to avoid clobbering concurrent writes).

**For vertical escapes:**
- Tracking page 2424012846: add/update row in the appropriate section
- Confirmed table 2428895268: add row when confirmed

**For horizontal escapes:**
- Tracking page 2461597780: add/update row
- Confirmed table 2460254399: add row when confirmed

**After every confirmed escape (vertical or horizontal):** Update the chart page 2440298505:
- Vertical: increment the off-diagonal cell at `[fix_layer_row][test_layer_col]`
- Horizontal: increment the diagonal cell at `[layer_row][layer_col]` where row==col

**Layers column format:**
- Under Investigation (vertical): `Test L{X} → Suspected Fix L{Y}` (Y < X)
- Under Investigation (horizontal): `Test L{X} → Suspected Fix L{X}` (same layer)
- Confirmed (vertical): `Test L{X} → Verified Fix L{Y}`
- Confirmed (horizontal, proven): `Test L{X} → Verified Fix L{X} (horizontal, proven)`
- Confirmed (horizontal, assumed): `Test L{X} → Assumed Fix L{X} (horizontal, assumed)`

---

## Cron Schedule

The cron fires **hourly** in both backfill and incremental modes. Each session does NOT follow a fixed clock schedule — it picks up from `current_step` in campaign-state.json and advances as far as possible within the session.

Typical progression across multiple hourly sessions:
- Session 1: Step 1 (Snowflake scan) + Steps 2–3 (layer filter + Opus)
- Session 2: Step 3.5 (PR proof check) + Step 4 dispatch (bisect probes)
- Session 3: Step 4 poll + Step 5 dispatch (verification)
- Session 4: Step 5 poll + Steps 6–7 (record + Confluence)

**De-duplication:** Check `seen-escapes.json` before processing any candidate. If escape ID already exists, skip.

---

## Context Reset Protocol

**Trigger:** Context compression detected (summary block at top of conversation, or working state is empty/inconsistent).

**Recovery steps (in order, no skipping):**

1. Read `MEMORY.md` — recover file paths, Confluence page IDs, constraints.
2. Read `campaign-state.json` — recover current step, active candidates, pending run IDs.
3. For each candidate with `status: "awaiting_probes"` or `status: "bisecting"`: poll GitHub Actions for status of all `active_runs` run IDs via `gh run view {run_id}`. Do NOT dispatch anything until existing run statuses are confirmed.
4. For each candidate with `confluence_updated: false` OR `dm_sent: false`: complete those notifications before starting new work.
5. Resume from `current_step` for the highest-priority incomplete candidate.
6. If `campaign-state.json` is missing or corrupt: read `seen-escapes.json` + `confirmed-escapes.json` to reconstruct what's been processed. Start Step 1 for candidates not in those files.

**INVARIANT:** Never dispatch a probe without first confirming no duplicate run is already in flight for that candidate in `campaign-state.json`.

---

## Hardware-Specific Dispatch Rules

### Single-card jobs (N150, N300, P150, P300)
- Verification: BEFORE and AFTER can run **in parallel**
- Bisect: sequential (one probe at a time)

### Multi-card jobs (T3K, Galaxy)
- **Only 1 run at a time per machine type** — never dispatch two T3K or two Galaxy runs simultaneously
- Verification: dispatch BEFORE, **wait for it to complete**, then dispatch AFTER. Strictly sequential.
- Weekends: T3K and Galaxy may run during the day (not only at night). Still 1-at-a-time rule.

---

## Subagent Rules (MANDATORY)

- ALL subagents MUST be launched with `run_in_background: true`
- Subagents MUST NOT commit or push code to any repository
- Subagents MUST NOT dispatch GitHub Actions workflow runs
- Subagents MUST NOT cancel or modify in-progress workflow runs

Only the main BrAIn session may: commit/push code, dispatch workflows, cancel runs.
Subagents may only: read data (Snowflake, GitHub API, logs), analyze, and return findings.

---

## What NOT to Do

- Do NOT dispatch two bisects simultaneously for the same hardware type
- Do NOT start verification without a completed bisect result (vertical) or assumed_horizontal rule (horizontal)
- Do NOT skip writing to `seen-escapes.json` even for fast refutals
- Do NOT delete branches until verification is complete
- Do NOT apply "assume horizontal" rule to vertical candidates
- Do NOT update Confluence with unconfirmed candidates
- Do NOT dispatch a probe without reading `campaign-state.json` first
- Do NOT treat Snowflake SHAs as authoritative — always verify against CI logs

---

## Continuous Campaign Rule (MANDATORY)

When a campaign ends with no bisects in progress and no pending candidates: immediately start a new one. Do not wait for the next scheduled cron. Query Snowflake for fresh candidates right away.

If a status update contains only "campaign is done" with no new runs dispatched or candidates found, that update is a failure. Always end each work session with either (a) active runs in flight, or (b) a fresh Snowflake scan producing new candidates, or (c) an explicit explanation of why neither is possible.

---

## Backfill Mode Notes

1. Run Step 1 with 90-day window, thresholds ≥10/≥10/72h. Expect 150-300 candidates.
2. Layer pre-filter (Step 2): run both vertical and horizontal classification paths simultaneously.
3. Opus (Step 3): batch process max 8 candidates at a time (rate limit). Vertical and horizontal prompts are different.
4. Step 3.5 (PR proof): run before any probe dispatch. Significant time savings.
5. Bisect: queue by hardware type, dispatch in nightly windows. 3 bisects per night max to avoid runner contention.
6. Verification: same nightly cadence. Single-card parallel, multi-card sequential.
7. Process newest candidates first (ORDER BY `last_fail_ts DESC`) — oldest candidates have highest log expiry risk.
8. For candidates >60 days old where CI logs have expired: apply "assume horizontal" more aggressively if same-layer commits exist. For vertical: require strong Opus causal evidence even without logs.

---

## Watchdog Task (run at 07:00Z)

Check `campaign-state.json`:
- Any candidate stuck in `step: "bisect"` with `dispatch_time` > 4 hours ago → log as `inconclusive_timeout`
- Any candidate stuck in `step: "verification"` with `dispatch_time` > 3 hours ago → log as `inconclusive_timeout`
- Post campaign summary (confirmed/refuted/inconclusive counts this run) to Slack
