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
Lighter threshold (2+ consecutive failures → 3+ consecutive passes).
Skip Opus pre-classification if layer pre-filter already rules it out.

---

## Pipeline Steps

### Step 1: Detection (Snowflake)

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
),
-- Most recent transition per test, joined back to get test metadata
result AS (
  SELECT
    t.CICD_TEST_CASE_ID,
    r.test_name,
    r.test_filepath,
    t.last_failing_sha,
    t.first_passing_sha,
    t.consecutive_fail_count,
    t.consecutive_pass_count,
    t.last_fail_ts,
    t.first_pass_ts,
    t.norm_signature
  FROM transitions t
  JOIN (SELECT DISTINCT CICD_TEST_CASE_ID, test_name, test_filepath FROM runs) r
    ON t.CICD_TEST_CASE_ID = r.CICD_TEST_CASE_ID
  QUALIFY ROW_NUMBER() OVER (PARTITION BY t.CICD_TEST_CASE_ID ORDER BY t.last_fail_ts DESC) = 1
)
SELECT *
FROM result
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
If any commit does → proceed to Step 3.

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
2. The cross-layer commits from Step 2 (SHA, message, files changed, diff if available)
3. The test filepath, name, and streak stats from Step 1

```
You are analyzing a potential vertical bug escape in tenstorrent/tt-metal CI.

Test: {test_name}
File: {test_filepath} (layer {test_layer})
Streak: {consecutive_fail_count} consecutive failures → {consecutive_pass_count} consecutive passes after fix
Failure window: {last_fail_ts} to {first_pass_ts}

Cross-layer commits between last failure and first pass:
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

CHECK 3 — Causal analysis:
  Is there a plausible causal link between one of the cross-layer commits and this specific failure?
  Look for: the commit touches code in the call path of the failing test, or changes an API/ABI
  that the test exercises. "Plausible" means you can name the mechanism.
  Red flags: hardware-specific errors (temp sensor, PCI link), timing-dependent races,
  external service failures (network, registry).
  → If no plausible causal link: verdict SKIP_UNRELATED. Stop.

CHECK 4 — Range size sanity:
  How many commits are in the bisect range? If > 50, the range is likely too wide to bisect
  efficiently unless Opus's causal candidate is very specific (single commit).
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
  "reasoning": "<2-3 sentences: what the error is, which commit is likely the fix and why>"
}
```

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

**⚠️ DO NOT use `bisect-dispatch.yaml`** — that workflow finds *breaking* commits (regressions),
not *fixing* commits. For bug escapes we want the first commit where the test started PASSING.

#### Method selection

Two dispatch methods are available. Choose based on the test type:

**Method A — `test-dispatch.yaml`** (workflow ID 103409066)
Use for tests that run in standard CI jobs (unit tests, fast-dispatch, etc.).
Each run dispatches directly at a target commit with no build overhead if that commit has existing artifacts.

**Method B — `perf-device-models.yaml` + artifact reuse** ← USE FOR DEVICE-PERF TESTS
Use when Snowflake shows the failing job name is `device-perf / N300 WH B0 Set 2 device perf`
(or any other `device-perf / *` job). These tests require a Tracy/profiler-enabled build and
**cannot** use `test-dispatch.yaml`.

Method B procedure:
1. Find the device-perf merge gate run ID for the target bisect commit:
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs?head_sha={target_sha}
   ```
   Look for a run named "Merge Gate" (or the scheduled device-perf run at that commit).
   Note: the merge gate builds **non-profiler** artifacts — they cannot be reused here.
   You need a prior **device-perf** run that built profiler artifacts at a nearby commit.

2. Check the device-perf workflow runs near your target commit for profiler artifacts:
   ```
   GET /repos/tenstorrent/tt-metal/actions/workflows/76728129/runs?branch=main
   ```
   For each run, check `/artifacts` for names containing `_profiler_`.
   The artifact name format is:
   `TTMetal_build_any_22.04_amd64_x86_64-linux-clang-20-libstdcpp_profiler_{SHA}_{RUN_ID}`
   and `ttnn-dist-cp310-Release-profiler-{SHA}-{RUN_ID}`

3. If profiler artifacts exist for a **nearby commit** (within ~5 commits of target):
   Dispatch `perf-device-models.yaml` on the `brain/efficientnet-bisect-probe` branch
   with `use-artifacts-from-run = {that run ID}`. The test will run using that commit's
   compiled code, not the branch's code.
   ⚠️ This tests the artifact commit's code, not your probe branch — only use when the
   artifact commit is close enough to the target that the difference is irrelevant.

4. If NO profiler artifacts exist near the target commit:
   - Update the `brain/efficientnet-bisect-probe` branch to base on the target commit
     (keeping the workflow changes), push it, then dispatch with `use-artifacts-from-run = ""`.
   - This triggers a full rebuild (~60 min) but correctly tests that exact commit.
   - Command sequence:
     ```bash
     cd /workspace/group/worktrees/efficientnet-bisect-probe
     # Save the workflow change commit hash first
     WORKFLOW_COMMIT=$(git rev-parse HEAD)
     git reset --hard {target_sha}
     git cherry-pick $WORKFLOW_COMMIT
     git push origin brain/efficientnet-bisect-probe --force
     ```

5. The overall job will likely **report failure** even when the specific pcc test passes
   (perf regression on another model, infra issue, etc.). You MUST check actual test logs:
   ```
   GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs
   ```
   Find the `device-perf / N300 WH B0 Set 2 device perf` job, then:
   ```
   GET /repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs
   ```
   Search for `test_efficientnetb0_model` and look for `PASSED` or `FAILED`.

6. Dispatch inputs for `perf-device-models.yaml` (workflow ID 76728129):
   ```json
   {
     "ref": "brain/efficientnet-bisect-probe",
     "inputs": {
       "architecture": "[\"wormhole_b0\"]",
       "requested-models": "[\"efficientnetb0\"]",
       "platform": "Ubuntu 22.04",
       "build-type": "Release",
       "use-artifacts-from-run": "{run_id or empty string}"
     }
   }
   ```

Use `test-dispatch.yaml` (workflow ID 103409066) to run the test at individual commits.

**Algorithm:**
1. Get the commit list between `last_failing_sha` (exclusive) and `first_passing_sha` (inclusive):
   ```
   GET /repos/tenstorrent/tt-metal/compare/{last_failing_sha}...{first_passing_sha}
   ```
   → `commits` array, index 0 = oldest (first after last_failing), last = first_passing
2. Pick midpoint index `mid = len(commits) // 2`
3. Dispatch test-dispatch.yaml at `commits[mid].sha`:
   ```
   POST /repos/tenstorrent/tt-metal/actions/workflows/test-dispatch.yaml/dispatches
   {
     "ref": "main",
     "inputs": {
       "arch":         "{wormhole_b0 or blackhole}",
       "runner-label": "{e.g. [\"BH-LoudBox\"] or [\"config-t3000\"]}",
       "command":      "{pytest command -x}",
       "commit":       "{commits[mid].sha}",
       "description":  "BrAIn bisect {escape_id} mid@{commits[mid].sha[:8]}"
     }
   }
   ```
4. After MANDATORY post-dispatch actions (Confluence + DM Evan), wait for result.
5. If **PASS** → fix is in the lower half: set `high = mid - 1`
6. If **FAIL** → fix is in the upper half: set `low = mid + 1`
7. Repeat from step 2 with `commits[low..high]`
8. When `low == high` (or range is empty), the fix commit is `commits[low]`
   - Special case: if ALL midpoints PASS → fix commit is `commits[0]` (first after last_failing)

Save each midpoint run ID to `campaign-state.json`.

**Arch determination:**
- BH-LoudBox / topology-6u → `blackhole`
- N150, N300, P150, P300, config-t3000 → `wormhole_b0`

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
| 892871__a249d854 | test_wan_decoder.py (layer 4) | T3K WAN tests | T3K (WH) | [a249d854](...) | [b3e8eb62](...) | Test L4 → Suspected Fix L3 | mid1@b2670a20 [run_id] ⏳ | PROCEED_TO_BISECT (MEDIUM) | Commit c53fc338 fixes nlp_create_qkv_heads precision. | 🔍 Under Investigation — mid1 dispatched {date} |
```

**After Step 4 (bisect completes) → update the entry, still in "Under Investigation":**
Add fix commit and bisect proof link. Update status to "Under Investigation — verification dispatched {date}".

**After Step 5 (verification complete):**
- CONFIRMED → move entry to "✅ Confirmed Escapes", update status, add before/after run links
- REFUTED → move entry to "❌ Refuted", update status, add explanation of why it was refuted
- INCONCLUSIVE_TIMEOUT → move entry to "⏳ Inconclusive", update status, note retry scheduled

**Confirmed escapes table** (page 2428895268) uses this header:
```
| Escape ID | Test File | Job | HW | Last Fail | Last Fail Commit | First Pass | First Pass Commit | Fail Rate | Layers | Runs Dispatched | Opus Verdict | Opus Reasoning | Status |
```

Layers column value for confirmed escapes: `Test L{X} → Verified Fix L{Y}`

**Entry format once confirmed** (add row to 2428895268; update row in 2424012846):
- Layers: `Test L{test_layer} → Verified Fix L{fix_layer}`
- Status: `✅ Confirmed — Fix: [PR #{pr_number}](...) {fix_commit_message}. Confirmed {date}`

Extended bullet-point format (for notes outside the table):
```
### Escape #{N}: {fix_commit_message[:60]}

- *Test*: {test_name}
- *Test layer*: {test_layer}
- *Fix layer*: {fix_layer}
- *Layers*: Test L{test_layer} → Verified Fix L{fix_layer}
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

## Continuous Campaign Rule (MANDATORY)

**When a campaign ends with no bisects_in_progress and no pending candidates, immediately start a new one.** Do not wait for the next scheduled cron. Query Snowflake for fresh candidates right away. Spending cron cycles posting "all done" without starting new campaigns is wasted time.

**If a status update contains only "campaign is done" with no new runs dispatched or candidates found, that update is a failure.** Always end each work session with either (a) active runs in flight or (b) a fresh Snowflake scan producing new candidates, or (c) an explicit explanation of why neither is possible.

---

## Hardware-Specific Dispatch Rules

### Single-card jobs (N150, N300, P150, P300)
- Method B (verification): BEFORE and AFTER can be dispatched **in parallel**, as long as no more than one set of verification per machine type is running at a time.
- Method A (binary search): always one run at a time (inherent to binary search).

### Multi-card jobs (T3K, Galaxy)
- **Only 1 run at a time per machine type** — never dispatch two runs simultaneously on T3K or Galaxy.
- Method B (verification): dispatch BEFORE first, **wait for it to complete**, then dispatch AFTER. Strictly sequential.
- Method A (binary search): unchanged — already sequential by nature (one run per midpoint).
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
