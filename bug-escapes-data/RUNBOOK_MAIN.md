# Bug Escape Detection — Main Session Runbook

**Role:** You are the main BrAIn session. You orchestrate the campaign: spawn the subagent, interpret its findings, dispatch verification probes, record confirmed escapes, and update Confluence. You are the only one who dispatches workflows, commits code, or writes state files.

---

## State Files (all in /workspace/group/bug-escapes-data/)

| File | Purpose |
|------|---------|
| `seen-escapes.json` | Set of escape IDs already analyzed. Prevents re-analysis. |
| `confirmed-escapes.json` | All confirmed escapes (source of truth for Confluence). |
| `campaign-state.json` | Active campaign: candidates, active run IDs, current step. |
| `RUNBOOK_SUBAGENT.md` | Instructions for the Opus subagent. |
| `RUNBOOK_MAIN.md` | This file. |

**Escape ID format**: `{test_case_id}__{last_failing_sha[:8]}`

---

## Session Locking Protocol

`campaign-state.json` contains a `lock` field.

**When starting:**
1. Read campaign-state.json.
2. If `lock.locked == true` AND `lock.expires_at` is in the future → **stand down. DM @ebanerjeeTT: "not working — another session is active."**
3. If unlocked (or expired): set `lock.locked = true`, `lock.locked_at = now`, `lock.expires_at = now + 90 minutes`, `lock.held_by = "brief description"`. Write before proceeding.

**While working:** Refresh `lock.expires_at = now + 90 minutes` every 15 minutes.

**When done:** Set `lock.locked = false`, clear all lock fields. Write to file.

**Expired lock:** Treat as unlocked — acquire and proceed.

---

## MANDATORY DISPATCH RULES

After **every** `workflow_dispatch` — no exceptions:

0. **Write to `campaign-state.json` FIRST** — record run ID, URL, candidate escape_id, current step. Do this before Confluence or Slack.
1. **Update the appropriate Confluence tracking page IMMEDIATELY** — run ID, URL, ⏳ status.
2. **DM @ebanerjeeTT** — run URL + one-line description of what it's testing.

### State Persistence Rule

After EVERY significant action, write state before proceeding:
- Dispatching a probe → write run ID to campaign-state.json immediately
- Receiving a probe result → write verdict before updating Confluence
- Classifying an escape → write candidate status update
- Confirming/refuting → write to confirmed-escapes.json and seen-escapes.json before Confluence

Context compression can happen at any moment. **Write first, then act on the result.**

---

## Step 1: Spawn the Subagent

Spawn an Opus subagent with `run_in_background: true` using `RUNBOOK_SUBAGENT.md` as its instructions.

**Subagent launch template:**

```
You are a bug escape detection subagent. Follow the instructions in RUNBOOK_SUBAGENT.md exactly.

State files:
- seen-escapes.json: /workspace/group/bug-escapes-data/seen-escapes.json
- confirmed-escapes.json: /workspace/group/bug-escapes-data/confirmed-escapes.json
- RUNBOOK_SUBAGENT.md: /workspace/group/bug-escapes-data/RUNBOOK_SUBAGENT.md

Campaign mode: backfill (90-day window)
Target: find 1 high-confidence escape candidate (horizontal or vertical)

Work through Steps 1–6 of RUNBOOK_SUBAGENT.md. Return your findings as a single JSON object in the format specified at the end of RUNBOOK_SUBAGENT.md. Do not write to any state files. Do not dispatch any workflows.
```

Adjust `target` as needed. You may spawn multiple subagents in parallel for different test layer ranges if you want broader coverage.

---

## Step 2: Interpret Subagent Output

The subagent returns a JSON object with `findings` and `rejected` arrays. For each finding:

| `action_required` | What to do |
|---|---|
| `assume_horizontal` | Go directly to Step 5 (Record). No probes needed. |
| `pr_ci_proof_check` | Run `verify-pr-proof.py` first (Step 3), then decide. |
| `probe_verification` | Dispatch BEFORE/AFTER probes (Step 4). |

Before acting on any finding:
1. Check `seen-escapes.json` — if escape_id already present, skip.
2. Update campaign-state.json to add the candidate with `status: "opus_classified"`.

---

## Step 3: PR-Linked CI Proof Check (Optional Fast Path)

If the subagent flagged `action_required: "pr_ci_proof_check"`:

```
python3 /workspace/group/bug-escapes-data/verify-pr-proof.py --pr {PR_NUMBER} --test {TEST_NAME}
```

- Exit 0 (CONFIRMED): record `confirmation_method: "pr_ci_proof"`, `confidence: "proven"`. Go to Step 5 (Record). You still need BEFORE=FAIL evidence — check if existing nightly CI logs cover `last_failing_sha`. If yes, record those run IDs. If no, dispatch a BEFORE probe only.
- Exit 1 (BLOCKED) or Exit 2 (error): fall through to Step 4 (probes).

**⚠️ Do NOT update Confluence without `pr_body_verification` present in campaign-state.json.**

---

## Step 4: Verification Probes

### 4a: Check for Artifact Reuse

Before dispatching, check if Merge Gate or other CI runs built artifacts at `fix_commit_sha^` and `fix_commit_sha`:

```
GET /repos/tenstorrent/tt-metal/actions/runs?head_sha={sha}&per_page=100
```

Look for a `Merge Gate` run. If found, check its artifacts:
```
GET /repos/tenstorrent/tt-metal/actions/runs/{merge_gate_run_id}/artifacts
```

Artifact reuse is possible if: build artifact exists, is not expired, and was NOT built with tracy (the Merge Gate builds without tracy by default). If reusing: set `tracy: false` in the workflow modification.

### 4b: Create Probe Branches

```bash
# Create BEFORE branch at fix_commit_sha^ (parent of fix)
gh api repos/tenstorrent/tt-metal/git/refs -X POST \
  -f ref="refs/heads/brain/escape-before-{escape_id}" \
  -f sha="{fix_commit_sha_parent}"

# Create AFTER branch at fix_commit_sha
gh api repos/tenstorrent/tt-metal/git/refs -X POST \
  -f ref="refs/heads/brain/escape-after-{escape_id}" \
  -f sha="{fix_commit_sha}"
```

### 4c: Modify Workflow for Artifact Reuse (if applicable)

If artifact reuse is available, modify the nightly workflow on both probe branches to add `use-artifacts-from-run` as a dispatch input and thread it into the `build-artifact` call. Also set `tracy: false` to match the Merge Gate artifact.

Use the GitHub Contents API to update `.github/workflows/{appropriate_workflow}.yaml` on each branch.

### 4d: Select the Correct Workflow and Inputs

Use the `workflow_name` and `hardware`/`architecture` from the subagent's finding:

- The subagent records `workflow_name` (e.g. "Nightly tt-metal L2 tests") and `workflow_id` from the original failing run.
- Match this to the correct workflow file.
- Use the `architecture` from the finding: `["wormhole_b0"]` or `["blackhole"]`.
- Use the correct test category flag — NOT a generic one. For example:
  - LLK C++ tests → `run_sd_unit_tests: true`
  - Python data_movement tests → `additional_test_categories: "data_movement"`
  - Python ttnn conv tests → `additional_test_categories: "conv"`
  - Verify the test file path against the workflow's category flags

**Never assume `run_sd_unit_tests` — always match the test file path to the correct flag.**

### 4e: Dispatch

```bash
gh api repos/tenstorrent/tt-metal/actions/workflows/{workflow_file}/dispatches \
  -X POST --input - <<EOF
{
  "ref": "brain/escape-before-{escape_id}",
  "inputs": {
    "architecture": "[\"wormhole_b0\"]",
    "additional_test_categories": "data_movement",
    "use-artifacts-from-run": "{merge_gate_run_id_before}"
  }
}
EOF
```

For single-card hardware (N150, N300, P150, P100, P150b): BEFORE and AFTER can run **in parallel**.
For multi-card hardware (T3K, Galaxy): dispatch BEFORE, **wait for completion**, then dispatch AFTER.

**MANDATORY after each dispatch:** Write run IDs to campaign-state.json → update Confluence → DM @ebanerjeeTT.

### 4f: Verdict Logic

- BEFORE=FAIL + AFTER=PASS → **confirmed escape** ✅ (`confirmation_method: "bisect"`, `confidence: "proven"`)
- BEFORE=FAIL + AFTER=FAIL → **refuted**
- BEFORE=PASS + AFTER=anything → **refuted**
- Either times out → **inconclusive_timeout** (retry next night)

Delete BEFORE/AFTER branches after verdict.

---

## Step 5: Record Confirmed Escape

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
  "fix_layer": 3,
  "candidate_fix_commits": [],
  "fix_pr": "...",
  "confirmation_method": "bisect|pr_ci_proof|assumed_horizontal",
  "confidence": "proven|assumed",
  "last_failure_run_id": "...",
  "last_failure_job_id": "...",
  "first_success_run_id": "...",
  "first_success_job_id": "...",
  "before_run_id": "...",
  "after_run_id": "...",
  "reasoning": "...",
  "confirmed_at": "2026-..."
}
```

Write escape_id to `seen-escapes.json` with appropriate status.

---

## Step 6: Confluence Updates

**Page IDs:**
- `2424012846` — TT-Metal Vertical Bug Escapes (main tracking)
- `2428895268` — Vertical Bug Escapes Confirmed Table
- `2461597780` — Horizontal Bug Escapes (main tracking)
- `2460254399` — Confirmed Horizontal Bug Escapes
- `2440298505` — Bug Escape Chart

Always fetch current page content before updating (avoid clobbering concurrent writes).

**For vertical escapes:** Update 2424012846 and 2428895268.
**For horizontal escapes:** Update 2461597780 and 2460254399.
**After any confirmed escape:** Update chart page 2440298505.

**Layer format:**
- Under Investigation (vertical): `Test L{X} → Suspected Fix L{Y}` (Y < X)
- Under Investigation (horizontal): `Test L{X} → Suspected Fix L{X}`
- Confirmed (bisect/pr_ci_proof): `Test L{X} → Verified Fix L{Y} (proven)`
- Confirmed (assumed_horizontal): `Test L{X} → Assumed Fix L{X} (assumed)`

---

## Messaging Rules

### Always DM @ebanerjeeTT

1. Every `workflow_dispatch` — run URL + one-line description.
2. Every confirmed escape — escape ID, test name, layer type, fix PR, method (proven vs assumed).
3. Blockers — Snowflake unreachable, probe timing out 2+, unresolvable structural issue.

### Daily summary at 08:00Z

- N new escapes since last summary (vertical / horizontal counts)
- Running campaign totals
- Probe queue depth
- Stuck/blocked candidates

### Never DM

- Refuted or skipped candidates
- Routine poll cycles
- Between-session state saves or Confluence updates
- Zero-result Snowflake scans

---

## Hardware-Specific Dispatch Rules

**Single-card (N150, N300, P150, P300, P100, P150b):** BEFORE and AFTER can run in parallel.

**Multi-card (T3K, Galaxy):** BEFORE first, wait for completion, then AFTER. Only 1 run at a time per machine type.

---

## Context Reset Protocol

If context was compressed:

1. Read MEMORY.md — recover file paths, Confluence IDs, constraints.
2. Read campaign-state.json — recover current step, active candidates, run IDs.
3. For each candidate with `status: "awaiting_probes"` or `"bisecting"`: poll `gh run view {run_id}`. Do NOT dispatch until existing run statuses confirmed.
4. For each candidate with `confluence_updated: false` OR `dm_sent: false`: complete those before new work.
5. Resume from `current_step` for highest-priority incomplete candidate.
6. If campaign-state.json is missing: read seen-escapes.json + confirmed-escapes.json to reconstruct. Spawn subagent for fresh candidates.

**INVARIANT:** Never dispatch a probe without first confirming no duplicate run is already in flight.

---

## Continuous Campaign Rule

When no bisects are in progress and no candidates are pending: spawn a new subagent immediately. Do not wait for the next cron. A session that ends with no new runs dispatched and no new candidates is a failure unless it explicitly explains why neither is possible.

---

## What NOT to Do

- Do not dispatch two bisects simultaneously for the same hardware type
- Do not start verification without a completed bisect or assumed_horizontal ruling
- Do not skip writing to seen-escapes.json even for fast refutals
- Do not delete branches until verification is complete
- Do not apply "assume horizontal" to vertical candidates
- Do not update Confluence with unconfirmed candidates
- Do not dispatch a probe without reading campaign-state.json first
- Do not treat Snowflake SHAs as authoritative — always verify against CI logs
- Do not assume `run_sd_unit_tests` for Python tests — match test filepath to correct workflow flag
