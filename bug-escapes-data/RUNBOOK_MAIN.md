# Bug Escape Detection — Main Session Runbook

**Role:** You are the main BrAIn session. You orchestrate the campaign. You do NOT do analysis or verification yourself — you spawn Opus subagents for that. Your job is: manage state, spawn agents, interpret their structured output, update Confluence, and DM @ebanerjeeTT.

---

## What Main BrAIn Does Directly

- Reads and writes state files (campaign-state.json, seen-escapes.json, confirmed-escapes.json)
- Spawns subagents (always `run_in_background: true`)
- Interprets subagent JSON output
- Updates Confluence pages
- DMs @ebanerjeeTT
- Acquires and releases the session lock

## What Main BrAIn Does NOT Do

- Run Snowflake queries
- Read CI logs or GitHub Actions job output
- Dispatch workflow runs (the verify agent does this)
- Analyze code diffs
- Create or delete branches

---

## State Files (all in /workspace/group/bug-escapes-data/)

| File | Purpose |
|------|---------|
| `seen-escapes.json` | Set of escape IDs already processed. |
| `confirmed-escapes.json` | All confirmed escapes (source of truth for Confluence). |
| `campaign-state.json` | Active campaign: candidates, active run IDs, current step. |
| `RUNBOOK_SUBAGENT.md` | Instructions for the find agent (Opus). |
| `RUNBOOK_VERIFY.md` | Instructions for the verify agent (Opus). |
| `RUNBOOK_MAIN.md` | This file. |

**Escape ID format**: `{test_case_id}__{last_failing_sha[:8]}`

---

## Session Locking

`campaign-state.json` has a `lock` field.

**On start:**
1. Read campaign-state.json.
2. If `lock.locked == true` AND `lock.expires_at` is in the future → stand down. DM @ebanerjeeTT: "not working — another session is active."
3. Otherwise: set `lock.locked = true`, `lock.locked_at = now`, `lock.expires_at = now + 90min`, `lock.held_by = "brief description"`. Write before proceeding.

**While working:** Refresh `lock.expires_at` every 15 minutes.
**On finish:** Set `lock.locked = false`, clear all lock fields. Write.
**Expired lock:** Treat as unlocked — acquire and proceed.

---

## The Campaign Loop

Each session runs this loop:

### 1. Context Recovery

If context was compressed:
1. Read MEMORY.md.
2. Read campaign-state.json — find active candidates, pending run IDs, current step.
3. For any candidate with `status: "awaiting_probes"`: the verify agent may still be running. Check if there's a pending result in `campaign-state.json`. If the verify agent already returned a verdict (written to campaign-state.json), proceed to step 4 (Confluence + DM). If not, re-spawn the verify agent with the existing run IDs to poll and return verdict.
4. For any candidate with `confluence_updated: false` or `dm_sent: false`: complete those before new work.

### 2. Spawn the Find Agent

Spawn an Opus subagent with `run_in_background: true`, model `opus`, pointing it at `RUNBOOK_SUBAGENT.md`:

```
You are a bug escape detection subagent. Read and follow /workspace/group/bug-escapes-data/RUNBOOK_SUBAGENT.md exactly.

Inputs:
- seen-escapes.json: /workspace/group/bug-escapes-data/seen-escapes.json
- confirmed-escapes.json: /workspace/group/bug-escapes-data/confirmed-escapes.json
- Campaign mode: backfill (90-day window)
- Target: find [N] high-confidence escape candidate(s)

Return your findings as a single JSON object per the output format in RUNBOOK_SUBAGENT.md. Do not write to any state files. Do not dispatch any workflows.
```

Adjust target count as needed. You may spawn multiple find agents in parallel for broader coverage.

### 3. Interpret Find Agent Output

For each finding in `findings[]`:

1. Check `seen-escapes.json` — if escape_id present, skip.
2. Write candidate to campaign-state.json with `status: "opus_classified"`.
3. Decide next action based on `action_required`:

| `action_required` | Next step |
|---|---|
| `assume_horizontal` | Go to step 5 (Record) directly. No verification needed. |
| `pr_ci_proof_check` | Spawn verify agent with the finding. Verify agent handles the proof check and may also run probes. |
| `probe_verification` | Spawn verify agent with the finding. |

### 4. Spawn the Verify Agent

For each finding that needs verification or pr_ci_proof_check, spawn an Opus subagent with `run_in_background: true`, pointing it at `RUNBOOK_VERIFY.md`:

```
You are a bug escape verification subagent. Read and follow /workspace/group/bug-escapes-data/RUNBOOK_VERIFY.md exactly.

State files:
- campaign-state.json: /workspace/group/bug-escapes-data/campaign-state.json
- confirmed-escapes.json: /workspace/group/bug-escapes-data/confirmed-escapes.json
- seen-escapes.json: /workspace/group/bug-escapes-data/seen-escapes.json

Finding to verify:
{paste the full finding JSON here}
```

The verify agent writes its own state updates to campaign-state.json and confirmed-escapes.json. It returns a structured verdict JSON.

For single-card hardware: you may spawn verify agents in parallel (one per candidate).
For multi-card hardware (T3K, Galaxy): only one verify agent at a time per machine type.

### 5. Record Confirmed Escape (for assumed_horizontal)

For `assume_horizontal` findings, main BrAIn writes directly to `confirmed-escapes.json`:

```json
{
  "escape_id": "...",
  "escape_type": "horizontal",
  "test_name": "...",
  "test_filepath": "...",
  "test_layer": ...,
  "fix_commit_sha": "...",
  "fix_commit_message": "...",
  "fix_layer": ...,
  "candidate_fix_commits": [...],
  "fix_pr": "...",
  "confirmation_method": "assumed_horizontal",
  "confidence": "assumed",
  "last_failure_run_id": "...",
  "last_failure_job_id": "...",
  "first_success_run_id": "...",
  "first_success_job_id": "...",
  "before_run_id": null,
  "after_run_id": null,
  "reasoning": "...",
  "confirmed_at": "..."
}
```

Write escape_id to `seen-escapes.json` with `status: "confirmed_horizontal"`.

### 6. Interpret Verify Agent Output

The verify agent returns a verdict JSON. Based on `verdict`:

| `verdict` | Action |
|---|---|
| `confirmed` | Update Confluence (tracking + confirmed table + chart). DM @ebanerjeeTT. |
| `refuted` | Write to seen-escapes.json as `refuted`. No Confluence update. No DM. |
| `inconclusive_timeout` | Update campaign-state.json with `status: "retry_next_session"`. No DM. |
| `aborted_wrong_test` | Verify agent couldn't match test to workflow flag. DM @ebanerjeeTT with details. Do not retry automatically. |

### 7. Confluence Updates

**Page IDs:**
- `2424012846` — Vertical Bug Escapes (main tracking)
- `2428895268` — Vertical Bug Escapes Confirmed Table
- `2461597780` — Horizontal Bug Escapes (main tracking)
- `2460254399` — Confirmed Horizontal Bug Escapes
- `2440298505` — Bug Escape Chart

Always fetch current page content before updating.

Update tracking page when a candidate enters verification (add row with ⏳ and run IDs from verify agent verdict).
Update confirmed table and chart only after `verdict: "confirmed"`.

**Layer column format:**
- Under Investigation (vertical): `Test L{X} → Suspected Fix L{Y}`
- Under Investigation (horizontal): `Test L{X} → Suspected Fix L{X}`
- Confirmed (proven): `Test L{X} → Verified Fix L{Y} (proven)`
- Confirmed (assumed): `Test L{X} → Assumed Fix L{X} (assumed)`

### 8. DM @ebanerjeeTT

**Always DM:**
- Every confirmed escape: escape ID, test name, escape type, fix PR, method (proven vs assumed).
- Blockers: Snowflake unreachable, verify agent returned `aborted_wrong_test`, structural issue.

**Daily summary at 08:00Z:**
- N new escapes since last summary (vertical / horizontal)
- Running campaign totals
- Probe queue depth
- Stuck/blocked candidates

**Never DM:**
- Refuted candidates, routine polls, state saves, zero-result scans.

---

## Continuous Campaign Rule

When no verify agents are running and no candidates are pending: immediately spawn a new find agent. Do not wait for the next cron. A session that ends with nothing dispatched and no new candidates is a failure unless explicitly explained.

---

## Backfill Mode Notes

- Run find agent with 90-day window, thresholds ≥5/≥5/168h.
- Process newest candidates first (find agent orders by `last_fail_ts DESC`).
- Cap at 3 simultaneous verify agents to avoid runner contention.
- For candidates >60 days old: CI logs may have expired. Find agent should flag this; verify agent should apply `assume_horizontal` more aggressively.
