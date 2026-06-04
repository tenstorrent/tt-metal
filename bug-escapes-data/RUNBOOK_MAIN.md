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

**Status glossary:**
- `sql_found` — found in Snowflake, not yet analyzed
- `confirmed` / `confirmed_horizontal` — verified escape, Confluence updated
- `refuted` — all candidate fix commits tried, none confirmed; transition likely caused by infra/firmware outside the visible range
- `refuted_wrong_fix` — at least one probe returned BEFORE=PASS; fix hypothesis wrong but untried candidate commits remain; eligible for retry at low priority only
- `expired` — `last_fail_ts` older than 90 days; CI logs gone, probes not possible; skip permanently
- `skipped_*` — noise, unrelated, or infra failure; skip permanently

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

## Modes

### Normal Campaign Mode (default)
Run the full campaign loop. After each verify agent completes, immediately spawn a new find agent. Never idle.

### Incremental Mode
Activated when the user says "use incremental mode" or similar, or when set in `campaign-state.json` as `"mode": "incremental"`.

**First: write `incremental` to the `mode` field in `campaign-state.json` immediately.**

Same behavior as Normal Campaign Mode except the find agent uses a 2-day Snowflake lookback window instead of 90 days. Use this for routine daily runs to catch recent regressions without re-scanning the full history. The Continuous Campaign Rule still applies.

### Just Find One Mode
Activated when the user says "use the just find one protocol" or similar.

**First: write `just_find_one` to the `mode` field in `campaign-state.json` immediately.** This ensures the mode survives context compression — on recovery, you will read this field and know to stay in just_find_one mode.

**Behavior:**
1. Write mode to campaign-state.json (see above).
2. Spawn the find agent with `target: 1` and `mode: just_find_one`.
3. Take the first qualifying finding returned — ignore the rest.
4. Spawn one verify agent for that finding. Wait for its verdict.
5. Complete Confluence update and DM if applicable.
6. **Stop. Reset `mode` back to `backfill` in campaign-state.json. Do not spawn another find agent.**

The Continuous Campaign Rule does NOT apply in Just Find One mode.

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

Before spawning, read `campaign-state.json` to get the current `mode` field. Then spawn:

```
You are a bug escape detection subagent. Read and follow /workspace/group/bug-escapes-data/RUNBOOK_SUBAGENT.md exactly.

Inputs:
- seen-escapes.json: /workspace/group/bug-escapes-data/seen-escapes.json
- confirmed-escapes.json: /workspace/group/bug-escapes-data/confirmed-escapes.json
- Campaign mode: {mode from campaign-state.json — backfill / incremental / just_find_one}
- Target: find {1 if just_find_one, otherwise N} high-confidence escape candidate(s)

Return your findings as a single JSON object per the output format in RUNBOOK_SUBAGENT.md. Do not write to any state files. Do not dispatch any workflows.
```

In `just_find_one` mode: target is always 1, and the subagent stops after the first qualifying candidate. You may spawn multiple find agents in parallel for broader coverage (except in just_find_one mode — spawn only one).

### 3. Interpret Find Agent Output

For each finding in `findings[]`:

1. Check `seen-escapes.json` — if escape_id present, skip.
2. Write candidate to campaign-state.json with `status: "opus_classified"`.
3. Decide next action based on `action_required`:

| `action_required` | Next step |
|---|---|
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

### 5. Interpret Verify Agent Output

The verify agent returns a verdict JSON. Based on `verdict`:

| `verdict` | Action |
|---|---|
| `confirmed` | Update Confluence (tracking + confirmed table + chart). DM @ebanerjeeTT. |
| `refuted` | All candidate commits exhausted. Write to seen-escapes.json as `refuted`. No Confluence update. No DM. |
| `refuted_wrong_fix` | Fix hypothesis was wrong but untried candidate commits remain. Update campaign-state.json with `status: "refuted_wrong_fix"`. **Only re-spawn the verify agent with the next candidate commit if no new unprocessed candidates exist.** New escapes take priority. No DM. No Confluence update. |
| `inconclusive_timeout` | Update campaign-state.json with `status: "retry_next_session"`. No DM. |
| `aborted_wrong_test` | Verify agent couldn't match test to workflow flag. DM @ebanerjeeTT with details. Do not retry automatically. |

### 6. Confluence Updates

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

### 7. DM @ebanerjeeTT

**Always DM:**
- Every confirmed escape: escape ID, test name, escape type, fix PR, method (proven vs assumed).
- Blockers: Snowflake unreachable, verify agent returned `aborted_wrong_test`, structural issue.
- Results from a finished subagent 

**Daily summary at 08:00Z:**
- N new escapes since last summary (vertical / horizontal)
- Running campaign totals
- Probe queue depth
- Stuck/blocked candidates

**Never DM:**
- Routine polls, state saves, zero-result scans.

---

## Continuous Campaign Rule

When no verify agents are running and no candidates are pending: immediately spawn a new find agent. Do not wait for the next cron. A session that ends with nothing dispatched and no new candidates is a failure unless explicitly explained.

**Exception:** This rule does not apply in Just Find One mode. In that mode, the session ends after one escape is fully processed.

---

## Backfill Mode Notes

- Run find agent with 90-day window, thresholds ≥5/≥5/168h.
- Process newest candidates first (find agent orders by `last_fail_ts DESC`).
- Cap at 3 simultaneous verify agents to avoid runner contention.
- For candidates >60 days old: CI logs may have expired. Find agent should flag this; verify agent should note expiry in the verdict notes field.
