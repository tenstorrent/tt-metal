# TT Auto Triage Port + Progressive Production Rollout Plan

## Goal

Move the current CI triage system from this large branch into `tt-auto-triage` in reviewable vertical slices, then deploy safely to production (`tenstorrent/tt-metal` + real Slack channels) with controlled blast radius.

This plan assumes:

- We cannot merge this full branch directly into `tt-metal`.
- We need manager-approved bounds before real side effects.
- We should optimize for small PRs, clear ownership, and reversible rollout.

---

## Deployment Principles

- Keep each PR vertically scoped (code + tests + docs for one capability).
- Default all new production paths to **read-only or dry-run** first.
- Gate all write side effects behind explicit flags and hard caps.
- Require measurable acceptance criteria before enabling next slice.
- Preserve kill-switches at every stage.

---

## Pre-Work (Manager + Policy Alignment)

Complete before opening production-side PRs:

1. Confirm allowed side effects by environment:
   - Issue create/update/close permissions.
   - Draft PR create/update permissions.
   - Workflow dispatch permissions.
   - Slack channels and mention policy.
2. Confirm production safety guardrails:
   - Max new issues/run.
   - Max Slack posts/run and per-thread.
   - Max PR actions/run.
3. Confirm on-call/ownership:
   - Who monitors first canaries.
   - Who can disable the system quickly.
4. Confirm final target architecture:
   - `tt-auto-triage` owns orchestration logic.
   - `tt-metal` remains the source system for CI signals.

---

## Target Repo Structure in `tt-auto-triage`

Port to a minimal, explicit layout:

- `workflows/` (or `.github/workflows/`): triage orchestration workflow(s)
- `actions/`: composite actions (`auto-disable-stale-ci`, report generation, etc.)
- `tools/ci/`: Python runtime and helper scripts
- `docs/`: runbooks, policy, rollout notes
- `tests/`: unit tests for parsing, state, and decision contracts

Avoid carrying over unrelated `tt-metal` files; move only required automation code.

---

## Vertical Slice Plan (PR-by-PR)

### Slice 0: Bootstrap + Safety Skeleton

Scope:

- Create base repo scaffolding in `tt-auto-triage`.
- Add `guarded_gh.py` policy wrapper and tests.
- Add workflow skeleton with `workflow_dispatch`, run-scope, and kill-switch input.

Acceptance:

- CI passes.
- Guard wrapper blocks disallowed commands.
- Workflow can run in no-op/read-only mode.

### Slice 1: Shared State + Artifact Restore

Scope:

- Port `ci_triage_state.json` schema management.
- Port `load_previous_triage_state.py`.
- Add artifact upload/download/restore steps to all stateful jobs.

Acceptance:

- Repeated runs preserve and reuse state.
- Missing artifact path fails closed to safe defaults (no duplicate spam).

### Slice 2: Slack Ingestion + Enrichment (Read-only)

Scope:

- Port Slack channel/thread export scripts.
- Port issue-status enrichment against target issue repo.
- Produce artifacts only; no issue/PR/slack side effects beyond exports.

Acceptance:

- Deterministic JSON artifacts generated.
- Enrichment success/failure behavior visible in summary.

### Slice 3: Agent-First Slack Analysis Layer

Scope:

- Port `slack_thread_agent_analysis.py`.
- Replace deterministic Slack interpretation in all active paths.
- Add strict output contract checks and marker parsing.

Acceptance:

- All Slack interpretation paths call agent analyzer.
- Contract parse failures degrade safely (no write action).

### Slice 4: Draft PR Stage (Fork/Test-only Side Effects)

Scope:

- Port stale selector + disable planning + execute pipeline.
- Keep target repo/channel in test scope initially.
- Preserve dedupe markers (`Auto-disable-source-ts`) and PR update path.

Acceptance:

- New PR create works under caps.
- Existing PR update creates a new commit and changelog comment.
- No duplicate PRs for same source ts.

### Slice 5: Issue Creation Stage (M4)

Scope:

- Port deterministic failure extraction + issue/slack bootstrap flow.
- Preserve job/fingerprint dedupe markers (`Auto-triage-job-key`, fingerprint).

Acceptance:

- No duplicate issues for same failing job identity.
- Slack bootstrap messages tie to created issue deterministically.

### Slice 6: Thread Lifecycle Stage (M5)

Scope:

- Port follow-up warning cadence and owner-claim assignment.
- Keep unassigned-only assignment rule.
- Preserve no-noise fallback from Slack thread history.

Acceptance:

- No repeated warning/final/post-disable messages when historical markers exist.
- Assignment failures post remediation guidance only once per gate window.

### Slice 7: Issue Lifecycle Stage

Scope:

- Port issue close/update/unchanged review with main-repo evidence.
- Preserve processed-hours gating and fallback dedupe using recent lifecycle comments.

Acceptance:

- Re-runs inside processed window do not reprocess same issue.
- Mixed medium-confidence defaults to update/watching behavior.

### Slice 8: Testing Harness + Mock Sessions

Scope:

- Port testing sessions (owner resolution, persona simulation, E2E thread test).
- Keep explicit testing-mode boundaries and low token defaults.

Acceptance:

- Test mode can run independently.
- Reports are generated and uploaded every run.

### Slice 9: Unified Summary + Operational Runbook

Scope:

- Ensure always-on run summary job.
- Add operator docs: escalation, kill-switch, rollback.

Acceptance:

- Every run has a human-readable summary.
- On-call can disable side effects in under 5 minutes.

---

## Progressive Production Rollout

## Phase A: Read-only Production Canary

- Real data sources (`tenstorrent/tt-metal`, production Slack read channels).
- No issue/PR/slack write side effects.
- Validate classification quality + candidate volume.

Exit criteria:

- Stable for 3-5 runs.
- No critical parsing/contract failures.

## Phase B: Limited Write Canary

- Enable only one write path at a time with caps:
  - First issue lifecycle comments/updates only.
  - Then thread follow-ups.
  - Then draft PR creation/update.
- Set hard limits to 1 action/run per path.

Exit criteria:

- No duplicate/noisy posts.
- Team confirms signal quality and acceptable cadence.

## Phase C: Controlled Expansion

- Raise limits gradually (1 -> 3 -> 5).
- Enable full run-scope path for scheduled runs.
- Keep daily review of summaries and action logs.

Exit criteria:

- Two weeks with no trust-breaking noise events.
- Error budget and rollback drills pass.

---

## PR Sizing and Review Strategy

- Keep PRs to ~300-600 LOC where possible.
- Include tests/docs in same PR for each slice.
- Use feature flags so code can merge before activation.
- Require one “operator review” and one “code review” for side-effect paths.

Recommended branch naming in `tt-auto-triage`:

- `slice/00-bootstrap-safety`
- `slice/01-state-restore`
- `slice/02-slack-ingestion`
- ...
- `slice/09-summary-runbook`

---

## Production Safeguards to Keep Enabled

- `guarded_gh.py` as the only github execution path.
- Run-scope input for isolated testing.
- Hard caps for actions per run.
- Cross-run state restore in every stateful job.
- Fallback dedupe from Slack thread and issue comment history.
- Always-on unified summary.

---

## Rollback Plan

If noise or bad actions occur:

1. Disable workflow dispatch/schedule immediately.
2. Rotate or revoke write tokens.
3. Set all write-path flags to false/no-op.
4. Keep read-only data collection enabled for diagnosis.
5. Post incident note in ops channel with affected actions and mitigation.

---

## Day-1 Execution Checklist (Tomorrow)

- Manager policy bounds documented.
- Create `tt-auto-triage` repo and baseline CI.
- Land Slice 0 + Slice 1.
- Run one read-only canary from `tt-auto-triage`.
- Review summary with manager before enabling any production writes.
