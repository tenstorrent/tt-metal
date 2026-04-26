# CI Triage Autonomy Roadmap

## Context And Starting Point

- Last pushed commit baseline: `35e66b7b72656a06eaeb9bec0f8cfd0dc5aae813`
- Current branch has new automation scaffolding but has not been validated end-to-end.
- Existing workflow path in `.github/workflows/triage-ci.yaml` already supports:
  - Slack export and issue-closure enrichment
  - Data gathering mode report generation
  - Auto-disable mode entrypoint + artifacts

This document lays out an implementation and testing plan to safely reach the end state:

1. Fully autonomous repeated triage runs
2. No duplicate disable PRs
3. Resume/update prior disable PRs when new failures appear
4. Terminal completion notification
5. Post-facto bug-escape tracking tied to fixing commits
6. Full ticket + Slack lifecycle automation with human approval boundary only at PR review/merge

---

## End-State Goals

### A) Autonomous CI Maintenance Loop

When triage runs repeatedly:

- It should ingest deterministic failures from aggregate workflow data.
- It should create/update/close tracking issues automatically in test repo pre-promotion, and create a new issue on recurrence.
- It should post/update Slack lifecycle messages (and thread replies) without human intervention.
- It should not create duplicate disable PRs for the same failure fingerprint.
- It should continue working on an existing disable PR branch if checks fail due to additional failures.
- It should re-kickoff targeted workflows after each incremental disable/fix action.
- It should stop only at configured terminal states and emit one terminal notification.

### B) Persistent Cross-Run Memory

Each run should read prior run state and write updated state:

- attempted actions
- PR links and branch names
- workflow run outcomes
- terminal status (`completed`, `needs_followup`, `paused`, etc.)
- owner follow-up/escalation metadata
- issue lifecycle metadata (open/updated/closed + recurrence-linked new issue)
- Slack lifecycle metadata (anchor message/thread update keys)

### C) Human Approval Boundary

Human involvement should be limited to review/merge boundaries:

- automation drafts PRs and runs validations
- automation keeps iterating while confidence below threshold
- automation requests review only when PR readiness threshold is met
- humans approve/merge or request changes; all other loop steps remain autonomous

### D) Bug-Escape Attribution Store

After resolution, capture:

- which commit/PR fixed the issue
- affected layer and fixing layer
- whether this looks like:
  - missing gate coverage (pr-gate/merge-gate shift-left gap)
  - lower-layer regression escaping into higher-layer tests (example: llk/metalium change causing ttnn/models failure)

This tracking is analytical/post-facto and should not block disable automation.

---

## Architecture Plan

## 0) Trigger + Ingestion Layer

Cycle starts from schedule/event triggers:

- cron tick
- workflow completion event
- optional Slack/issue comment events

Data ingestion pulls:

- aggregate workflow failure clusters (`extract_failing_jobs.py`-style fingerprints)
- latest Slack channel/thread context exports
- current issue/PR state snapshots

Hard rule: all downstream actions must reference deterministic failure fingerprint + evidence links.

## 1) State Model (Single Source Of Truth)

Create `build_ci/triage_state/ci_triage_state.json` artifact schema:

```json
{
  "version": 1,
  "updated_at_utc": "2026-03-26T00:00:00Z",
  "items": [
    {
      "key": "fingerprint:<deterministic-id>",
      "slack_ts": "1773931010.433769",
      "failure_fingerprint": "workflow/job/test/signal hash",
      "issue_numbers": [40111],
      "status": "pr_open",
      "ticket": {
        "repo": "ebanerjeeTT/issue_dump",
        "number": 0,
        "url": ""
      },
      "slack": {
        "channel_id": "C0APK6215B5",
        "anchor_ts": "",
        "last_thread_update_ts": ""
      },
      "disable_pr": {
        "number": 0,
        "url": "",
        "branch": "",
        "head_sha": ""
      },
      "attempts": 1,
      "last_kickoff_runs": [],
      "owner_state": "unassigned|assigned|responding|unresponsive",
      "terminal_reason": "",
      "history": []
    }
  ]
}
```

Status enum:

- `new`
- `planned`
- `pr_open`
- `kickoff_running`
- `kickoff_failed_new_failure`
- `completed`
- `needs_human`
- `paused`

Why this matters: state should drive decisions, not ad hoc scraping.

---

## 2) Controller Loop Behavior Per Run

For each candidate/fingerprint:

1. Load/reconcile state entry (or create new)
2. Ensure ticket lifecycle state is current (create/update/close, and open a new issue on recurrence)
3. Ensure Slack lifecycle state is current (anchor + threaded updates)
4. Run action selector:
   - small fix path
   - disable path (SLA breach / blocker)
   - stale validation refresh path
   - observe-only path
5. If `pr_open`:
   - inspect latest workflow outcomes for that PR branch
   - if passed: mark `completed`, send terminal notify
   - if failed with new target: update same branch + commit + rerun kickoff
6. If no PR exists and action is disable/fix:
   - create draft PR
   - kickoff targeted workflows
   - set status to `kickoff_running`
7. If readiness threshold met:
   - request human review/approval
8. After merge/decision:
   - post-decision cleanup (ticket/slack updates, re-enable follow-up, reassignment/escalation)

Apply hard limits:

- max edits per item per run
- max cumulative attempts before `needs_human`

---

## 3) Artifact Strategy

Use two artifacts per run:

- `triage-state`:
  - `ci_triage_state.json`
- `triage-actions`:
  - planned actions
  - execution results
  - summary markdown
  - action-selector trace (why fix vs disable vs refresh)
  - ticket/slack lifecycle mutations

At start of each run:

- download the most recent successful `triage-state` artifact from this workflow
- if absent, bootstrap new state

At end of each run:

- upload new state artifact
- append compact changelog into run summary

---

## 4) Duplicate Prevention Rules

Use multiple keys to avoid repeated PR creation:

- primary idempotency key: failure fingerprint
- secondary linkage: issue number set + channel id
- PR body marker: `Auto-disable-source-ts: <ts>`

Before creating a new PR:

- check state entry status
- check open PRs with marker
- check closed PR history linked to same key

If any active PR exists, resume that PR instead of opening another.

---

## 5) Ticket Lifecycle Automation

For each stable failure fingerprint:

1. create issue if none exists
2. update issue with fresh evidence and state transitions every triage cycle
3. close when objective pass window met (or obsolete with rationale)
4. create a new linked issue on recurrence beyond hysteresis threshold
5. enforce label policy: `CI auto triage`

Pre-promotion routing:

- issue create/update/close only in `ebanerjeeTT/issue_dump`; recurrence creates a new issue in the same repo

## 6) Slack Lifecycle Automation

For each active fingerprint lifecycle:

1. create one anchor lifecycle message
2. post state transitions in thread
3. include issue/PR/workflow links and owner info
4. suppress spam with keying (`fingerprint + phase + day`)
5. send terminal-only summary notification on completion/escalation

Thread replies become state input:

- owner acknowledged
- owner requested defer/changes
- disputed signal / false positive hint
- escalation required

## 7) Resume-And-Extend Existing PR Branch

When workflows fail after an existing disable change:

1. checkout existing PR branch
2. analyze latest failing runs (same protocol as disable command)
3. apply minimal incremental disable
4. commit with additive message (`ci: extend disable scope for #<issue>`)
5. update PR body section:
   - latest attempt number
   - what new failing target was disabled
6. rerun kickoff-workflows command
7. persist state and run URLs

This is the core capability needed for "keep running until CI is stabilized."

---

## 8) Action Selector (Fix vs Disable vs Refresh vs Observe)

Action selector inputs:

- failure persistence + blast radius
- owner responsiveness and SLA age
- fix confidence and scope estimate
- previous attempts / disable debt

Paths:

- Path A: Auto Solve Ticket (small high-confidence fix)
- Path B: Auto Disable After SLA breach
- Path C: Refresh stale validation and rerun targeted checks
- Observe-only when signal is too noisy/ambiguous

## 9) Slack Notification Policy

Channel target (for now): `C09EC0QNSB0`

Send Slack messages only for terminal events:

- completed success
- escalated to `needs_human`

Message fields:

- issue number(s)
- disable PR URL
- latest kickoff run URLs
- status and brief rationale

Do not spam on intermediate loop iterations unless a run transitions state.

---

## 10) Bug Escape Tracking (Post-Facto Analytics)

This should run as a separate, non-blocking stage after completion signals are observed.

### Data to capture

For each resolved item:

- `source_issue_numbers`
- `source_slack_ts`
- `disable_pr` (if any)
- `fix_pr` and `fix_commit_sha` (if a true fix later replaces disable)
- `files_changed` and inferred component/layer
- `failing_test_layer` (llk/metalium/ttnn/models)
- `fix_layer` (same taxonomy)
- `escape_type`

### Escape taxonomy

- `gate_coverage_gap`:
  - signal suggests pr-gate/merge-gate should have caught this
- `layer_escape_lower_to_higher`:
  - fix touches lower layer; failing symptom appears in higher layer tests
- `unknown`

### Minimal first-pass heuristic

- infer layer from file paths:
  - `tt_metal/third_party/tt_llk/**` -> `llk`
  - `tt_metal/**` -> `metalium`
  - `ttnn/**` -> `ttnn`
  - `models/**` -> `models`
- infer failure layer from disabled test/workflow target path
- classify `layer_escape_lower_to_higher` when fix layer is lower than failure layer

Store this in:

- `build_ci/triage_state/bug_escape_events.json`

Add periodic rollup:

- counts by escape type
- top repeated components
- unresolved attribution rate

---

## Implementation Phases

## Phase 0: Stabilize Current Baseline

- validate current workflow dry-run path
- confirm artifacts and summary are produced reliably
- verify no duplicate markdown sections

Exit criteria:

- two consecutive dry runs complete with deterministic outputs

## Phase 1: State Persistence

- implement state artifact download/load/write
- reconcile state with live GitHub PR reality each run

Exit criteria:

- state survives across runs and recovers if artifact missing

## Phase 2: Resume Existing PRs

- implement "update existing PR branch" path
- add PR body update section for attempt history

Exit criteria:

- simulated failed rerun updates same PR (no new PR)

## Phase 3: Terminal Completion + Slack Notify

- detect success terminal condition
- send one Slack completion message
- mark state `completed`

Exit criteria:

- rerun after completion does not trigger new disable actions

## Phase 4: Failure Ingestion + Fingerprint Correlation + New-Issue Slack Bootstrap

- ingest aggregate workflow failures
- correlate with existing ticket/PR/Slack state
- create new issue for unmatched high-confidence fingerprints
- send initial Slack lifecycle notification/thread anchor for each newly created issue
- fail closed on low-confidence signals

Exit criteria:

- deterministic fingerprint generation and replay stability
- exactly-once issue-create + initial Slack notification per new fingerprint

## Phase 5: Ticket Lifecycle Automation

- update/close issues automatically in test repo and create a new issue on recurrence
- enforce `CI auto triage` label policy
- continue threaded Slack lifecycle updates for existing incidents

Exit criteria:

- full ticket lifecycle runs on controlled scenarios without manual edits

## Phase 6: Slack Lifecycle + Reply-Aware Control

- post anchor + threaded lifecycle updates automatically
- consume thread replies as control signals
- if owner confirms imminent fix, switch to defer/observe path and suppress disable PR creation for that cycle

Exit criteria:

- state transitions adapt correctly when humans reply in thread

## Phase 7: Action Selector + SLA Paths

- implement fix/disable/refresh/observe selector
- enforce approval boundary and confidence thresholds
- explicitly validate fix-vs-disable-vs-defer transitions from Slack thread input

Exit criteria:

- controlled fixtures cover Path A/B/C decisions deterministically

## Phase 8: Bug Escape Tracking

- capture fix commit mappings and escape classification
- produce artifact + summary table

Exit criteria:

- can answer: "what fix commit resolved which issue, and was it an escape?"

---

## Detailed Test Plan

## A) Unit Tests (Python scripts)

Cover:

- stale candidate filtering edge cases
- state merge/reconciliation logic
- duplicate detection with marker parsing
- layer classification from path sets
- escape type classification

Add golden test fixtures:

- representative Slack JSON
- representative workflow outcomes
- representative PR metadata

## B) Integration Tests (Local/CI dry-run)

1. No prior state:
   - creates initial state correctly
2. Prior state with open PR:
   - does not create second PR
3. Prior state completed:
   - skips item
4. Prior state kickoff failed with new target:
   - produces branch update plan

Use `auto-disable-dry-run=true` to validate decisioning without writes.

## C) Controlled Live Tests (Small Scope)

Run on one known stale issue:

1. live PR creation and kickoff
2. rerun with same input: verify no duplicate PR
3. force scenario with additional failing target: verify same PR is extended
4. success scenario: verify completion mark + Slack notification
5. thread reply scenario: verify owner response changes selector behavior
6. ticket lifecycle scenario: auto create + update + close + recurrence-new-issue

## D) Regression Tests

Ensure `data-gathering-mode=true` path remains unchanged:

- report generation still works
- summary output still single-write
- artifacts unchanged

---

## Operational Safeguards

- concurrency lock on workflow (`concurrency` group in GitHub Actions)
- max actions per run
- max attempts per item before escalation
- allowlist of workflow names that can be dispatched
- explicit audit log in state history

---

## Suggested Next Work Items (In Order)

1. Add aggregate failure fingerprint ingestion + correlation
2. Add ticket lifecycle create/update/close controller plus recurrence-new-issue policy
3. Add Slack lifecycle sender + thread-reply parser
4. Extend action selector (fix/disable/refresh/observe)
5. Keep M0-M3 invariants while integrating expanded loop
6. Add approval-boundary readiness checks before review requests
7. Add bug-escape event capture + rollup artifact
8. Add synthetic fixture suite and controlled live validations

---

## Definition Of Done

System is done when all are true:

- repeated triage runs do not create duplicate disable PRs
- existing disable PRs are iteratively extended when new failures emerge
- successful stabilization marks terminal completion and emits one Slack notification
- aggregate failure ingestion continuously feeds new deterministic fingerprints
- issues are auto-created/updated/closed with correct labels and evidence, and recurrence creates a new linked issue
- Slack lifecycle is autonomous and thread replies influence decisions
- action selector can choose fix vs disable vs refresh vs observe with auditable rationale
- human involvement is limited to PR review/merge boundary
- bug escape records map issue -> fixing commit/PR with classification
- dashboards/artifacts can show "what escaped and where to shift left"
