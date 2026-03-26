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

---

## End-State Goals

### A) Autonomous Disable Lifecycle

When triage runs repeatedly:

- It should not create duplicate disable PRs for the same Slack/issue signal.
- It should continue working on an existing disable PR branch if checks fail due to additional failures.
- It should re-kickoff pruned workflows after each incremental disable.
- It should stop when success criteria are met and record terminal completion.

### B) Persistent Cross-Run Memory

Each run should read prior run state and write updated state:

- attempted actions
- PR links and branch names
- workflow run outcomes
- terminal status (`completed`, `needs_followup`, `paused`, etc.)

### C) Bug-Escape Attribution Store

After resolution, capture:

- which commit/PR fixed the issue
- affected layer and fixing layer
- whether this looks like:
  - missing gate coverage (pr-gate/merge-gate shift-left gap)
  - lower-layer regression escaping into higher-layer tests (example: llk/metalium change causing ttnn/models failure)

This tracking is analytical/post-facto and should not block disable automation.

---

## Architecture Plan

## 1) State Model (Single Source Of Truth)

Create `build_ci/triage_state/ci_triage_state.json` artifact schema:

```json
{
  "version": 1,
  "updated_at_utc": "2026-03-26T00:00:00Z",
  "items": [
    {
      "key": "slack_ts:1773931010.433769",
      "slack_ts": "1773931010.433769",
      "issue_numbers": [40111],
      "status": "pr_open",
      "disable_pr": {
        "number": 0,
        "url": "",
        "branch": "",
        "head_sha": ""
      },
      "attempts": 1,
      "last_kickoff_runs": [],
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

For each stale unresolved candidate:

1. Load state entry (or create new)
2. If `completed`/`paused`, skip
3. If `pr_open`:
   - inspect latest workflow outcomes for that PR branch
   - if passed: mark `completed`, notify Slack
   - if failed with new target: update same branch + force-push/commit + rerun kickoff
4. If no PR exists:
   - create disable PR
   - kickoff pruned workflows
   - set status to `kickoff_running`

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

At start of each run:

- download the most recent successful `triage-state` artifact from this workflow
- if absent, bootstrap new state

At end of each run:

- upload new state artifact
- append compact changelog into run summary

---

## 4) Duplicate Prevention Rules

Use multiple keys to avoid repeated PR creation:

- primary idempotency key: Slack `ts`
- secondary linkage: issue number set + channel id
- PR body marker: `Auto-disable-source-ts: <ts>`

Before creating a new PR:

- check state entry status
- check open PRs with marker
- check closed PR history linked to same key

If any active PR exists, resume that PR instead of opening another.

---

## 5) Resume-And-Extend Existing PR Branch

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

## 6) Slack Notification Policy

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

## 7) Bug Escape Tracking (Post-Facto Analytics)

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

## Phase 4: Bug Escape Tracking

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

1. Add state artifact load/save helpers
2. Add state reconciliation against open PRs
3. Implement resume-existing-PR execution branch
4. Add terminal success detector + Slack notifier
5. Add bug-escape event capture + rollup artifact
6. Add integration tests and one controlled live validation run

---

## Definition Of Done

System is done when all are true:

- repeated triage runs do not create duplicate disable PRs
- existing disable PRs are iteratively extended when new failures emerge
- successful stabilization marks terminal completion and emits one Slack notification
- bug escape records map issue -> fixing commit/PR with classification
- dashboards/artifacts can show "what escaped and where to shift left"
