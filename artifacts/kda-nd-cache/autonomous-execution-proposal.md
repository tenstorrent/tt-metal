# Proposal: Keep Autonomous Engineering Work Moving

## Verdict

An autonomous task should enter implementation quickly and continue along the
critical path without treating ordinary setup, validation failures, or status
messages as stopping points. The agent should stop only when it lacks required
authority, faces an ambiguous destructive action, or has exhausted credible
technical paths.

## What went wrong in this run

The branch, specifications, tracker, source survey, build, Python environment,
and hardware reservation were established, but no implementation concern had
landed when progress was requested. That is poor throughput: useful preparation
was allowed to consume the whole opening phase.

Three behaviors caused the delay:

1. Setup work was serialized even where a long build or hardware run could have
   overlapped with source inspection and implementation planning.
2. Environment friction received open-ended attention instead of a time-boxed
   primary path followed by a known fallback.
3. Progress was measured by activity completed rather than by user-visible
   outcomes: implemented concerns, passing tests, commits, and pushes.

## Proposed operating contract

### 1. Kickoff has a deadline

Within the first 15 minutes, or the first three tool round trips when operations
are slow, the agent must:

- fetch and create the requested branch;
- write the minimum sufficient design and development spec;
- create the work items and dependencies; and
- begin the first implementation concern.

Specs are written and revised as evidence develops. They are not approval gates
when the user has explicitly authorized autonomous execution.

### 2. Maintain an executable critical path

At all times, identify one current deliverable and its immediate validation:

`edit -> focused test -> commit -> push -> next concern`

Repository surveys and broad builds must answer a specific question on that
path. If they do not change the next edit or validate it, defer them.

### 3. Time-box recoverable friction

- First failure: diagnose from exact output.
- Second attempt: use the repository-documented fallback.
- Repeated failure: record the blocker, choose another safe path, and continue
  independent work.

Do not repeatedly ask the user about commands already inside the requested
scope. Platform-required privilege approvals are still requested through the
approval mechanism, preferably batched by a narrow reusable command prefix.

### 4. Overlap waiting with productive work

While a build, test, or benchmark owns a long-running session, continue with
read-only inspection, test design, report scaffolding, or the next independent
code change. Poll often enough to preserve logs and respond to failures.

### 5. Commit and push at concern boundaries

Each bead has an explicit observable outcome and focused validation. As soon as
that outcome passes:

1. record exact validation evidence;
2. perform the closure reflection;
3. commit only that concern;
4. push the requested upstream branch; and
5. start the next ready bead immediately.

This limits lost work and makes asynchronous review useful before the whole task
finishes.

### 6. Use outcome-based progress reporting

Every update should state:

- completed and pushed concerns;
- current executable action;
- latest validation result;
- next concern; and
- any true blocker requiring user attention.

Setup is supporting evidence, not implementation progress. A task with specs and
a green build but no product change should be reported as “implementation not
started,” not as substantially underway.

### 7. Apply an autonomy stop rule

Continue without asking when the next action is reversible, in scope, and has a
reasonable validation path. Stop only for:

- missing authorization for an externally consequential action;
- unclear ownership of destructive or irreversible data changes;
- a design ambiguity that would materially change the requested behavior and
  cannot be resolved from repository evidence; or
- an exhausted technical blocker documented with attempted alternatives.

## Success measures

For future tasks of this shape:

- first implementation edit begins within the kickoff deadline;
- no long-running command leaves the critical path idle;
- every completed concern is committed and pushed before the next concern;
- status reports distinguish preparation from delivered behavior; and
- after an unexpected failure, the next recovery or independent action begins
  in the same turn.

## Immediate application to the KDA task

The current queue is fixed: capture baseline, implement recurrent direct-ND
state, implement convolution direct-ND state, run accuracy and comparative
performance, then publish the HTML report. Each passing concern will be pushed
immediately to `origin/momcilo/kda-nd-cache`; ordinary test failures will be
diagnosed and acted on without waiting for further approval.
