---
name: tt-multiagent
description: Coordinator procedure for running several bring-up agents in parallel in separate git worktrees on separate hosts - splitting tracks, creating branches, assigning hosts, monitoring cadence, hold rules, status reports, handoff and merge protocol, kill vs resume, recovery after a session restart. Use when you are the coordinator, or when you are an agent writing a handoff or deliverable.
---

# Multi-agent bring-up (coordinator and agents)

Last validated against commit: <none yet>
Evidence: `bringup/GUIDE_dsv4_lessons.md`. Related: `tt-moe-bringup` (tracks and gates), `tt-lab-ops` (hosts), `tt-bringup-log` (evidence).

## 1. Splitting work
1. Take tracks and gates from `tt-moe-bringup`. Parallel only where files and hosts do not collide: reference, attention, MoE first; assembly after; perf only on a merged correct baseline.
2. One brief per agent (file under `bringup/<model>/briefs/<track>.md`): goal, pass criterion and merge gate, branch, worktree, host(s), time box, files it may touch, files it must not touch,
   what to read first, how to log, final report format. Shared rules go in one common brief file all agents read.
3. Branches/worktrees (create in background; NFS is slow): `git -C <main> worktree add /mnt/tt-data/ssinghal/wt/<WT> -b ssinghal/<model>-<track> ssinghal/<model>`.
4. Do not start more agents than idle hosts + 1. Six agents on four hosts and one lock meant measurement agents queued behind long runs for hours. Cheap CPU-only tracks (reference, dumps, static analysis) are fine as extras.
5. Each agent needs an early cheap deliverable (a 4-layer or unit-level result) in < 3 h; long runs only after it.

## 2. Hosts
Assign hosts in the brief (`tt-lab-ops` bad-host list applies). Hold rule when hosts are scarce: priority = (1) merge-gate runs of the critical path, (2) correctness
reproductions, (3) perf A/B, (4) exploratory. Agents on lower priority wait (poll every 20 min, give up after 3 h and report). Never preempt by killing: ask the owner.

## 3. Monitoring cadence
Every 20-30 min while jobs run: `status_report.py` (intended; else per host `pgrep`, last line and age of each log, flock holder). Flag: log silent > the phase threshold,
slot-wait vs hang (check for "waiting for slot"), duplicate processes, agents idle without result, a host with unexpected holders. Report to the human at most every few hours or on events.

Status report format (short): per agent - track, branch@commit, host, job (log path, age, phase), last result with LOG id, blocker; then merge queue, hosts table (idle/busy/bad), open decisions for the human, risks.

## 4. Agent deliverable (handoff)
Final report: worktree, branch, commit ids, files changed, evidence (LOG ids, log paths, exact repro commands), what did not work, what is unverified,
which other tracks it composes with or conflicts with. Write a `handoff` LOG entry. Commit everything (hooks are slow over NFS: commit in background with a long timeout and retries, message ends with
the `Co-Authored-By` line; do not push).

## 5. Merge protocol (coordinator, on the integration worktree, never on a tree a job runs from)
1. Gate check: LOG has FRESH evidence on the candidate commit for the track's gate. No evidence, no merge.
2. `git merge-tree $(git merge-base HEAD <branch>) HEAD <branch>` (or `git merge --no-commit --no-ff`) for conflicts; resolve by hand, keep both sides' intent.
3. `git diff --stat HEAD...<branch>` and `git diff --diff-filter=D --name-status HEAD...<branch>`: look for UNEXPECTED DELETIONS and rewrites (an overlay copy of an old tree deletes newer work). Reject or fix.
4. Verify cross-feature wiring after merge: spec decode / paged pools / traces still hooked (grep the call sites; run the 4-layer smoke); defaults unchanged unless the brief says so; env gates read at the right time.
5. Re-run the cheap gate on the MERGED commit on an idle host (the merged result is a new commit: old numbers are stale). Log it. Then update the integration branch.
6. Branches stay (never delete); do not push unless the human says so.

## 6. Kill vs resume
Resume (SendMessage with the agent id) when the agent has context and its job survived or is cheap to relaunch. Kill/stop only your own agents, and only after their processes are accounted for (log, pid, host).
Start a fresh agent (with the human's OK if policy requires) when resume is refused or the context is poisoned by a retracted claim: give it the branch, the LOG, the brief and the retraction list.
Stray watchers/sleep loops of stopped agents must be cleaned up by the coordinator (own ones only).

## 7. After a session restart
Agents are stopped; branches, worktrees, logs, LOG.md and detached jobs survive. Procedure: (1) `git worktree list`, `git status` per worktree (uncommitted work), (2) process list per host, (3) read each LOG tail,
(4) resume agents with SendMessage; tell each: "check which of your jobs died, relaunch, report" and give the current host table, (5) if resume is refused, spawn a replacement with the brief + LOG.

## 8. Correct claims discipline (coordinator)
Do not relay a number to agents unless it is FRESH in LOG. Mark inferred statements as such in briefs. A claim reversed after agents acted on it cost days (batch-is-free, tile padding, 40% gaps, bfp4, "prefill works").
When you retract: LOG `retraction` entry, tell affected agents explicitly.

## Escalate to the human
Scope or priority changes, precision/accuracy trade-offs, destructive actions (reset, delete, force-push, kill foreign jobs), installs, C++ rebuilds in the shared build, merging with failing gates, when hosts stay scarce > 3 h.
Stop and report when: the critical path is blocked and no one can unblock, or a gate fails twice with the same cause.
