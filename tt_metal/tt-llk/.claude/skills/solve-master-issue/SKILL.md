---
name: solve-master-issue
description: Work a master/umbrella GitHub issue (many sub-problems in one issue or as sub-issues) OR a local findings file (a bug-audit CONFIRMED.json / OPEN.md, or any list of file:line problems) to completion UNATTENDED — split it into items, group items that touch the same function/op into one PR (capped so the PR body still fits), find a fail-without/pass-with test per item, open a DRAFT PR per group, handle bot reviews, dispatch sanity + nightly CI only once bots are quiet and only within a CI budget (default 5 Sanity + 5 Nightly runs in flight) so other developers are not crowded off the runners, fix PR-caused failures, rebase past unrelated main breakage, until each PR is green or fails only in proven-unrelated tests AND every bot comment (on every push) is addressed. Writes progress back into the findings file (and commits/pushes or syncs it) so a re-run or another box never repeats work. Never idles; while anything waits on bots/CI/device/budget, it advances another item. Headless driver (run_unattended.sh) survives disconnects. Use when the user hands over an issue or a findings file with N problems and says to work through it without them.
argument-hint: <issue URL | owner/repo#N | /path/to/findings.json|.md> [--sync-cmd '<cmd>'] [--ci-budget sanity=5,nightly=5] [--group-max 8] [--only-open] [--plan-only] [--worker <name> --arch <arch>[,<arch>]] [--unattended --run-name <name>] [extra constraints]
user_invocable: true
---

# /solve-master-issue — work an umbrella issue to draft PRs, unattended

Target: **$ARGUMENTS**

The user is away; their hours away are the budget. The two failure modes this skill exists to prevent:
1. **Idling** — sitting on a foreground wait (bot review, CI, build) or cycling the same PR while other
   items could advance. An earlier unattended run spent ~90 min re-dispatching a crashing review bot
   while three items with buildable test paths sat parked.
2. **Asking** — stopping for input when a feasible path exists. "Needs the user" is a last resort,
   reached only after the feasible paths below are exhausted, and it never stops the other items.

**Done, per PR — all three, on the final head SHA:**
1. a test (or tests) that **fails without** the PR and **passes with** it, on shipped code;
2. a full Sanity + Nightly (+ LLK e2e) run **for that SHA**, every job **green** or proven
   **unrelated** with evidence;
3. **every bot comment addressed** — valid ones fixed, wrong ones pushed back on, each with a short
   reply — including comments that arrive after a CI-fix push or a rebase. `bot_threads.sh` exits 0
   and prints nothing.

Invoking this skill **is** the authorization to: create worktrees/branches, push your own branches
(force-with-lease on them only), create sub-issues, open **DRAFT** PRs, request **bot** reviews, reply
to/resolve **bot** threads, dispatch and cancel workflow runs you dispatched. It is **not**
authorization to: mark a PR ready for review, merge, touch anyone else's PR/branch/runs, reply to a
human thread, or file anything without a shipped-code repro.

**Two kinds of source.** A GitHub issue (URL or `owner/repo#N`), or a **local findings file**: a
path ending in `.json` or `.md` — the bug-audit skill's `CONFIRMED.json` / `OPEN.md`, or any list of
`file:line` problems with a summary. With a file there is no parent issue: items are grouped (§1b),
issues go to **tt-metal** (never tt-llk), and the file itself is the shared record of what has been
done (§0b) — it is marked after every state change and pushed, so a re-run, a crash, or another box
never repeats work. Options (anywhere in the arguments): `--sync-cmd '<cmd>'` (how to publish the file
when it is not in a git repo — e.g. a mirror's pack-and-push script), `--ci-budget sanity=N,nightly=M`
(default 5,5), `--group-max N` (default 8 items per PR), `--only-open` (work only items with no state
at all or `planned` by this skill — skip ones already filed upstream), `--unattended --run-name <name>` (set by
`run_unattended.sh`: end the turn instead of sleeping when everything waits; write `DONE` at the end),
`--plan-only` (triage + group every workable item, write the plan into the file, sync, stop — no fixes),
`--worker <name> --arch <arch>[,<arch>]` (this session is one of several workers sharing the file, each
on its own box; `--arch` = the architectures this box can run, named as the harness names them —
`wormhole`, `blackhole`, `quasar`, …: take groups by claim, serve the other workers' test legs — §0c).

Conventions used below: `<me>` = `gh api user -q .login`; `$SKILL` =
`tt_metal/tt-llk/.claude/skills/solve-master-issue` (scripts in `$SKILL/scripts/`) — if that path
does not exist in the checkout you are in (the skill's branch is not merged there), `$SKILL` is
`$(readlink -f ~/.claude/skills/solve-master-issue)`; `$MASTER_ISSUE_DIR` = where ledgers live,
default `~/.claude/master-issues` — point it at shared storage if you have some, so a run can be
resumed from another machine. Scripts avoid `gh run list --branch` and `gh run rerun --failed`, which
older `gh` (2.4) lacks.

| script | use |
|---|---|
| `mkwt.sh <name> <branch>` | worktree off origin/main (or the existing branch on resume) |
| `dispatch.sh [--match <regex>] "<workflow>" <ref> [-f k=v]` | dispatch a workflow **and get its run id** (gh prints none); `--match` pins the run's identity when the workflow+ref is shared |
| `ci_budget.sh [--need-nightly N] [--verify] [sanity=5] [nightly=5]` | room for one more PR's CI? counts **my** in-flight Sanity / Nightly(+e2e) dispatches on GitHub (so the cap is global across boxes); exit 0 = dispatch, 1 = queue it; `--verify` right after a dispatch: exit 1 = over the cap, cancel what you just dispatched (§6) |
| `ci_triage.sh <run_id>` | every non-green job of a run with its failing tests |
| `main_job_status.sh "<workflow>" "<job>"` | is the same job failing on main — on the same tests? |
| `bot_threads.sh <pr>` | bot feedback still unanswered; ack handled summaries by URL |
| `bots_pending.sh <pr>` | what must still finish before the head SHA is "bots quiet"; exit 0 + empty = quiet |
| `perf_ab.sh <fix-wt> <base-sha> <arch> <perf-module> <out-dir> [-k …] [--op …] [--goal …]` | same-board perf A/B of one tt-llk perf module, fix vs base, judged by `perf_eval.py` (§3b); exit 0 pass · 1 regression · 2 not comparable · 3 run failed |
| `perf_eval.py --current <csv> --baseline <csv> …` | the judge: variant-matched cycle deltas, ±0.5 % noise band, intent-aware verdict; vendored from the LLK issue solver, `test_perf_eval.py` guards it (run with `pytest -c /dev/null --noconftest`) |
| `findings.py list\|mark\|groups\|next\|claim\|release\|reconcile\|sync <file> …` | the findings file as source: workable items as JSON; write state + links per item (dups follow); the plan's groups; the next groups for a worker and the atomic claim (§0c); GitHub-vs-file drift on resume; commit+push (pull --rebase first) or `--sync-cmd` |
| `legs.sh request\|wait\|serve\|status --run-dir <dir> …` | a test leg on the *other* worker's board: the requester files a request in the shared run dir, that box's serve loop runs it under its device lock and writes the result back (§0c, §3) |
| `run_unattended.sh <run-name> [--worker <w>] <args…>` | detached headless driver: `claude -p` turns, resumed every 15 min until `DONE` (§2); one per box with `--worker`, state under `workers/<w>/` |

The two gate scripts print nothing when there is nothing left — and also when GitHub could not be
read, so their **exit status is part of the gate**: a non-zero exit is "not quiet / not answered",
retry after the poll interval. Never treat an empty failed read as done.

---

## 0. Setup (once)

1. **Verify the box** — notes from an earlier session may describe another machine. `hostname`; device
   (`tt-smi -s` / `ls /dev/tenstorrent`); arch; existing builds; sfpi; `gh auth status`; free disk; and,
   because `$HOME` is usually one shared filesystem for a whole cluster: `readlink -f ~/.claude/skills/solve-master-issue`
   and `readlink -f "$(command -v claude)"` must resolve **on this box** — a symlink into another box's
   local disk dangles here. Never relocate anything under a shared home onto local disk; use shared storage.
2. **Who else is on this device?** `ListAgents` — other Claude sessions on this box share the card.
   If one is active, message it: agree that device runs go through
   `flock /tmp/tt-device-$(hostname).lock <cmd>`. Hangs and corrupt results from two sessions on one
   card look exactly like the bug you are hunting.
3. **Read the whole issue** — body, every comment, and its native sub-issues
   (`gh api repos/<o>/<r>/issues/<N>/sub_issues`). No truncation. File source: §0b instead.
4. **Repo routing.** The PR goes to the repo that owns the code *now*: tt-llk is frozen, its code lives
   in `tt-metal/tt_metal/tt-llk`, so tt-llk issues get tt-metal PRs. **Every issue this skill creates
   goes to tenstorrent/tt-metal** — a tt-llk parent still gets its native sub-issue link (cross-repo
   sub-issues are allowed), but the child itself lives in tt-metal. Cross-repo `Fixes` needs the
   `owner/repo#N` form.
5. **Check for overlap** — `gh pr list --author <me> --state open` and open PRs touching the same
   files. An overlapping stale draft is a note in your PR, not a blocker.
   **Stay clear of the Copilot pipeline** (`squad-plan` → `issue-monster` → `copilot-flow-ready` →
   `pr-sous-chef`, all in `.github/workflows/`): an existing sub-issue labeled `copilot-ready`, assigned
   to Copilot, or with an open Copilot PR is *taken* — don't work it, note it in the ledger. Never put
   the `copilot-ready` label on a sub-issue you create, or Issue Monster will hand the same item to
   the Copilot coding agent while you are on it. That pipeline is for items Copilot can fix from a
   runner without hardware; this skill exists for the ones that need a card, a local build and a
   fail-without/pass-with proof.
6. **Create the ledger** at `$MASTER_ISSUE_DIR/<repo>-<N>/ledger.md` from `$SKILL/ledger-template.md`.
   It must let a *fresh* session continue with no conversation history: worktrees, branches, head SHAs,
   dispatched run ids, what each wait is for, retry counters. **After any context compaction, re-read
   the ledger before acting** — the summary loses details like run ids.
7. If persistent memory is available, write a resume note pointing at the ledger; update it at
   milestones (PR opened, CI verdict), not every step.

## 0b. A findings file as the source

The file is the record shared with the user and with any later run, so it is kept truthful at every
step; the ledger holds the operational detail (run ids, counters) that the file does not.

1. **Read it with the script, not by eye:** `findings.py list <file>` prints one JSON line per open
   canonical item — key (`path:line`, `#n` when two findings share a line), severity, class, summary,
   failure scenario, suggested fix, evidence, the arch-copies / same-defect duplicates that ride along.
   `--all` adds items that already carry a state. Items the audit's filed check marked `already_filed`
   (an open upstream issue exists) are worked too unless `--only-open` — link `Fixes <that issue>`
   instead of filing; skip any with an assignee or an open PR. `filed_closed` items: skip if the
   closing fix postdates the tree the finding was confirmed on, else treat as open and say so.
2. **Reconcile before touching anything** (every start and resume): `findings.py reconcile <file>`.
   It looks for your PRs and issues whose body carries `audit-key: <key>` — every issue and PR body this
   skill writes has one line per item — and writes any missing mark into the file, reports PRs closed
   without merge (a decision for the user, never silently reopened) and items left `in-progress`
   (resume from their worktree/branch if present, else restart). Cheap: two bulk queries, matched
   offline; only drift reaches you. Its purpose is the window the marks cannot close — a crash between
   `gh pr create` and the mark, or a push that failed.
3. **Mark on every state change, then sync:**
   ```bash
   $SKILL/scripts/findings.py mark <file> <key>... --state in-progress --branch <b> --worktree <wt> --group <g>
   $SKILL/scripts/findings.py mark <file> <key>... --state issue --issue <url>
   $SKILL/scripts/findings.py mark <file> <key>... --state pr --pr <url> --sha <sha>      # dups get it too
   $SKILL/scripts/findings.py mark <file> <key>... --state done|skipped|needs-user --note '<why>'
   $SKILL/scripts/findings.py sync <file> [--sync-cmd '<cmd>']      # commit+push, or the sync command
   ```
   `sync` commits and pushes the file when it lives in a git repo with an upstream (GitLab or GitHub
   alike), runs `--sync-cmd` when it does not (the bug-audit runs use a pack-and-push mirror script),
   and otherwise says the file is local-only (then say so in the report). Sync **immediately after
   each PR opens**, after each issue, and at least every **30 min** while items are in progress —
   `sync` takes a lock, so run it in the background and keep going. A crash loses at most the step in
   flight.
4. Key → location: strip any worktree prefix; the key is stable across checkouts. Two audit items
   that are one defect (`duplicate_of`, `also_at … (arch-copy|same-defect)`) are **one item** with all
   the sites listed; the mark propagates to the duplicates automatically.
5. **The file's line numbers are from the tree the audit ran on; you work on current `origin/main`**
   (`mkwt.sh` fetches it). Locate each finding by its `evidence` snippet, not the line. If main has
   already changed that code, re-check the claim there: still present → proceed (note the new line in
   the ledger); gone → `mark --state skipped --note 'fixed on main by <sha>'`, no issue, no PR.

## 0c. Several workers, one per box (optional)

One run can be worked by several sessions on several machines — one box per architecture, several
boxes per architecture, a box with two kinds of card, any architecture the harness names (`wormhole`,
`blackhole`, `quasar`, a future one) — **without splitting PRs and without exceeding the CI budget**.
Nothing below is specific to two boxes or to those two names. Prerequisites, all
verified in §0 step 1 on every box: `$MASTER_ISSUE_DIR` and the findings file on storage every box mounts
(the run dir holds the claims and the leg requests; `mkdir` on it is the arbiter); a tt-metal checkout
(built only if ttnn tests are expected — tt-llk pytest legs need just the harness, `sfpi` + `.venv`, ~2 GB; a Release build adds ~3 GB and 30–60 min and can be started after the worker); `gh auth status` under the
same login (the CI budget is counted per login, so it is shared automatically); this skill reachable on
that box (`~/.claude/skills/solve-master-issue` resolves there — a symlink into another box's local disk
does not). The findings file's own sync (git or `--sync-cmd`) runs from whichever box marks; with one
shared checkout that is just serialized; with per-box clones `sync` pulls with rebase first.

**The unit of ownership is the group, never the arch.** A defect with copies on several architectures
stays one PR (§1b). The owner of a group does everything for it — worktree, fixes for every arch copy,
issue, PR, reviews, CI — and borrows another box's card for the legs it cannot run (below). Nothing about
§4–§6 changes for a worker.

**Phase 1 — plan once** (`--plan-only`, run by one session, before any worker starts; cheap, reading only):
triage every workable item (§1), group them (§1b), then write the plan into the file and stop:
```bash
$SKILL/scripts/findings.py mark <file> <key>... --state planned --group <g> --arch <arch>[,<arch>]|neutral
$SKILL/scripts/findings.py sync <file> [--sync-cmd …]
```
`--arch` is what the fix touches — every architecture whose copy is in the group, comma-separated
(`list` prints an `arch_guess` from the `tt_llk_<arch>` trees the sites live in); `neutral` for host/ttnn
code. Items already `in-progress`/`pr` keep their owner and are not re-planned. Write the plan summary
(groups, arch counts) into `ledger-plan.md` in the run dir.

**Phase 2 — workers** (one driver per box, same `<run-name>`, different `--worker` names):
```bash
nohup setsid $SKILL/scripts/legs.sh serve --run-dir $MASTER_ISSUE_DIR/<run> --arch <this box's arch> --repo <built tt-metal checkout> > /dev/null 2>&1 &
nohup setsid $SKILL/scripts/run_unattended.sh <run> --worker <name> <file> --arch <this box's arch> [options] > /dev/null 2>&1 &
```
A worker's ledger is `$MASTER_ISSUE_DIR/<run>/workers/<name>/ledger.md`; the shared file is the only
cross-worker state. A worker **takes groups by claim**, never by list order:
```bash
$SKILL/scripts/findings.py next  <file> --owner <name> --arch <arch> --claims $MASTER_ISSUE_DIR/<run>/claims -n 3
$SKILL/scripts/findings.py claim <file> <group> --owner <name> --claims $MASTER_ISSUE_DIR/<run>/claims   # exit 1 = TAKEN → next
```
`next` orders the unclaimed planned groups: first those whose every architecture is on this box, then
neutral ones, then groups where only some legs can run here (the rest go out as remote legs) — and inside
each rank every worker walks a different, stable order (a hash of worker name and group), so n workers on
one architecture rarely attempt the same group; when they do, the `mkdir` claim decides and the loser
moves on. Groups needing none of this box's architectures are left to a box that has them
(`--take-other` overrides when it has nothing left). `claim` marks the group's items
`in-progress` with `--owner`; a claim you cannot start after all is `release`d. On resume, `reconcile`
lists `in-progress` items with their owner: **a worker resumes only its own; the rest belong to the other
box even if that box is silent** (it will resume them; if it is gone for good, the user reassigns by
`release`).

**Remote legs — another worker's box as the missing board.** When a group needs a fail-without /
pass-with / perf leg on an arch this box lacks, do not go to CI silicon first:
```bash
git push -u origin <branch>                                   # the serving box fetches it
id=$($SKILL/scripts/legs.sh request --run-dir $MASTER_ISSUE_DIR/<run> --arch blackhole --label item42-without \
      --branch <branch> --sha <sha> --cmd '.venv/bin/python -m pytest -q python_tests/test_x.py -k case' [--cwd tt_metal/tt-llk/tests] [--timeout 1800])
$SKILL/scripts/legs.sh wait --run-dir $MASTER_ISSUE_DIR/<run> $id        # exit = the leg's exit; prints the log path
```
Any box whose serve loop runs for that arch may take the request (several boxes per arch share one
queue). Branch mode runs in a detached worktree on the serving box (tt-llk pytest, gtest already built there);
`--patch <diff>` mode applies a patch to that box's **built** checkout under its device lock and reverts it
after — for ttnn tests that need a build. Probe arm = the same request on a `<me>/probe-<item>-shipped`
branch, exactly as in §3. A leg is a wait like any other: file it, record the id in the pending table, arm
nothing that must outlive the turn, and move on; `legs.sh status` shows the queue. The PR body names the
serving host and board and links the log in the run dir. **CI silicon (§3) remains the fallback** when no
worker serves that arch (`legs.sh status` shows the serve loop's last heartbeat) or the leg needs a SKU
neither box has.

**CI budget across boxes.** `ci_budget.sh` counts runs on GitHub per login, so every worker shares one
5 + 5 pool with no coordination, however many boxes there are. The gate is check-then-dispatch, so right
after every dispatch run `ci_budget.sh --verify`: exit 1 means another box dispatched in the same window and a lane is now over
its cap — cancel the run(s) you just dispatched (`gh run cancel <id>`) and put the PR back at the head of
your ci-queue. The cap holds exactly; a few seconds of overshoot is the worst case.

**Stopping.** `touch $MASTER_ISSUE_DIR/<run>/STOP` stops every worker after its current turn;
`workers/<name>/STOP` stops one. Each worker writes its own `workers/<name>/DONE` with its report (§8);
the run is done when every worker's is.

## 1. Split into items and triage

Turn the issue into numbered items — one per independently fixable problem (two symptoms of one
defect = one item; one paragraph naming two defects = two items). For each, record in the ledger:
file(s), claimed defect, arch(es), what hardware a test needs, and an initial class:

| class | meaning |
|---|---|
| `local` | testable on this box |
| `arch-other` | needs an arch/board not here — compile-only here + probe/fix A/B on the CI lane that has the board (§3) |
| `test-only` | the issue asks for coverage, not a code fix — deliverable is the test; evidence = it detects an injected regression |
| `assert-only` / `api-hardening` | no runtime behavior change; look for a test that *can* observe it before parking |
| `race/timing` | needs a forced repro (`perturb` skill, NOP injection) — but see the repro gate |
| `design` | genuinely a choice between defensible behaviors → park for the user, with the options written out |

**Order of work:** cheapest-to-prove first, so PRs enter the bot/CI pipeline early and their waits
overlap with the harder items. Two items that are the same defect share one PR and one sub-issue
(say so in the sub-issue).

## 1b. Group items into PRs (fewer PRs, fewer CI runs)

One PR per *group*, one sub-issue per group, one CI round per group. Grouping happens after triage,
before any fix, and is written into the ledger (`group` column) and the file (`--group`).

**Same group when** (any of): same function or op (the WH and BH copies of one defect, the SFPU kernel
and its ckernel wrapper, a kernel and the `llk_api` layer that calls it); same file within ~100 lines;
the audit's own `same-defect` / `arch-copy` links; one test file would prove all of them. **Never the
same group when:** different owners' areas (a `models/` kernel with an LLK header), one item is a
`design` or `needs-user` item (it would block the others), or the fixes would conflict textually.

**Cap, measured not guessed:** before the group is closed, render its PR body (template in §4) with
every item's problem, fix, test, and fail-without/pass-with result. It must fit GitHub's 65,536-char
body limit with **≥ 20 % headroom** (CI links, the tested-on section and review replies grow it), and
the sub-issue body likewise. Over → split along the weakest link (different file first, then
different function). Soft cap `--group-max` (default **8**) items per PR for reviewability: beyond it,
split even if the body fits. Items that match nothing go out alone, as today.

**Inside a group** each item still gets its own fail-without/pass-with proof, in the group's single
worktree, one commit per item (test + fix), so a reviewer can drop one without unpicking the rest. The
PR body lists items in a table; CI runs once for the whole group.

**Never stack PRs.** Every PR branches from `origin/main` and targets `main`; no PR may contain, depend
on, or wait for another open PR's commits. If a fix turns out to need code that is only in another open
PR of yours, the two belong to one group: fold the item into that PR (re-check the body cap) rather than
opening a second PR on top of it. Overlapping *files* between independent PRs are acceptable (the user
handles merge conflicts; the rebase before final CI resolves them); a merge-order *dependency* is not.

## 2. The scheduler — how to never idle

Each item moves through: `triage → repro → fix → verify → PR → bot-review → CI → done | parked`.

**Resources:**
- **Device** — serial, shared with other sessions. Every device run goes through
  `flock /tmp/tt-device-$(hostname).lock <cmd>`. Never hold it while reading code or building. **Cap the
  queue:** at most 4 device legs waiting on the lock at once (`pgrep -fc "tt-device-$(hostname)"`); beyond
  that, batch an item's legs into one run and hold the rest — tonight 18–20 queued legs meant a 30-min wait
  per leg and a box at load 200, which is slower than running them in turn.
- **Builds** — a Release build of tt-metal is ~3 GB and 30–60 min (a built worktree ~8 GB in all; the tt-llk harness, sfpi + .venv, another ~2 GB; tt-llk pytest needs no build); do **not** build per worktree. Keep one
  built worktree and A/B an item there by applying / reverse-applying its patch (`git apply`,
  `git apply -R`) — never `git stash` (the stash is shared across worktrees). tt-llk pytest items need
  no tt-metal build. At most 2 concurrent builds, always `run_in_background`.
- **GitHub API** — poll each run at most every 2 min (`gh run watch -i 120 --exit-status <id>`); many
  tight pollers trip secondary rate limits and then *everything* stalls.
- **CI budget** — shared with every other developer. At most **5 Sanity** and **5 Nightly(+LLK e2e)**
  runs of yours in flight at once (`--ci-budget`), counted on GitHub by `ci_budget.sh`, never from the
  ledger alone. A full budget blocks **only the dispatch step of that one PR**: the PR is opened, its
  reviews are requested and answered, it enters `ci-queue` in the ledger, and the skill goes straight
  on to the next item — triage, repro, fix, test, next PR. Queued PRs get their CI, in order and
  nearest-to-done first, as slots free up (§6). Probe/fix pairs, re-dispatches after a rebase, and
  rerun-failed-jobs calls all draw on the same budget.
- **Remote board** (workers, §0c) — another worker's card, through `legs.sh`; one request per leg,
  served in order under that box's device lock. Minutes, not the hour a CI probe pair costs.
- **Thinking/reading** — unlimited; parallelize with subagents.

**The loop**, after every action and every notification:
1. Anything finished? (bot review, CI run, build, background test, remote leg) → handle it now. A
   finished CI run also frees a budget slot: check `ci_budget.sh` and dispatch the head of the `ci-queue`.
2. Otherwise pick the next **unblocked** action across all items, preferring: device work if the
   device is free → items earliest in the pipeline → reading/analysis for parked items.
3. Before any wait, **arm it in the background** — a background Bash `gh run watch -i 120
   --exit-status <id>` (re-invokes you when it exits), or `Monitor` with an until-loop — then go
   straight to step 2.
4. Only when *every* item is genuinely waiting on something external may you stop, with a background
   `sleep 1800` heartbeat armed and the ledger's "pending events" table current. **Unattended**
   (`--unattended`): arm nothing that must outlive the turn — write the pending table and the file
   marks, sync, and **end the turn**; `run_unattended.sh` resumes the session every 15 min, and the
   first thing the resumed turn does is step 1 for everything that finished meanwhile. A turn's end
   kills its background shells, so anything local that must outlive it — a build, a long test sweep,
   a `findings.py sync` — is started with `nohup setsid … > <ledger dir>/<what>.log 2>&1 &`, its pid
   and log recorded in the pending table, and its log read on the next turn instead of re-running it.

**Surviving a disconnect.** Never run the skill in a foreground terminal you might lose. Launch it
detached — from the repo root, with `$MASTER_ISSUE_DIR` on shared storage:
```bash
nohup setsid $SKILL/scripts/run_unattended.sh <run-name> <source> [options] > /dev/null 2>&1 &
tail -f $MASTER_ISSUE_DIR/<run-name>/driver.log        # attempt-NN.log holds each turn's output
touch $MASTER_ISSUE_DIR/<run-name>/STOP                # stop after the current turn
```
Each attempt is one `claude -p` turn in auto mode that resumes the same session (so context carries);
if the session is lost, the next turn starts fresh and rebuilds state from the ledger and the file
marks (§0b reconcile). The skill writes `$MASTER_ISSUE_DIR/<run-name>/DONE` only with the final report.
**After the skill text changes, restart the run with a fresh session** (move `session.id` aside): a resumed
session keeps the old text in its context and follows it.

**Parallelism:** fan out independent items to subagents for triage, root-causing, and writing fix+test.
**You create the worktree** (`mkwt.sh`) and pass the path — don't use `isolation: "worktree"`, its
branch names and cleanup don't fit the PR flow. Give each agent: the item text, the worktree path, the
repro gate, the device-lock rule, "search only inside the worktree or a named path — never `find`/`bfs`
from `/` (such searches crawl the network mounts for days and load the box for everyone), and
"do not push, do not open PRs, do not touch GitHub — report diff, test, and fail/pass output." You (the
coordinator) own pushing, PRs, bot handling, CI, and the ledger.

**Retry caps** (one table; counts live in the ledger):

| what | cap | then |
|---|---|---|
| re-dispatch a review bot that infra-failed | 1 per SHA | note it; proceed as if that bot is quiet |
| rerun a flaky/infra CI job | 2 per cause | treat as unrelated with the log line as evidence |
| rebase + full re-dispatch because main was broken | 3 per PR | record the main breakage as the unrelated verdict |
| wait for main to go green on a job | 4 h | same — record with the main run URL, finish the PR |
| full CI cycles caused by code changes (review fixes / CI fixes) | 4 per PR | escalate (§6) |
| attempts at one item with no new evidence | 2 consecutive | find a new path (§3) or park with the reason |

Never repeat an action that produced no new information.

## 3. Repro → fix → verify (per item)

**The gate:** a test that **fails on shipped code and passes with the fix**, failing for the bug's
reason (not compile error, fixture, or timeout from something else). Without it: no sub-issue, no PR.
Synthetic mechanism-only kernels and injection-only failures are evidence, not a repro. For
`test-only` items the gate is: inject the regression the test is meant to catch, show it fails.

**Before parking an item as "can't repro", try every feasible path:**
- an existing full build on the box (a gtest target may already exist in another worktree);
- the other arch's code path compiles here — compile-only test here, then **CI silicon as the
  missing board** (below); a board you don't have is a slower gate, not a reason to park;
- data that actually hits the edge (denormals, bounds around 2^31, odd tile dims, max/min pads);
- a seed sweep / repeated runs for nondeterministic bugs; the `perturb` skill for timing;
- a test that observes the assert-only/API-only change at compile time (static_assert, a
  `requires`-style check), if the repo has that idiom.

**CI silicon as the missing board.** When the test needs a board this box lacks, the A/B runs on a
CI runner and the gate still holds — both arms, not just pass-with:
1. **Find the lane.** `grep -rl '<sku label>' .github/workflows .github/sku_config.yaml` for the
   board's runner label (`wh_n150`, `wh_n300`, `bh_p150b`, `wh_galaxy`, …). The hit is usually a
   reusable `*-impl.yaml` (or the SKU config), which is not dispatchable: follow `uses:` back to the
   caller that has `workflow_dispatch`, and read *its* inputs — the job you need is behind an opt-in
   flag, and a bare dispatch runs nothing. Known lanes: `Nightly tt-metal L2 tests` with
   `run_llk_unit_tests` (`unit_tests_llk` on every enabled SKU of each side — `run_wormhole` /
   `run_blackhole`, both default on, select the sides; the SKU list is computed by its arch-matrix job),
   `run_cpp_tests`; `LLK e2e Tests` (tt-llk pytest on wh_n150 + bh_p150b; its "failure" conclusion is
   a junit-report action crash unless a test job failed — read the job list).
2. **Two branches.** Fix branch = test + fix. **Probe branch** = the same test on shipped code
   (`<me>/probe-<item>-shipped`), pushed with no PR. Keep the split as `item<N>-test-only.patch` /
   `item<N>-fix-only.patch` in the ledger dir.
3. **Before spending the round**, compile the other arch's header in the harness here and check
   whatever is arch-neutral on the present board (golden layout, test plumbing), so the ~1 h CI round
   is not lost to a compile error or a wrong golden.
4. **Dispatch both arms at once** with `dispatch.sh`, record both run ids in the ledger's pending
   table, arm a watcher, and advance another item (§2). Fail-without = the probe run fails or hangs
   on the target SKU(s) for the bug's reason; pass-with = the fix run passes and the lane's full
   suite stays green.
5. **Evidence and cleanup.** The PR body names the SKUs, links both runs, and says no local board
   existed. Delete the probe branch once the fix run is recorded (`git push origin --delete`).
Another box with the board, if one is reachable, is the same thing with the device lock (§0) on that
host — but CI is the default because it leaves a link anyone can check.

**Where a board comes from, in order:** (1) this box (`tt-smi -s`); (2) **another worker of this run**
that has the board — `legs.sh request` / `wait` (§0c), minutes per leg and no CI budget; (3) another box
of the user's that already has the board — `ListAgents` shows their other sessions and the hostnames in
the user's notes; use it only if `ssh <host> true` works with the keys you have (never add keys or change
auth to get there), through that host's device lock; (4) **CI silicon**, above. This skill does not
reserve machines: if the user wants a reservation it is a line in the report, and the item proceeds
through a remote leg or CI silicon meanwhile — a missing board is never a reason to park.

**Fix:** `mkwt.sh <issue>-<item> <me>/<branch>`. Minimal fix derived from the codebase's correct
sibling usage. **Check the sibling arch** for the same defect — same line is not the same bug, but a
one-sided fix on a symmetric defect draws a valid bot comment; fix both only if both are proven,
otherwise note it in the PR. **Every architecture copy you change gets its own fail-without /
pass-with** on that architecture's silicon when a board for it is reachable — this box, a worker's box
through `legs.sh` (§0c), or CI silicon. "Compile-only" for a copy is acceptable only while no such board
exists, must be stated in the PR body as such, and is **backfilled before the PR is called done**: when a
worker for that architecture appears mid-run, every open PR of yours with a compile-only copy gets its
remote legs (probe branch for fail-without, fix branch for pass-with) and the body updated — it is
waiting in the CI queue anyway. Test goes in the most relevant existing test file; if new, wire it so CI
collects it — and check the **lane** that runs it is one the fixed file's path triggers.

**Verify, 3 ways, before any push:** fail-without + pass-with output captured to the ledger dir;
build succeeds; adversarial self-review of the diff. Run the neighboring tests too (no collateral
breakage). Then the perf A/B below, when the change is on a measured path.

### 3b. Perf A/B — a correctness fix must not get slower

**When:** the diff touches code a tt-llk perf module exercises — anything under
`tt_llk_*/common/inc/sfpu/`, `tt_llk_*/llk_lib/`, the ckernels `llk_api/llk_sfpu/` or `llk_math_*` /
`llk_unpack_*` / `llk_pack_*` wrappers, or `hw/inc/api/compute/` — and a perf module covers the op
(`grep -l '<MathOperation or op name>' tests/python_tests/perf_*.py`; the pack/unpack/tilize/matmul/reduce
modules are named for their op). Assert-only, host-side, test-only and doc changes need none; say so in
the ledger. A perf-motivated item uses `--goal improve`; everything else `--goal no_regress`.

**How** (the board must be in this box; `perf_ab.sh` creates and reuses a detached baseline worktree
at the PR's merge-base, runs current then baseline back to back under the device lock, and judges):
```bash
base=$(git -C <fix-wt> merge-base HEAD origin/main)
$SKILL/scripts/perf_ab.sh <fix-wt> $base <arch> <perf-module> $LEDGER_DIR/perf/<item>-<arch> \
    [-k '<pytest expr narrowing to the op>'] [--op <mathop substring>] [--primary-metric 'mean(MATH_ISOLATE)']
```
`-k` narrows what runs (a full module is 10–40 min); `--op` narrows what is compared. Modules that
declare only `MATH_ISOLATE` (`perf_sfpu_binop_scalar`, `perf_sfpu_reduce`, `perf_sfpu_ternary`) need
`--primary-metric 'mean(MATH_ISOLATE)'`; the evaluator never picks a kinder metric by itself. Run it once
per architecture you have; an architecture you lack is **"perf not measured on <arch>"** in the PR body,
never a blocker (CI silicon runs no perf).

**Verdict → action:** exit 0 (neutral or improved) → quote the summary line in the PR body and link
the two CSVs in the ledger dir. Exit 1 (regression past the noise band, median variant by default) →
a failure of class `PERF_REGRESSION`: fix it (counts as a code change: §5 bots, then §6 CI again), or,
if the slowdown is inherent to the correct behavior, keep the fix and say so in the PR with the numbers —
that is a decision the reviewer sees, not one you hide. Exit 2 (not comparable) → read
`verdict.json: reason_code`; `no_matching_baseline_variants` / `current_selection_empty` mean the
selector is wrong, fix the `-k` / `--op`; a schema mismatch means the module changed between base and
fix, measure a narrower selection. Exit 3 → the perf run itself failed; that is an environment or plan
problem (card, timeout, a variant that cannot compile), recorded as such, not a verdict on the fix.
Re-run the A/B after any later code change to the measured path (review fix, CI fix); a pure rebase
needs it only if main touched that path. Pre-commit hooks: if a hook rewrites and aborts, re-add and make a **fresh** commit, never
`--amend`. Never stage `tt_metal/third_party/*` submodule pointers or symlinked `sfpi`/`.venv`.

## 4. Open the DRAFT PR

In order:
1. **Sub-issue** for the item or group (skip if it already exists as one — then just `Fixes` it). Body
   states the bug on its own terms: what breaks, scenario, evidence, `file:line`; a group's issue has
   one section per item. **Issues are always created in tenstorrent/tt-metal**, whatever repo the
   source issue or the file came from — tt-llk is frozen (§0.4). With a parent issue: native child link
   (`gh api --method POST repos/<o>/<r>/issues/<PARENT>/sub_issues -F sub_issue_id=<child .id>` — the
   numeric `id`, with `-F`), assign `<me>`, verify the link took. With a file source there is no parent:
   a standalone issue, assign `<me>`, and one `audit-key: <key>` line per item at the end of the body
   (what `findings.py reconcile` matches on). Then `findings.py mark … --state issue --issue <url>`.
2. Push the branch; `gh pr create --draft`. Body: problem, fix, the test and its fail-without /
   pass-with result, what was and wasn't run on hardware (be exact — name the board), the perf A/B
   summary line per architecture or "perf not measured on <arch>" / "not on a measured path" (§3b),
   `Fixes <issue>`.
   A group's body: a table (item · file:line · defect · test · fail-without/pass-with), then one short
   section per item; `Fixes` each issue; the `audit-key:` lines at the end. Comment on the issue with
   the PR link. Then `findings.py mark … --state pr --pr <url> --sha <sha>` and `findings.py sync`.
   **Body edits right after a push get lost:** `.github/workflows/pr-description-inject-branch-name.yaml`
   re-renders the body on every `synchronize`, and a `gh pr edit` in its read/write window is overwritten.
   Edit the body only once `bots_pending.sh` shows no run of that workflow in flight, and re-read the body
   afterwards to confirm the edit took.
3. **Public-repo rule:** no RTL signal names, `.sv` paths, waveform decodes, or internal tracker ids
   in code comments, commits, PR bodies, or replies.
4. Ledger (and resume note): item → sub-issue, PR, branch, worktree, head SHA.

Then **immediately** dispatch reviews (§5) and go back to the scheduler.

## 5. Bot reviews

Known bots (GitHub account type `Bot`): `copilot-pull-request-reviewer`, `github-actions[bot]`
(LLK PR Review, skills reviewers), `cycode-security`. Humans are everyone else, whatever their
login looks like.

**On a draft PR no reviewer runs unasked.** The skills reviewers trigger only on `opened` /
`ready_for_review` and skip drafts outright; Copilot's automatic review fires on ready-for-review,
never on a push to a draft (checked: zero Copilot reviews on the draft PRs of an earlier run). So
"bots quiet" on a draft is only meaningful if you asked. **After every push that changes the tree,
request both:**
```bash
$SKILL/scripts/dispatch.sh --match "PR #<PR>\b" "LLK PR Review" main -f pr_number=<PR>   # ~45 min; record the run id
printf '{"reviewers":["copilot-pull-request-reviewer[bot]"]}' |
    gh api --method POST repos/<o>/<r>/pulls/<PR>/requested_reviewers --input -   # Copilot on the head SHA
```
(JSON body on purpose: `-f 'reviewers[]=…'` is a 422 on gh 2.4.) Within seconds Copilot shows as a
`Running Copilot Code Review` run on the head SHA (queued, then in progress) — it does not stay in
the requested-reviewers list — and the other checks that do fire on a push (static checks, PR gate,
Repo Assist, Silencer, …) are likewise **workflow runs attached to the head SHA**. So whether bots
are done is **observed, not guessed**:
```bash
pending=$($SKILL/scripts/bots_pending.sh <PR>) && [ -z "$pending" ]   # true = bots quiet on the head SHA
```
It lists: `RUN` (a run on the head SHA still queued/in progress — reviewers, static checks, PR gate;
any event, incl. `pull_request_target`), `CHECK` (a completed run on the head SHA that failed — an
automatic check that posts no comment, such as the static checks; fix it, or ack its URL with an
evidence-backed unrelated verdict in the ledger: the same check fails on `main` the same way —
distinct from the review-bot retry cap), `DISPATCH` (an LLK PR Review run for this PR, or a run id you
pass, not completed), `FAILED` (the latest LLK PR Review run for this PR did not succeed — re-dispatch
once, then ack its URL), `COPILOT` (review requested, not submitted), `SETTLE` (no run has registered
yet and the script first saw this head SHA under 10 min ago — GitHub needs a minute or two to queue
runs after a push; after 10 min with nothing registered, nothing is coming), and the `bot_threads.sh`
lines. The SETTLE clock starts when the script first sees the SHA, not at the commit date (commits
are made long before they are pushed), so **run it right after every push**.

**Bots quiet for a SHA** = `bots_pending.sh <PR>` exits 0 and prints nothing **and** both reviews
above were requested for that SHA (the ledger records the dispatch run id and the request). A reviewer
run that infra-failed counts as completed once it has been re-dispatched once (retry table). Pure
rebases (no code change) don't need fresh review requests, but `bots_pending.sh` must still be empty
before CI is spent on the new SHA — the checks that trigger on `synchronize` will have run again.
Waiting for the auto-triggered static checks / PR gate here is deliberate: they are already running,
and a failure there means another push anyway, so it costs nothing and saves a Sanity + Nightly run.

Poll it with the same discipline as any wait: a background loop every 2–5 min, not a tight loop.

**Every bot comment gets handled, on every push, for the life of the PR.** A CI fix, a review fix, or a
rebase is a new head SHA, you request the reviews again, and those comments are in scope too.
```bash
$SKILL/scripts/bot_threads.sh <PR>   # THREAD: unresolved threads a bot opened, latest comment from a bot; REVIEW: <url> <body-hash> summaries/comments
```
Read each in full (no truncation). Bots are not authorities — they are often wrong. For each:

- **Valid → fix it.** Verify the claim yourself first (build, run a test, spec/ISA docs, the sibling
  call site). **Push first**, then reply with what changed and the commit, then resolve.
- **Wrong → push back.** Reply with why, citing concrete evidence (`file:line`, spec section, test
  output, the sibling that does the same). Resolve. Don't change code to appease a bot.
- **Nit → push back, no code change.** A *nit* is anything that would not change behaviour or catch a
  bug: naming, comment wording, "consider extracting a helper", test-structure taste, "add a docstring",
  reordering, style the pre-commit hooks already accepted, "please clarify" on a point the body or code
  already states. Reply in one line that it is a style preference not taken (point at the idiom or the
  sibling it matches if there is one), **resolve, and do not push**. Applying nits is what creates the
  reply cycle: every push re-requests both reviewers, who return fresh nits on the new diff. The user's
  rule: *push back on unnecessary comments; never get into a cycle of replies.*
- **Partly valid** → fix the valid part, push back on the rest, in one reply.
- **Design choice / needs hardware not here** → no code change; leave the thread unresolved and put it
  in the report for the user with the options. Use sparingly — most comments are decidable.
- **No-findings summaries** ("No issues found") need no reply — ack it so it drops off the list:
  `echo '<url> <body-hash>' >> $MASTER_ISSUE_DIR/acked-<PR>.txt`, the pair exactly as the REVIEW line
  prints it. The hash is of the full body, so a bot that edits its comment in place (the perf gate
  does) comes back with a new hash and is handled again. Ack a findings summary only after every
  finding in it is handled.

Reply to a THREAD with `gh api repos/<o>/<r>/pulls/<PR>/comments/<root-rest-id>/replies -f body='…'`
(the thread's top-level comment id), resolve with the GraphQL `resolveReviewThread` mutation on the
thread id (both ids are in the `bot_threads.sh` line). Reply to a REVIEW with an issue comment on the PR.

**Replies: short, but keep the substance.** One to three sentences: the verdict, the evidence or the
change, the commit. No preamble, no thanking, no restating the comment. E.g.
- `Fixed in abc1234: LLK_ASSERT on num_faces before the MOP config; new case in test_foo covers 2-face.`
- `Not an issue: unpack waits on SRCA_VLD before this (llk_unpack_A.h:142), so the bank can't be in use.`
- `Same point as on the previous revision — see reply above; code unchanged since.`

**Before replying to a "description doesn't match the diff" comment**, read the *current* PR body —
bots quote the body as it was at review time.

**Human comments: never reply, resolve, or act on them.** One exception to "never act": a human asking
you to stop, close, or change direction on a PR **freezes that PR** — no more pushes or dispatches —
and goes to the top of the report. Otherwise, list human threads in the final report.

**Loop guards for reviews.** A bot repeating an already-refuted point on a new SHA: one line pointing
at the earlier answer, resolve, no code change. A bot asking to undo what it (or another bot) asked
for earlier: keep the verified version, push back, note it. Don't push to a PR while its LLK PR Review
run is in flight unless you will re-dispatch — batch the fixes instead. These guards limit
*re-dispatching and churn*, never *answering*: every comment that does arrive is handled.

**Convergence cap — a review cycle ends, it is not re-entered for nits.** Count review rounds per PR
in the ledger (`bot-review(rN)`), and keep a second counter: rounds since the last push that fixed a
*verified correctness finding* (wrong result, hang, missing guard, wrong encoding, a test that did not
test what it claims). Rules:
- From round 3 on, a push is justified **only** by a verified correctness finding. Everything else in
  that round is answered as a nit (above) or a push-back, resolved, and the PR is **bots-quiet for
  dispatch** the moment `bots_pending.sh` is empty — do not re-request reviews for a round that
  changed no code.
- Two consecutive rounds with no correctness finding → the PR is converged. Ack the summaries, move it
  to `ci-queue`/dispatch (§6), and spend no more review rounds on it; later pushes (CI fixes, rebases)
  re-request reviews as usual, and the counters restart from that push.
- A round that would be round 6+ on one PR is a red flag for the ledger log: write what each round
  changed, and whether any of it was behaviour. If none was, the earlier rounds were nits that should
  have been declined.
- Review rounds are external waits. While they run, the scheduler (§2) advances other items; a session
  whose only activity is review rework while items sit in `triage` is idling.

## 6. CI

**CI is spent only on a SHA the bots are already quiet on (§5) — never on a push you expect bots
to comment on.** The review loop converges first; CI comes after. The target is **one full CI run
per PR**; every additional run must be justified in the ledger by exactly one of: a code change made
*after* bots were quiet (review had nothing left, CI exposed a real defect), or a rebase past main
breakage. A bot finding that arrives while CI is already running is handled, but the fix is held
(batched) until that CI run reports, so one re-run covers both.

**Budget gate, right before every dispatch** (first run, re-run after a fix or rebase, probe/fix
pair, rerun-failed-jobs alike):
```bash
$SKILL/scripts/ci_budget.sh [--need-nightly 2]   # 2 when LLK e2e is needed too; exit 0 = go, 1 = queue
```
Exit 1 → the PR's status becomes `ci-queue(<since>)` in the ledger and **nothing else about it
waits**: reviews stay answered, and the scheduler moves to other items. Exit 0 → dispatch, then
`ci_budget.sh --verify` at once: exit 1 means a lane is over its cap after all (only another worker of
yours on another box can cause it, §0c) — cancel what you just dispatched and re-queue. The queue drains in order,
PRs nearest to done first (a re-run after a one-line CI fix before a first run), one PR per freed
slot, whenever a run of yours completes — check the budget in step 1 of the loop, not on a timer. Both
lanes of a PR are dispatched together or not at all, so a PR never holds a Sanity slot while waiting
for a Nightly one. The point is pacing, not throughput: other developers share these runners.

Once bots are quiet on the current head SHA **and the budget has room**, dispatch on the branch and
record run ids. A manual dispatch is **not** the nightly by default: every test family the cron schedule turns on is an
opt-in `workflow_dispatch` input that defaults to false, so a bare dispatch reports success with the
test jobs skipped. "Full Nightly" for this skill is the set of scheduled suites that execute kernels:
```bash
$SKILL/scripts/dispatch.sh "Sanity tests"              <branch> -f run-llk-sanity-tests=true
$SKILL/scripts/dispatch.sh "Nightly tt-metal L2 tests" <branch> \
    -f run_llk_unit_tests=true -f run_cpp_tests=true -f run_p100_sanity_tests=true \
    -f run_triage_tests=false \
    -f additional_test_categories=<union of the category lists the `schedule` arms pass>
$SKILL/scripts/dispatch.sh "LLK e2e Tests"             <branch>     # if tt-llk files changed
```
Build the category list at dispatch time from `.github/workflows/tt-metal-l2-nightly.yaml`: every
`github.event_name == 'schedule' && '<list>'` expression feeding an `additional_test_categories`
(the ops unit-test list, `compute_fused`, `ops_docs_check` — there are several, take the union). The
scheduled suites **not** dispatched, and why: DIDT (power stress, not correctness), tutorials and
tt-cnn (ttnn front-end flows already covered by the ops categories), TT-Train unit + perf and
tt-triage (own stacks; perf jobs are not pass/fail evidence). All of those default off except
`run_triage_tests`, which defaults **on** — hence the explicit `=false`. If the PR touches code one
of those owns, turn its flag on (`run_didt_tests`, `run_tutorials_tests`, `run_tt_cnn_unit_tests`,
`run_tt_train_*`, `run_triage_tests`) and say so in the ledger. Then **check the run's job list**,
not its conclusion: a green rollup whose test jobs are all skipped is not a pass. ("PR - Sanity tests" is not dispatchable. On
drafts the PR-Gate llk lanes are skipped; silicon coverage for tt-llk changes comes from Sanity's
`run-llk-sanity-tests` smoke on WH+BH, Nightly's `run_llk_unit_tests`, and LLK e2e, the whole tt-llk
suite.) Nightly takes hours — arm the watchers and move on.

**Triage every failing job** (`ci_triage.sh <run_id>`) by reading the job log, not the rollup. Its
first line tallies success / skipped / failed jobs: a run whose test jobs are all skipped was
dispatched without its opt-in inputs — re-dispatch, don't record it.

| verdict | evidence needed | action |
|---|---|---|
| **caused by PR** | failure touches changed code/test, or reproduces locally with the PR and not without | fix (with a test if the failure exposed a gap), push → §5 for the new SHA, then full re-dispatch |
| **unrelated, main also broken** | same job fails on `main` in the same window **on the same failing tests / error line** — `main_job_status.sh "<workflow>" "<job substring>"` prints each failed job's FAIL lines, compare them with `ci_triage.sh`'s for the PR run; a main failure on different tests is "can't tell", and "no matching job" is *not* evidence | arm a watcher on main's runs of that workflow; when main's job goes green (cap 4 h), `git fetch origin main && git rebase origin/main` (the worktree's `origin/main` is only as new as its last fetch), `push --force-with-lease`, wait for bots quiet on the new SHA (§5), *then* full re-dispatch |
| **flaky / infra** | runner setup, rate limit, `infra:timeout`, card off bus, perf a hair past band; main passes | `gh api --method POST repos/<o>/<r>/actions/runs/<id>/rerun-failed-jobs` (max 2) |
| **can't tell** | | rerun once; if still ambiguous, run the failing test locally with/without the PR if the box can |

Never call a PR green from a partial rollup — enumerate every job of every dispatched workflow.

**After a rebase** (always `git fetch origin main` first — watching main through the API does not
move the local `origin/main`), re-run the item's test with the fix (must still pass). If main touched
any of the PR's files (`git diff <old-base>..origin/main --stat -- <files>`), redo the fail-without
check too — main may have fixed or moved the defect.

**The final SHA must have its own full Sanity + Nightly (+ LLK e2e) pass.** Results from an older SHA
don't count. Any push that changes the tree — review fix, CI fix, rebase — means a full re-dispatch
on the new head. Only PR-body edits and replies (no new commit) need none.

**Convergence loop, per PR:**
```
push → bots review new SHA → address all (§5) ─┬─ code changed? → push → (back to top)
                                               └─ bots quiet   → NOW dispatch full CI on this SHA
CI done → triage ─┬─ PR-caused failure → fix → push → (back to top: bots first, CI after)
                  ├─ unrelated, main broken → wait main green (≤4 h) → rebase → push → (back to top)
                  └─ green / proven unrelated → DONE
```
Keep it converging:
- **Batch.** Collect every bot finding and every PR-caused CI failure known at the time into one push,
  not one push per comment.
- **Don't spend CI on a SHA you know will change.** Dispatch full CI once the bots are quiet on that
  SHA. If a push lands while CI is running on an older SHA, cancel the stale runs **you dispatched**
  (`gh run cancel <id>`) and re-dispatch on the new head.
- **Push back on non-essential bot nits** (style preferences, speculative hardening, rewording) once
  the PR is past its first review round — they're the usual cause of extra iterations. Fix only what
  is actually wrong or materially unclear.
- **Escalate instead of spinning:** after **4 code-change CI cycles** on one PR, stop iterating it,
  record why it hasn't converged (which bot or which test keeps reopening it), and put it in the
  report for the user. The other items keep going.

**Done for a PR** only when, on the same final head SHA: a full Sanity + Nightly (+ LLK e2e) run exists
**for that SHA**, its test jobs are present in the job list (not skipped for want of an opt-in input),
and every job is green or has a recorded unrelated verdict with evidence (main run URL with the same
failing tests, or log line showing infra); `bot_threads.sh <PR>` exits 0 and prints nothing (or what's
left is in the report as a decision for the user); and the fail-without/pass-with test still holds.
Record it, update the PR body's "tested on" section if it changed, and leave the PR as **draft**.
File source: `findings.py mark … --state done --pr <url> --sha <sha>` and sync.

## 7. Revisit parked items

Whenever the scheduler finds nothing else unblocked, re-read each parked item with fresh eyes against
§3's list. Things learned on other items (a build now exists, a test idiom found, a CI lane that has
the other arch, a probe-branch A/B already proven on one item) often unblock them. An item parked
only because its board is not on this box is not parked — it is waiting on a CI round (§3).

## 8. Final report (and the end state of the ledger)

One table: item · group · class · status · sub-issue · PR · fail-without/pass-with evidence · perf A/B
(verdict + worst delta per arch, or not measured / not applicable) · CI verdict (green / unrelated
failures with links) · bot comments: fixed / pushed back / left for the user (counts) · time spent in
`ci-queue`.
Then, separately:
- PRs frozen by a human request, with the quote;
- items parked for the user — each with the exact decision needed and the options;
- human review threads awaiting them, quoted briefly;
- anything run on only one arch/board;
- file source: counts by state in the file, whether the last `sync` pushed (or is local-only), and
  the reconcile findings (PRs closed without merge, drift repaired).

Write it to the ledger, update the resume note, run a final `findings.py sync`, and — unattended —
`touch $MASTER_ISSUE_DIR/<run-name>/DONE` **only now**: not while any PR is in `ci-queue`, `ci`, or
`bot-review`, and not while any item is `in-progress`. A worker (§0c) writes
`workers/<name>/DONE` instead, and "in-progress" means *its own* items: the other worker's open items
and claims are listed in the report, not waited for. `--plan-only` ends with the plan summary in
`ledger-plan.md` and the synced file — no DONE, no worktrees, no PRs.
