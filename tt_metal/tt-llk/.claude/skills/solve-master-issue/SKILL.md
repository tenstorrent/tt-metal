---
name: solve-master-issue
description: Work a master/umbrella GitHub issue (many sub-problems in one issue or as sub-issues) to completion UNATTENDED — split it into items, find a fail-without/pass-with test per item, open a DRAFT PR per fixed item, handle bot reviews, dispatch sanity + nightly CI, fix PR-caused failures, rebase past unrelated main breakage, until each PR is green or fails only in proven-unrelated tests AND every bot comment (on every push, including CI-fix pushes) is addressed — fixed if valid, pushed back on if wrong, with a short reply. Never idles; while anything waits on bots/CI/device, it advances another item. Use when the user hands over an issue with N problems and says to work through it without them.
argument-hint: <issue URL or owner/repo#N> [extra constraints]
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
   reply — including comments that arrive after a CI-fix push or a rebase. `bot_threads.sh` prints
   nothing.

Invoking this skill **is** the authorization to: create worktrees/branches, push your own branches
(force-with-lease on them only), create sub-issues, open **DRAFT** PRs, reply to/resolve **bot** threads,
dispatch and cancel workflow runs you dispatched. It is **not** authorization to: mark a PR ready for
review, merge, touch anyone else's PR/branch/runs, reply to a human thread, or file anything without a
shipped-code repro.

Conventions used below: `<me>` = `gh api user -q .login`; `$SKILL` =
`tt_metal/tt-llk/.claude/skills/solve-master-issue` (scripts in `$SKILL/scripts/`); `$MASTER_ISSUE_DIR`
= where ledgers live, default `~/.claude/master-issues` — point it at shared storage if you have some,
so a run can be resumed from another machine. Scripts avoid `gh run list --branch` and
`gh run rerun --failed`, which older `gh` (2.4) lacks.

| script | use |
|---|---|
| `mkwt.sh <name> <branch>` | worktree off origin/main (or the existing branch on resume) |
| `dispatch.sh "<workflow>" <ref> [-f k=v]` | dispatch a workflow **and get its run id** (gh prints none) |
| `ci_triage.sh <run_id>` | every non-green job of a run with its failing tests |
| `main_job_status.sh "<workflow>" "<job>"` | is the same job failing on main? |
| `bot_threads.sh <pr>` | bot feedback still unanswered; ack handled summaries by URL |

---

## 0. Setup (once)

1. **Verify the box** — notes from an earlier session may describe another machine. `hostname`; device
   (`tt-smi -s` / `ls /dev/tenstorrent`); arch; existing builds; sfpi; `gh auth status`; free disk.
2. **Who else is on this device?** `ListAgents` — other Claude sessions on this box share the card.
   If one is active, message it: agree that device runs go through
   `flock /tmp/tt-device-$(hostname).lock <cmd>`. Hangs and corrupt results from two sessions on one
   card look exactly like the bug you are hunting.
3. **Read the whole issue** — body, every comment, and its native sub-issues
   (`gh api repos/<o>/<r>/issues/<N>/sub_issues`). No truncation.
4. **Repo routing.** The PR goes to the repo that owns the code *now*: tt-llk is frozen, its code lives
   in `tt-metal/tt_metal/tt-llk`, so tt-llk issues get tt-metal PRs. Sub-issues go under the parent, in
   the parent's repo. Cross-repo `Fixes` needs the `owner/repo#N` form.
5. **Check for overlap** — `gh pr list --author <me> --state open` and open PRs touching the same
   files. An overlapping stale draft is a note in your PR, not a blocker.
6. **Create the ledger** at `$MASTER_ISSUE_DIR/<repo>-<N>/ledger.md` from `$SKILL/ledger-template.md`.
   It must let a *fresh* session continue with no conversation history: worktrees, branches, head SHAs,
   dispatched run ids, what each wait is for, retry counters. **After any context compaction, re-read
   the ledger before acting** — the summary loses details like run ids.
7. If persistent memory is available, write a resume note pointing at the ledger; update it at
   milestones (PR opened, CI verdict), not every step.

## 1. Split into items and triage

Turn the issue into numbered items — one per independently fixable problem (two symptoms of one
defect = one item; one paragraph naming two defects = two items). For each, record in the ledger:
file(s), claimed defect, arch(es), what hardware a test needs, and an initial class:

| class | meaning |
|---|---|
| `local` | testable on this box |
| `arch-other` | needs an arch/board not here — compile-only here + the CI lane that has the board (§3) |
| `test-only` | the issue asks for coverage, not a code fix — deliverable is the test; evidence = it detects an injected regression |
| `assert-only` / `api-hardening` | no runtime behavior change; look for a test that *can* observe it before parking |
| `race/timing` | needs a forced repro (`perturb` skill, NOP injection) — but see the repro gate |
| `design` | genuinely a choice between defensible behaviors → park for the user, with the options written out |

**Order of work:** cheapest-to-prove first, so PRs enter the bot/CI pipeline early and their waits
overlap with the harder items. Two items that are the same defect share one PR and one sub-issue
(say so in the sub-issue).

## 2. The scheduler — how to never idle

Each item moves through: `triage → repro → fix → verify → PR → bot-review → CI → done | parked`.

**Resources:**
- **Device** — serial, shared with other sessions. Every device run goes through
  `flock /tmp/tt-device-$(hostname).lock <cmd>`. Never hold it while reading code or building.
- **Builds** — a full tt-metal build is ~40 GB and 30–60 min; do **not** build per worktree. Keep one
  built worktree and A/B an item there by applying / reverse-applying its patch (`git apply`,
  `git apply -R`) — never `git stash` (the stash is shared across worktrees). tt-llk pytest items need
  no tt-metal build. At most 2 concurrent builds, always `run_in_background`.
- **GitHub API** — poll each run at most every 2 min (`gh run watch -i 120 --exit-status <id>`); many
  tight pollers trip secondary rate limits and then *everything* stalls.
- **Thinking/reading** — unlimited; parallelize with subagents.

**The loop**, after every action and every notification:
1. Anything finished? (bot review, CI run, build, background test) → handle it now.
2. Otherwise pick the next **unblocked** action across all items, preferring: device work if the
   device is free → items earliest in the pipeline → reading/analysis for parked items.
3. Before any wait, **arm it in the background** — a background Bash `gh run watch -i 120
   --exit-status <id>` (re-invokes you when it exits), or `Monitor` with an until-loop — then go
   straight to step 2.
4. Only when *every* item is genuinely waiting on something external may you stop, with a background
   `sleep 1800` heartbeat armed and the ledger's "pending events" table current.

**Parallelism:** fan out independent items to subagents for triage, root-causing, and writing fix+test.
**You create the worktree** (`mkwt.sh`) and pass the path — don't use `isolation: "worktree"`, its
branch names and cleanup don't fit the PR flow. Give each agent: the item text, the worktree path, the
repro gate, the device-lock rule, and "do not push, do not open PRs, do not touch GitHub — report diff,
test, and fail/pass output." You (the coordinator) own pushing, PRs, bot handling, CI, and the ledger.

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
- the other arch's code path compiles here — compile-only test + the CI lane that has that board
  (`LLK e2e Tests` runs WH+BH; tt-llk tests are auto-selected by llk-smoke on both arches);
- data that actually hits the edge (denormals, bounds around 2^31, odd tile dims, max/min pads);
- a seed sweep / repeated runs for nondeterministic bugs; the `perturb` skill for timing;
- a test that observes the assert-only/API-only change at compile time (static_assert, a
  `requires`-style check), if the repo has that idiom.

**Fix:** `mkwt.sh <issue>-<item> <me>/<branch>`. Minimal fix derived from the codebase's correct
sibling usage. **Check the sibling arch** for the same defect — same line is not the same bug, but a
one-sided fix on a symmetric defect draws a valid bot comment; fix both only if both are proven,
otherwise note it in the PR. Test goes in the most relevant existing test file; if new, wire it so CI
collects it — and check the **lane** that runs it is one the fixed file's path triggers.

**Verify, 3 ways, before any push:** fail-without + pass-with output captured to the ledger dir;
build succeeds; adversarial self-review of the diff. Run the neighboring tests too (no collateral
breakage). Pre-commit hooks: if a hook rewrites and aborts, re-add and make a **fresh** commit, never
`--amend`. Never stage `tt_metal/third_party/*` submodule pointers or symlinked `sfpi`/`.venv`.

## 4. Open the DRAFT PR

In order:
1. **Sub-issue** for the item (skip if it already exists as one — then just `Fixes` it). Body states the
   bug on its own terms: what breaks, scenario, evidence, `file:line`. Then: native child link
   (`gh api --method POST repos/<o>/<r>/issues/<PARENT>/sub_issues -F sub_issue_id=<child .id>` — the
   numeric `id`, with `-F`), assign `<me>`. Verify the link took.
2. Push the branch; `gh pr create --draft`. Body: problem, fix, the test and its fail-without /
   pass-with result, what was and wasn't run on hardware (be exact — name the board), `Fixes <child>`.
   Comment on the child with the PR link.
3. **Public-repo rule:** no RTL signal names, `.sv` paths, waveform decodes, or internal tracker ids
   in code comments, commits, PR bodies, or replies.
4. Ledger (and resume note): item → sub-issue, PR, branch, worktree, head SHA.

Then **immediately** dispatch reviews (§5) and go back to the scheduler.

## 5. Bot reviews

Known bots (GitHub account type `Bot`): `copilot-pull-request-reviewer`, `github-actions[bot]`
(LLK PR Review, skills reviewers), `cycode-security`. Humans are everyone else, whatever their
login looks like.

**Dispatch.** The LLK PR Review bot does not run on drafts by itself:
```bash
$SKILL/scripts/dispatch.sh "LLK PR Review" main -f pr_number=<PR>     # ~45 min; record the run id
```
Other reviewers may or may not fire on a draft. **Learn it on the first PR**: note which bots posted
within 60 min of the push, and treat that set as "the auto reviewers" for the run.

**Bots quiet for a SHA** = (a) every review run you dispatched for that SHA has completed (or
infra-failed and been re-dispatched once), **and** (b) 45 min have passed since the push with no new
bot comment, **and** (c) `bot_threads.sh <PR>` is empty. Pure rebases (no code change) don't need a
fresh LLK PR Review dispatch — but the 45-min quiet window still applies before any CI is spent on it.

**Every bot comment gets handled, on every push, for the life of the PR.** A CI fix, a review fix, or a
rebase is a new head SHA, bots review it again, and those comments are in scope too.
```bash
$SKILL/scripts/bot_threads.sh <PR>   # THREAD lines: unresolved inline threads; REVIEW lines: summaries/comments
```
Read each in full (no truncation). Bots are not authorities — they are often wrong. For each:

- **Valid → fix it.** Verify the claim yourself first (build, run a test, spec/ISA docs, the sibling
  call site). **Push first**, then reply with what changed and the commit, then resolve.
- **Wrong → push back.** Reply with why, citing concrete evidence (`file:line`, spec section, test
  output, the sibling that does the same). Resolve. Don't change code to appease a bot.
- **Partly valid** → fix the valid part, push back on the rest, in one reply.
- **Design choice / needs hardware not here** → no code change; leave the thread unresolved and put it
  in the report for the user with the options. Use sparingly — most comments are decidable.
- **No-findings summaries** ("No issues found") need no reply — ack the URL so it drops off the list:
  `echo <url> >> $MASTER_ISSUE_DIR/acked-<PR>.txt`. Ack a findings summary only after every finding
  in it is handled.

Reply to a THREAD with `gh api repos/<o>/<r>/pulls/<PR>/comments/<rest-id>/replies -f body='…'`,
resolve with the GraphQL `resolveReviewThread` mutation on the thread id (both ids are in the
`bot_threads.sh` line). Reply to a REVIEW with an issue comment on the PR.

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

## 6. CI

**CI is spent only on a SHA the bots are already quiet on (§5) — never on a push you expect bots
to comment on.** The review loop converges first; CI comes after. The target is **one full CI run
per PR**; every additional run must be justified in the ledger by exactly one of: a code change made
*after* bots were quiet (review had nothing left, CI exposed a real defect), or a rebase past main
breakage. A bot finding that arrives while CI is already running is handled, but the fix is held
(batched) until that CI run reports, so one re-run covers both.

Once bots are quiet on the current head SHA, dispatch on the branch and record run ids:
```bash
$SKILL/scripts/dispatch.sh "Sanity tests"               <branch>
$SKILL/scripts/dispatch.sh "Nightly tt-metal L2 tests"  <branch>
$SKILL/scripts/dispatch.sh "LLK e2e Tests"              <branch>     # if tt-llk files changed
```
("PR - Sanity tests" is not dispatchable. On drafts the PR-Gate llk lanes are skipped, so LLK e2e is
the only silicon run for LLK changes.) Nightly takes hours — arm the watchers and move on.

**Triage every failing job** (`ci_triage.sh <run_id>`) by reading the job log, not the rollup:

| verdict | evidence needed | action |
|---|---|---|
| **caused by PR** | failure touches changed code/test, or reproduces locally with the PR and not without | fix (with a test if the failure exposed a gap), push → §5 for the new SHA, then full re-dispatch |
| **unrelated, main also broken** | same job fails on `main` in the same window (`main_job_status.sh "<workflow>" "<job substring>"`; "no matching job" is *not* evidence) | arm a watcher on main's runs of that workflow; when main's job goes green (cap 4 h), `git rebase origin/main`, `push --force-with-lease`, wait for bots quiet on the new SHA (§5), *then* full re-dispatch |
| **flaky / infra** | runner setup, rate limit, `infra:timeout`, card off bus, perf a hair past band; main passes | `gh api --method POST repos/<o>/<r>/actions/runs/<id>/rerun-failed-jobs` (max 2) |
| **can't tell** | | rerun once; if still ambiguous, run the failing test locally with/without the PR if the box can |

Never call a PR green from a partial rollup — enumerate every job of every dispatched workflow.

**After a rebase**, re-run the item's test with the fix (must still pass). If main touched any of the
PR's files (`git diff <old-base>..origin/main --stat -- <files>`), redo the fail-without check too —
main may have fixed or moved the defect.

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
**for that SHA** and every job is green or has a recorded unrelated verdict with evidence (main run
URL, or log line showing infra); `bot_threads.sh <PR>` is empty (or what's left is in the report as a
decision for the user); and the fail-without/pass-with test still holds. Record it, update the PR
body's "tested on" section if it changed, and leave the PR as **draft**.

## 7. Revisit parked items

Whenever the scheduler finds nothing else unblocked, re-read each parked item with fresh eyes against
§3's list. Things learned on other items (a build now exists, a test idiom found, a CI lane that has
the other arch) often unblock them.

## 8. Final report (and the end state of the ledger)

One table: item · class · status · sub-issue · PR · fail-without/pass-with evidence · CI verdict
(green / unrelated failures with links) · bot comments: fixed / pushed back / left for the user (counts).
Then, separately:
- PRs frozen by a human request, with the quote;
- items parked for the user — each with the exact decision needed and the options;
- human review threads awaiting them, quoted briefly;
- anything run on only one arch/board.

Write it to the ledger, and update the resume note.
