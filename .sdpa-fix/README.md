# SDPA autofix

Turns the watcher's ❌ pipelines into **draft PRs marked for human review**. It
covers both watched domains, the SDPA family and Kimi K3 (triage owners
`sdpa_op` and `k3_model`).
It runs no devices and no builds. Sibling of `~/.sdpa-watch/`: it reads that
watcher's `state.json` and reuses its config, auth, Slack bot and log extraction.

Cron runs the live copy in `~/.sdpa-fix/`; `.sdpa-fix/` in the repo is a
snapshot of its code (no ledger, proposals or logs). Edit the live copy, then
copy the changed files here before committing. `flow.html` is the interactive
walkthrough of the whole flow.

## Flow (one tick, cron `10 * * * *`)

1. **Follow-up** (live mode only). For every open autofix draft, poll the CI
   runs it dispatched:
   - All green: set the `autofix/targeted-ci` status to success and comment
     "✅ targeted CI passed". The PR stays a draft.
   - Any red: comment, mark `ci_failed`, and post to Slack.
   - PR merged or closed by a human: the ledger records `merged` or `rejected`.
2. **Triage**, per job. The watcher records the finished in-scope jobs of the
   run it reports, which may still be in progress. Each job is triaged once by
   job id. The ledger keeps `job_runs` (the last run in which each job
   finished), so a test whose job is still queued in a new run stays "still
   failing" rather than looking fixed:
   - ❌ runs go to `TRIAGE_MODEL` with a JSON schema
     (`schemas/triage.json`). Every in-scope failure gets a
     `kind`/`owner`/`fixable`/`group`.
   - ✅ runs reset streaks. A signature whose job went green while it still had
     a PR or proposal open becomes `resolved_on_main`, and in live mode the
     draft PR is commented on and closed.
   - ⚠️/🚫/🟡 runs are recorded as seen, with no other effect.
2b. **Look for fixes made by people**, every tick, for every still-failing
   signature whatever its triage kind:
   - **merged:** commits on main since the first failing run that touch the
     failing test or the files named in the error (`git log first_sha..origin/main -- <paths>`),
     mapped to their PR through `(#N)` in the title;
   - **open:** open PRs whose title or body mentions the test function or the
     failing file (the bot's own `skrstic/autofix/*` PRs excluded).

   A judge call (`prompts/judge.txt`, `schemas/judge.json`) checks each new
   candidate once against the failure; verdicts are cached in the record's
   `checked`, and `SCAN_MAX_JUDGE_PER_TICK` (8) caps the calls per tick.
   - Merged fix → `fixed_upstream` with its commit, shown as
     ":git-merged: already fixed on main by #N", confirmed by the next run that
     contains it.
   - Open fix → `fix_pending` (":git-opened: fix in progress"). The bot drafts
     no fix of its own, and each tick it checks whether that PR merged
     (→ `fixed_upstream`) or closed (→ back to `tracking`).
3. **Fix**. Eligible groups are fixed best-first, `MAX_FIX_PER_TICK` per tick
   and `MAX_NEW_PER_DAY` per day. A signature is eligible when:
   - kind is `code_regression` or `perf_threshold`, owner is `sdpa_op` or
     `k3_model`, and `fixable` is true;
   - it is still failing in the latest run;
   - its streak is at least `MIN_STREAK` (1 by default), or triage pinned a culprit commit.

   The fix stage then:
   - **GitHub de-dup.** A PR whose body has the hidden marker
     `<!-- autofixsig<sig> -->`, or an existing `skrstic/autofix/<sig8>*`
     branch, means the signature is adopted, not re-created.
   - **Fix agent.** `FIX_MODEL` runs in the isolated worktree
     `/localdev/skrstic/sdpa-fix-wt`, reset to fresh `origin/main` each
     attempt. Its allowed tools are only Read/Edit/Grep and read-only git
     (plus `git revert --no-commit`). It cannot push, commit, run pytest,
     python or tt-smi, or build. It returns a verdict (`schemas/fix.json`).
   - **Guard** (`fixlib.py guard`) rejects the diff when it:
     - is empty;
     - is over 300 lines;
     - touches `.github/`;
     - deletes a file;
     - adds skip/xfail;
     - changes a threshold without saying so.
   - **dryrun:** writes `proposals/<date>-<sig8>/` (patch, PR title and body,
     dispatch plan, verdict, guard) and posts "would open" to Slack.
   - **live:** creates branch `skrstic/autofix/<sig8>-<slug>`, commits,
     pushes, then runs `gh pr create --draft` with labels
     `made-by-ai,automated`, a title prefix of
     `[autofix · needs human review]`, and you as assignee. It sets a pending
     `autofix/human-review` commit status, dispatches the failing legs on the
     branch, and links the runs in the body.

## Slack: threaded, with emojis

**One message per fix, edited in place.** The first message about a failing
test is posted as a reply in the thread of the watcher's current digest
(`_slack.ts` in `~/.sdpa-watch/state.json`), and its ts is stored on the
ledger record (`slack_ts`). Every later status edits that same message with
`chat.update`, which sends no new notification: dry-run proposal → 🛠️ opened →
CI ✅/❌ → 🟣 merged → ✅ verified, or 📌 already fixed on main → ✅ verified.
A new reply is posted only when there is no earlier message to edit.

| | |
|---|---|
| 🛠️ opened | draft PR opened (live) |
| 🟣 merged | our PR merged; waiting for a run that contains the merge commit |
| 📌 already fixed on main | upstream commit fixes it; waiting for a run that contains it |
| ✅ verified | a run containing the fix passed that test: nothing more to add |
| ❌ still failing after fix | a run containing the fix still fails: back to `tracking` |

`merged` and `fixed_upstream` are not terminal. Each records `fix_sha`, and
`fixlib update` checks every new run of that pipeline with
`git merge-base --is-ancestor fix_sha run_sha`: runs older than the fix change
nothing; a newer green run makes it `verified`; a newer red run reopens it.

**Who owns a PR, and does it need you?** PR links are labelled by owner:

| Label | Means |
|---|---|
| ***autofix draft #N — needs your review*** (bold) | our draft PR; a human has to review it (also CI ✅/❌ variants) |
| ***autofix dry-run proposal — take a look*** (bold) | a proposal on disk, not pushed |
| autofix #N merged | our fix, merged; waiting for the run that verifies it |
| already fixed on main by #N by @author | someone else's merged fix; nothing to do |
| fix in progress by @author: open PR #N | someone else's open PR; the bot will not draft its own |

## Decisions in Slack (buttons)

When the right fix is a human call (a CI time limit, running fewer cases on a
platform, a threshold whose cause is unclear, anything that changes what a test
checks), the fix agent returns `decision.needed` with 2-3 options instead of a
patch. The bot then **opens nothing**:

1. The record becomes `awaiting_decision`. One message in the digest thread
   asks the question, with a button per option (★ = recommended) plus Reject.
2. `decide.py`, a Socket Mode listener (outbound websocket, no public URL),
   receives your click, checks `DECIDERS`, writes `decision_choice` to the
   ledger, swaps the buttons for "decided: … by @you, applying…", and starts
   `DECISIONS_ONLY=1 fixer.sh`.
3. The fixer applies it: a **patch** option becomes `decided`, and the agent
   implements only that option, guard, then a real draft PR (live even when
   cron runs dryrun, because you chose it). **Reject** sets `rejected`.
   Options are always code changes; the bot never comments on other people's PRs.

Setup: Socket Mode on, Interactivity on, an app-level token (`connections:write`)
in `~/.sdpa-fix/slack_app_token` (chmod 600). Cron runs `decide.sh` every
5 min; it keeps exactly one listener alive (flock) and does nothing without the
token. Log: `logs/decide.log`. `FORCE_SIG=<sig> sdpa-fix-dry` runs the fix
agent on one failure by hand.

## No duplicate PRs

Deduplication is by regression **signature** = `sha1(workflow :: job-without-SKU :: test-id)`,
never by run or tick. One record per signature lives in `ledger.json`, written
under flock. GitHub is the backstop: the body marker and branch prefix are
checked before every creation, so even a wiped ledger cannot double-post. A
signature gets one attempt. It re-opens only if it went green on main and
then regressed again. `rejected` (a human closed the PR) is never retried.

## Knobs (`config.sh`, env-overridable where marked)

| | |
|---|---|
| `FIX_MODE` | `dryrun` (default) / `live`. Going live re-attempts dry-run proposals once, for real. |
| `MIN_STREAK` (env) | consecutive failing runs before attempting (1: attempt on the first red run) |
| `MAX_NEW_PER_DAY` (env), `MAX_FIX_PER_TICK` (env) | caps (3, 1) |
| `FIX_SLACK=0` (env) | silence Slack posts |
| `ONLY=<workflow.yaml>` (env) | restrict a manual tick to one pipeline |
| `NO_FIX=1` (env) | triage + ledger only |

## PR style

Every PR the bot opens has the same short shape (`fixlib.py render`):

- **Warning:** automated, not run on hardware, needs human review. It also
  says when the PR changes a threshold.
- **Regression:** a red badge linking the failing run, then one line with the
  test, the measured value and the expected value or band.
- **Why:** at most two sentences, the cause and the evidence, naming the
  culprit PR.
- **Fix:** what changed, with old → new values.
- **Validation:** GitHub's live workflow badge for each dispatched leg on the
  branch, linked to the run.
- **Review:** at most three checkboxes, plus possibly related PRs.

The agent writes the `regression`, `why` and `fix` fields to these rules
(`prompts/fix.txt`). Code comments carry no history: no dated notes, no run or
job ids, no "re-centred …". A threshold change also removes the dated history
around it. The guard rejects any added comment line that looks like history.

## Shell aliases (~/.bashrc)

| | |
|---|---|
| `sdpa-fix-dry` | one dry-run tick |
| `sdpa-fix` | one live tick: pushes, opens a draft PR, dispatches CI |
| `sdpa` / `sdpa-dry` | the watcher: one tick / print the digest without posting |

Cron: watcher `0 * * * *`, fixer `10 * * * *`. `ensure-cron.sh` restores both
after a reboot.

## Everyday commands

```bash
python3 ~/.sdpa-fix/fixlib.py list                 # ledger, one line per signature
tail -F ~/.sdpa-fix/logs/today.log                 # tick log
ls ~/.sdpa-fix/proposals/                          # dry-run output (patch.diff, pr_body.md, ...)
FIX_SLACK=0 ~/.sdpa-fix/fixer.sh                   # manual tick
python3 ~/.sdpa-fix/fixlib.py mark --state rejected <sig>   # never touch this signature again
```

The watcher's digest (`~/.sdpa-watch/watch.sh`, `autofix_note`) appends one
🛠 line per autofix state under each failing pipeline.
