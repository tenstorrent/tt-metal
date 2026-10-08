# Lessons

Rules learned from real mistakes. Each line starts with the agents that must
follow it; `fixer.sh` and `watch.sh` inject the matching lines into every
prompt (triage, fix, judge, skeptic, watcher). `ops` lines are for whoever
maintains the bots.

**When a human corrects the bot, add one line here, phrased as a rule, and fix
the source that taught the wrong thing.** The procedure is in `CLAUDE.md` (here
and in the global `~/.claude/CLAUDE.md`).

## Claiming that a PR fixes a failure

- [judge,skeptic,triage] A guessed cause in a summary ("likely a race", "uninit state") is not evidence. Reason from the raw log, and never match a guess against a PR title.
- [judge,skeptic] Match the failure MODE, not the topic. A race / missing barrier gives wrong values (PCC), not a hang. A hang comes from a deadlock (CB push/pop mismatch, a semaphore never signalled, a loop that never ends) or from a device assert halting the core.
- [judge,skeptic,triage,watcher] In the LLK-asserts and watcher debug lanes, a device assert that fires halts the Tensix core silently, and the test hangs until the timeout with no message. A hang seen only in such a lane (plain lane green) points to an assert tripping, not to a race.
- [judge,skeptic] Claim a fix only with confidence=high, the same failure mode, a stated diff → symptom chain (including the lane / SKU it appears in), and a skeptic that cannot refute it. Medium is a "no". A wrong "fix in progress" is worse than none. (#59427 was claimed twice, for a sim timeout and an LLK-asserts hang; it fixes neither.)
- [judge] A bundle PR counts only if one of its changes, on its own, removes the cause.

## Classifying failures (triage)

- [watcher,triage] Describe the symptom, not a guessed mechanism: write "hung with no output under LLK asserts, likely a device assert halting the core", never "likely a race / uninit state". A guessed mechanism in a summary misled the fix judge twice. (The watcher's own Sanity Debug hint taught that wrong rule; corrected.)

- [triage] A timeout is not "infra" by default. First compare the job's runtime with its recent history and its limit, and check what landed in between. A job sitting at its limit for days, pushed over by new test cases, is a test-time regression with a culprit PR. (ttnn sdpa group on sim: 39-41 min against 40, pushed over by #59055.)
- [triage] A missing file or path right after a CI or config change (a renamed mount, a moved golden trace) is a config regression owned by whoever owns the test (k3_model for K3 runners), not "missing_data / other". (K3 SC1 trace path, fixed by #59432.)
- [triage,watcher] Name the debug mode (plain / watcher / LLK asserts) and the SKU of every failure. A failure only under asserts or watcher is a different signal from a plain-lane failure.

## Writing fixes

- [fix] Changing a CI time limit, a time budget, or what a test runs on a platform is a human call: return a decision poll, never edit it directly. For time-budget overruns, offer offsets: lower groups with >= 3x headroom (from `fixlib.py runtime-stats`), same owner first, or shrink the increase.
- [fix] Never offer or make a skip / xfail / deselect, even as a decision option; the guard rejects it anyway.
- [fix] No history in code comments (dates, run or job ids, "re-centred …"). When changing a threshold, delete the dated history around it and keep only the durable rule.
- [fix] PR text is short and direct: regression with numbers, why in at most 2 sentences, fix as old → new, at most 3 review checks.

## Validating and babysitting PRs (code, enforced; listed so agents know)

- [ops,fix] Judge a targeted CI run job by job: the fix's own jobs must pass, and failures that also fail on main in the last 24 h are pre-existing. Never use the run's overall conclusion. (#59658: run "failure" with 9 failures, all pre-existing.)
- [ops] Validation must be on the head that would merge. When the PR head moves, cancel stale targeted runs and re-dispatch.
- [ops] The bot's own `autofix/*` commit statuses are not PR checks; never "repair" them.
- [ops] Every workflow has its own dispatch inputs (sanity-tests-debug.yaml ≠ sanity-tests.yaml). Map them per workflow, and treat a rejected dispatch as "run by hand", not a crash.

## Platform gotchas (ops)

- [ops] `gh pr edit` fails on gh 2.63 (Projects classic GraphQL): edit PR bodies with `gh api -X PATCH repos/…/pulls/N -F body=@file`.
- [ops] Outside a Claude Code session, `git push` needs `-c credential.helper= -c 'credential.helper=!gh auth git-credential'` (the global helper points at a missing /usr/bin/gh).
- [ops] The GitHub run list is stale for `branch=` / `event=` filters: always add `created=>(last 10 days)`.
- [ops] Never call a shell helper that sets variables inside `$( … )`; the subshell drops them.
- [ops] `pkill -f` patterns must be anchored (`^python3 …`), or they match your own shell.
- [ops] `label` is a reserved word in jq.

## Known open gaps

- [ops] Sanity Debug runs 3 variants a night (plain, watcher, LLK asserts), but the watcher reports only the latest run, so a variant that fails can be hidden by the next one. The watcher lane timed out (4 h) for 5+ days unseen. Fix: track Sanity Debug per variant.
