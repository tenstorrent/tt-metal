# Backtest: one-commit-back sanity Slack notify vs. real main history

Backtest of the decision logic in `culprits.js` (used by
`.github/workflows/sanity-tests-slack-notify.yaml`) against real,
already-settled "Sanity tests" push-to-main run history, run 2026-08-21
via `backtest.js`. Reproduce with:

```
GITHUB_TOKEN=$(gh auth token) node .github/scripts/sanity-notify/backtest.js --days 30
```

## Coverage

- Window: **2026-07-22 .. 2026-08-21** (30 days).
- **1088** push-to-main "Sanity tests" runs fetched (chunked queries; the
  list API silently caps any single filtered query at 1000 results).
- **417** gate-passing final-red events (unique broken head_sha) examined.
  Each was fed through the real `findBaseline()` against the live API with
  `maxWaitMs: 0` (closed history is final; the poll path must not be needed,
  and `sleep` throws if reached -- it never was).

## Results (after the [skip ci] short-circuit)

| decision                      | count | share | before the short-circuit |
|-------------------------------|------:|------:|--------------------------|
| notify (parent green)         |    17 |  4.1% | 17 (byte-identical set)  |
| abstain: conclusively red     |   361 | 86.6% | 361                      |
| abstain: parent is [skip ci]  |    35 |  8.4% | 34 as full-budget timeouts + 1 as conclusively-red (nuance below) |
| abstain: parent never settled |     4 |  1.0% | 4 (unchanged; distinct category, still full-budget timeouts) |

The short-circuit is a latency fix, not a decision change: every event
abstains or notifies exactly as before, but the [skip ci] cases now
abstain instantly instead of burning the full 270-minute poll budget
first. One extra event vs. the first pass (#13812) simply landed on main
between the two runs.

- **notify (17)**: every baseline was the parent's own run, and all 17 agreed
  with the naive time-ordered reading of the run list. Spot-checked by hand:
  #13790 (the 2026-08-21 incident-day break; baseline #13789 -- matches the
  correct notification actually sent that morning), #12717 (parent run #12714
  success@1), and #11483 (parent run #11482 success **on attempt 2** -- a real
  case where the auto-retry flipped a failure to green, validating the
  any-final-success rule). Three of the 17 events were `cancelled@1` and one
  `startup_failure@2`; the finality gate passes those immediately by design.
- **conclusively red (361)**: main spends long stretches red; each red streak
  is one notify followed by many correct silences. #13802 (the incident's
  89-mention mass ping) resolves here: its parent's run #13798 is
  failure@attempt-3, so the new logic stays silent instead of blaming 250
  commits.
- **One time-order disagreement, resolved in the new logic's favor**: #13609
  abstains (its git parent's run #13607 was failure@3) even though the
  time-ordered run list shows a later green run in between -- merge-queue
  interleaving means list order is not commit order. A human reading the flat
  list would have pinged someone whose parent state was in fact red.

## [skip ci] parents: known, ACCEPTED gap (maintainer decision, 2026-08-21)

**8.4% (35/417)** of red events sit on top of a `[skip ci]` / `[ci skip]`
parent. Such a parent (usually) never gets a push run, so its greenness is
unknowable under the one-commit-back rule and the break above it goes
un-notified. The maintainer decision is to accept that missed notification
rather than complicate the rule -- but not to spend the poll budget
discovering it: `findBaseline` checks the parent's commit message and its
associated PR titles upfront and abstains immediately
(`reason: 'parent-skip-ci'`) with a distinct log line, entering no poll
loop at all.

Nuance surfaced by the re-run: one event (#12597) has a parent whose
message says `[skip CI]` yet a push run exists anyway (#12596, red) -- the
marker does not always suppress CI. The short-circuit preempts that run's
verdict, which changed the abstain *reason* (conclusively-red ->
parent-skip-ci) but not the abstain itself. In principle this arm could
also preempt a *green* marked parent (suppressing a legitimate notify);
zero such cases in this window, and it falls inside the accepted
[skip ci] trade-off.

## Current-sha sibling runs: known, ACCEPTED gap (maintainer decision, 2026-08-21)

`findBaseline` settles the PARENT's runs but takes the triggering run's
final-red conclusion for the current sha as-is. Duplicate push deliveries
can give one sha several runs, so in theory a sibling run of the failing
sha could have succeeded (or still be in flight), making a notification a
false blame. Deliberately not handled: this 30-day window contains exactly
one duplicate-run pair (#13788/#13789) and its outcomes matched
(success/success); zero split-outcome pairs were observed. Revisit if a
bad notification is ever traced to a green sibling of the blamed sha.

## Finding that still needs a human decision

**`_auto-retry-post-commit.yaml` is not 100% reliable: 4/417 (1.0%).**
Four parents are frozen at `failure@1, completed` -- the auto-retry never
fired, so by the finality rule they are permanently "not settled" and the
notify event polls the full budget before abstaining. Example: red run
#12159's parent run #12119. These are NOT [skip ci] commits and are
deliberately untouched by the short-circuit. Worth a look at the retry
workflow's own failure modes; a time-based finality escape hatch (a
failure@<3 older than N hours is final -- no retry is coming that late)
would also close this.

All abstain categories fail toward silence, never toward wrong blame.

## Method notes

- "Gate-passing final-red" mirrors the workflow's finality gate:
  `conclusion != 'success' && (conclusion != 'failure' || run_attempt >= 3)`,
  deduped per head_sha (the idempotency marker allows one post per sha).
- The naive time-order expectation (previous distinct-sha, non-cancelled run
  in the window was green => expect notify) approximates a human reading the
  run list; it is a cross-check, not ground truth -- see #13609 above.
- API cost: 1615 calls for the full backtest (the skip-ci check adds a
  commit lookup per event and a PR lookup for unmarked parents), well
  within core rate limits.
