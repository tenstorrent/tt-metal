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
- **416** gate-passing final-red events (unique broken head_sha) examined.
  Each was fed through the real `findBaseline()` against the live API with
  `maxWaitMs: 0` (closed history is final; the poll path must not be needed,
  and `sleep` throws if reached -- it never was).

## Results

| decision                        | count | share |
|---------------------------------|------:|------:|
| notify (parent green)           |    17 |  4.1% |
| abstain: conclusively red       |   361 | 86.8% |
| abstain: parent has no push run |    34 |  8.2% |
| abstain: parent never settled   |     4 |  1.0% |

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

## Findings that need a human decision

1. **`[skip ci]` parents blind the one-commit-back rule: 34/416 (8.2%).**
   All 34 "parent has no push run" cases are commits whose parent's message
   contains `[skip ci]` / `[ci skip]` -- those commits never get a Sanity
   tests push run, so the parent's greenness is unknowable by this rule. The
   logic abstains (safe: never misattributes), but in production each such
   event polls the full 270-minute budget before giving up, and a genuine
   break lands silently. Options if this matters: hop past parents whose
   commit message matches a skip-ci marker (bounded by consecutive skip-ci
   depth), or short-circuit the wait when the parent commit is already old.
2. **`_auto-retry-post-commit.yaml` is not 100% reliable: 4/416 (1.0%).**
   Four parents are frozen at `failure@1, completed` -- the auto-retry never
   fired, so by the finality rule they are permanently "not settled" and the
   notify event polls the full budget before abstaining. Example: red run
   #12159's parent run #12119. Worth a look at the retry workflow's own
   failure modes; a time-based finality escape hatch (a failure@<3 older than
   N hours is final -- no retry is coming that late) would also close this.

Both cases fail toward silence, never toward wrong blame. Combined they are
~9% of red events; each costs a full-budget polling job and a missed
notification.

## Method notes

- "Gate-passing final-red" mirrors the workflow's finality gate:
  `conclusion != 'success' && (conclusion != 'failure' || run_attempt >= 3)`,
  deduped per head_sha (the idempotency marker allows one post per sha).
- The naive time-order expectation (previous distinct-sha, non-cancelled run
  in the window was green => expect notify) approximates a human reading the
  run list; it is a cross-check, not ground truth -- see #13609 above.
- API cost: 883 calls for the full backtest, well within core rate limits.
