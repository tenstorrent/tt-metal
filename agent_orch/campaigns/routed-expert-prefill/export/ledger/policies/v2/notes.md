# v2: one-step rounds after the first (W=4, R=1)

Promoted from `v1-cand4` after the r04 dreaming. The code is the same; only the docstring's first
lines changed.

**What it does differently from v1:**
- **Rounds after the first are one step deep.** Once there is history, `plan()` returns W=4, R=1
  (R=2 above beta 0.7). A round is four independent attempts from the round root, and the next
  round re-roots on the best node.
- **Round 1 is unchanged.** With no history it keeps the campaign defaults (4 x 4) and v1's
  `select_batch`: frontier prune, collapse close, recover role.

**Replay (r01-r04, beta 0.6):** V 1.5703 vs v1 1.5511 (+0.0192, about four attempts' cost).

| round | V | vs v1 | attempts | best |
|---|---|---|---|---|
| r01 | 1.4736 | 0 | 11 | 1.5011 |
| r02 | 1.5514 | +0.0159 | 4 | 1.5314 |
| r03 | 1.6087 | +0.0259 | 4 | 1.5887 |
| r04 | 1.6475 | +0.0350 | 4 | 1.6275 |

v2 trades 0.85% of mean per-round best (1.5622 vs 1.5755) for about 45% fewer attempts per round
and full-width steps. It bets that re-rooting compounds what the dropped steps 2-4 used to find.

**Why:** in r02-r04, step 1 set each round's largest gain. Steps 2-4 never paid for their attempts
at cost 0.005, even in r02, where they added a real +2.5%. Those steps were mostly rebases onto the
leader or one hypothesis copied onto every branch, and a re-root delivers the same thing at full
width. See `../v1-cand1..5/notes.md`:
- the leader guard can't be tested in replay;
- the flat stop and depth 2 are about +0.005;
- an in-round repair step costs one attempt.

**What to watch online:**
- **Round overhead.** Each round adds a root build, a drift check and dreaming, about 10 min
  between r03 and r04. If that overhead dominates, raise beta above 0.7 to get R=2.
- **Failed first attempts.** Check whether failed step-1 ideas get retried from history in the next
  round.
- **Duplicate hypotheses in step 1.** Watch whether several workers in the same step pursue one
  hypothesis.
- **Support collapse.** R=1 rounds record no depth, so future dreaming can judge depth only from
  r01-r04.
