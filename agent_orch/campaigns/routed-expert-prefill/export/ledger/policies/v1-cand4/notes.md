# v1-cand4: one-step rounds after the first (W=4, R=1)

**Change (one behavior over cand3):** once there is history, `plan()` returns W=4, R=1 (R=2 above
beta 0.7). A round is four independent attempts from the round root, and the next round re-roots
on the best node. Round 1, which has no history, keeps the campaign defaults and v1's
`select_batch` (frontier prune, collapse close, recover role).

**Evidence, from the replay and the round summaries:**
- **Step 1 carries the round.** In every re-rooted round, step 1 set the largest single-step gain:
  - r02: +2.0% (1.5011 → 1.5314);
  - r03: +1.2% (the only gain outside noise);
  - r04: +1.07% (all of the round's gain).
- **Steps 2-4 never paid for their attempts at cost_per_attempt 0.005.** With the full 4 x 4 grid,
  V by stopping step is:

  | round | stop after s1 | s2 | s3 | s4 |
  |---|---|---|---|---|
  | r02 | **1.5514** | 1.5314 | 1.5235 | 1.5297 |
  | r03 | **1.6087** | 1.5952 | 1.5903 | 1.5703 |
  | r04 | **1.6475** | 1.6275 | - | - |

  Even r02's real late gain (+2.5% over steps 2-4) costs 12 attempts (0.06).
- **Later steps duplicate work.** Steps 2-4 mostly ran rebases onto the leader, or one shared
  hypothesis on every branch (r03 step 4: 3 of 4 on SrcB; r04 step 2: 4 of 4 on activated-stream
  decoupling). A re-root hands every worker the leader's code at full width.

**Replay (r01-r04, beta 0.6):** V 1.5703 (+0.0192 vs v1, about 4 attempts' cost; +0.0134 vs cand3).

| round | V | vs v1 | attempts | best |
|---|---|---|---|---|
| r01 | 1.4736 | 0 | 11 | 1.5011 (defaults) |
| r02 | 1.5514 | +0.0159 | 4 | 1.5314 |
| r03 | 1.6087 | +0.0259 | 4 | 1.5887 |
| r04 | 1.6475 | +0.0350 | 4 | 1.6275 |

- **Beta sweep:** 1.5703 at beta 0.2-0.6 (5.75 attempts, mean best 1.5622) and 1.5569 at beta
  0.8-1.0 (R=2, 8.75 attempts, mean best 1.5638).
- r04's replay was recorded under v1, so it can't reward patience. Leaving r04 out, cand4 still
  leads v1 by +0.014 on r01-r03.

**Risks online (not in the objective):**
- **Per-round overhead.** Each round adds a root build, a drift check and a dreaming phase, about
  10 min between r03 and r04. With 30-min steps, that becomes a larger share of wall time.
- **Repairs move to the next round.** A failed first attempt is never repaired in place. Its idea
  survives only through the shared history (cand5 tests an in-round repair step).
- **Support collapse.** Rounds recorded under R=1 have no depth, so later dreaming can only judge
  depth from r01-r04. The beta knob (R=2 above beta 0.7) is the way back.
- **Duplication within step 1 stays possible.** In r04, two of four step-1 workers targeted the
  gate/up handshake.

**Verdict:** the winner, promoted as v2.
