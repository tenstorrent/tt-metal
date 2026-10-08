# v3: v2 + stall closure + no lone trailing step (promoted from v2-cand4)

What changed vs v2: two batch rules. The plan is unchanged (W=4, R = 2 + round(2*beta) = 3).
1. **Stall closure:** a branch is closed once its last refinement (2 refinements
   from beta 0.8) was valid and within noise_pct of the branch's prior anchor.
   A clear regression is a repair and stays open. First attempts are not judged.
2. **No lone trailing step:** if the next batch would be a single refinement of a
   branch that doesn't hold the round lead, stop (below beta 0.8).
The gap-to-leader closure (noise_pct * (2 + 8*beta)) and R from beta are unchanged from v2.

Why: in r02 and r03, attempts after a within-noise refinement never lifted
their branch (r02-b02 +0.09%; r03 b01/b02/b03 +0.26% / -2.5% / -0.7%). In r03
the only step left after those closes was a lone repair on a trailing branch. Under
the parallel bonus that step costs about 1.1% of score, and it didn't lift the
round best (1.3318 < 1.3333). Evidence and risks: `../v2-cand1/notes.md`,
`../v2-cand4/notes.md`.

Replay (rounds 1,2,3, beta 0.6): mean V 1.2553 vs v2 1.2457.
r01 1.1942 (8 attempts / 3 steps, unchanged), r02 1.2385 (5 / 2), r03 1.3333 (8 / 2).

Candidates considered:
| id | change | mean V |
|---|---|---|
| v2 | baseline (gap close, R=3) | 1.2457 |
| v2-cand1 | + stall closure | 1.2503 (tie with v2) |
| v2-cand2 | cand1 + step-1 duplicate closure (score band + tag overlap) | 1.2553 (tie with cand4, bigger rule) |
| v2-cand3 | cand1 with R back to 4 | 1.2483 |
| v2-cand4 | cand1 + no lone trailing step | **1.2553** |
| v2-cand5 | v2 + duplicate closure only | 1.2490 (tie with v2) |

Notes for the orchestrator:
- r01's beta sweep is unchanged from v2 because neither new rule fires in r01. The
  gain comes only from r02/r03. Both new rules are cutting rules, so the
  "no capacity after the prune" concern from the r02 summary still can't be scored.
- r03 has no summary.md in the ledger yet. Evidence for r03 was read from
  history.md and decisions.jsonl.
