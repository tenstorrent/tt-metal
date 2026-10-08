# v2: v1 with a shallower round, R from beta (promoted from v1-cand2)

What changed vs v1: one behavior, in the plan only. R = clamp(2 + round(2*beta),
2, default R), which is 3 at beta 0.6. W stays 4, and the batch rule is v1's unchanged
(open W branches, refine every open branch, gap-close a branch whose anchor
trails the leader by more than noise_pct * (2 + 8*beta)).

Why: in r01 and r02, none of the five 4th attempts lifted its round's best by
more than noise_pct, while 3rd attempts carried the large repairs. Rounds re-root
at the previous best, so a lineage that is still climbing continues next round.
Evidence and risks: see `../v1-cand2/notes.md`.

Replay (rounds 1,2, beta 0.6): mean V 1.2119 vs v1 1.2051.
r01 1.1942 (8 attempts, 3 steps, best 1.2075). r02 1.2296 (6 attempts, 3 steps, best 1.2396).

Candidates considered:
| id | change | mean V |
|---|---|---|
| v1 | baseline (gap-to-leader pruning, R=4) | 1.2051 |
| v1-cand1 | + plateau close (k=2 attempts, anchor lift <= noise) | 1.2089 (tie with v1) |
| v1-cand2 | R = 2 + round(2*beta) -> 3 | **1.2119** |
| v1-cand3 | cand2 + cand1 | 1.2119 (exact tie, more rules) |
| v1-cand4 | final-attempt gate: only a climbing leader takes attempt R | 1.2102 (tie with cand2, more complex) |
| v1-cand5 | depth cap learned from history | 1.2089 (can't act in round 1) |

Replay caveats (for the orchestrator):
- The r02 view scores the round root as 1.0, not as r01-b04-a04's 1.2132.
  `delta_vs_parent` of r02 first attempts and `view.best()` with no valid node
  therefore use 1.0. The gap rule is unaffected (it compares to the leader).
- r02 has one recorded attempt on b01/b03/b04 (v1 closed them). Any policy
  that keeps one of those open is out of support. The keep-open refinements reveal
  nothing, and if the branch stays legal the replay runs to max_replay_steps (100), which
  collapses the parallel bonus. So the r02 summary's "no exploration capacity
  after the prune" concern can't be scored yet. Loosening the gap only costs
  attempts on r01 (see v0-cand3).
