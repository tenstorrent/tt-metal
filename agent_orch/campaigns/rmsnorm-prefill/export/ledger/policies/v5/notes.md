# v5: v4 + earned depth (promoted from v4-cand1)

What changed vs v4: one plan-time signal. plan() looks at the previous round's deepest recorded attempts.
If the best of them lifted that round's best by no more than noise_pct, this round caps depth at
max(2, min(R, d) - 1) (2 in practice). Branches reaching the cap are closed, and the next round re-roots at
the round best. If the deepest attempts did pay, R=3 stays. A capped round whose depth-2 attempts pay earns
depth 3 back. Plan R is still 3 (in support). Gap closure, stall closure, the lone-step stop and the beta
schedule are unchanged. The cap is off from beta 0.8.

Why: on rounds 1-5, depth-3 attempts paid only in r01 (repairs of a far-from-optimized kernel). In r03 and r04
they lifted the round best by -0.11% and -0.34%, and in r05 by -5.19%. v4 spent r04 step 3 (4 attempts) and
r05 step 3 (2 repairs) without moving the best, and the previous round's depth-3 result predicted both.

Replay (rounds 1-5, beta 0.6): mean V **1.3511** vs v4 1.3444 (+0.0067, more than one attempt's cost).
r01 1.1942 (8/3), r02 1.2585 (4/1), r03 1.3333 (8/2), r04 1.3997 (8/2), r05 1.5696 (6/2).
Sweep: 0.2 1.3340 / 0.4 1.3511 / 0.6 1.3511 / 0.8 1.3334 / 1.0 1.3319.
Round 6 (rooted at r05-b01-a01) will run with cap 2, because r05's depth-3 attempts lifted -5.19%.

Candidates considered:
| id | change | mean V |
|---|---|---|
| v4 | baseline (gap close, stall close, no lone step, R=3) | 1.3444 |
| v4-cand1 | earned depth: cap at d-1 when the previous round's deepest attempts didn't pay | **1.3511** |
| v4-cand2 | close branches whose head is forbidden_edit | 1.3447 (tie) |
| v4-cand3 | lenient earned depth: cap at (deepest paying depth + 1) | 1.3484 (misses r05) |
| v4-cand4 | aggressive earned depth: cap at deepest paying depth, floor 1 | 1.3504 (tie; loses r04 depth-2 stacks) |
| v4-cand5 | cand1 + forbidden_edit close | 1.3511 (tie with cand1, bigger rule) |

Notes for the orchestrator:
- Cross-round value replay can't price: r05's a03 repairs found the "un-gate stick push" fix that r06 plans to
  port. With the cap, the r06 workers get it from the r05 nodes and reflections, which is the same route a
  re-root would take anyway.
- cand2/cand5 (close forbidden_edit heads) is principled but has no replay value now that verify_node rejects
  forbidden patterns online. Adopt it if a forbidden head shows up in a live round.
- All recorded attempts in r02-r05 were valid apart from the r04 overrides, so the build/accuracy repair rules are
  still only exercised by r01.
