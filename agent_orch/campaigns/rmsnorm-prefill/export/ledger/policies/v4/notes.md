# v4: v3 + no lone step at all (promoted from v3-cand1)

What changed vs v3: one batch rule. Below beta 0.8, if the next batch would be a single
refinement, the policy stops and the next round re-roots. v3 did this only when the lone
branch trailed the leader. v4 also does it when the lone branch holds the lead.
Plan, gap closure, stall closure and beta schedule are unchanged (W=4, R=3 at beta 0.6).

Why: a one-item step costs about as much V as a full-width step (one attempt plus the
loss in mean batch width, about 0.02 V after a 4-wide step), while it tests one idea.
In r02, v3 spent that step on the leader b02 and got 1.2292 (-0.75%). The recorded b02
line plateaued (1.2385 / 1.2292 / 1.2396 / 1.2346). Re-rooting on the leader gives the
next round full width, and r03 shows that pays (+7.6% from the same node with 4 branches).
Details: `../v3-cand1/notes.md`.

Replay (rounds 1-4, beta 0.6): mean V **1.3034** vs v3 1.2984 (+0.0050, one attempt's
cost; r02 alone gains 0.020).
r01 1.1942 (8 attempts / 3 steps), r02 1.2585 (4 / 1), r03 1.3333 (8 / 2), r04 1.4275 (12 / 3).
Sweep: 0.2 1.2751 / 0.4 1.3038 / 0.6 1.3034 / 0.8 1.2896 / 1.0 1.2878.

Candidates considered:
| id | change | mean V |
|---|---|---|
| v3 | baseline (gap close, stall close, no lone trailing step, R=3) | 1.2984 |
| v3-cand1 | no lone step at all | **1.3034** |
| v3-cand2 | cand1 + stop when all step-1 results converge within noise | 1.3058 (tie with cand1, +0.0024; skips r03 steps whose discoveries rooted r04) |
| v3-cand3 | cand1 + R=4 for branches still climbing past depth 3 | 1.3019 (r01 4th attempts don't pay; r04 depth out of support) |
| v3-cand4 | cand1 + tighter gap (2+6*beta) | 1.3038 (tie; closes r04-b02, which recovered by porting) |
| v3-cand5 | cand1 + lone climbing branch may continue | 1.3034 (identical in replay; bigger rule) |

Notes for the orchestrator:
- cand2 has the highest replay V, but its margin over cand1 is below one attempt's cost.
  Its gain comes from skipping r03 steps 2-3. Those steps produced the L1 stats scratch and
  streamed gamma that r04's 1.4475 stack was built on. Replay can't price that cross-round
  value, so it was not promoted.
- R stays at 3. The r04 summary flags that R=3 cut off still-improving branches. The only
  scoreable evidence (r01 4th attempts after a climb) says depth 4 doesn't pay (cand3).
  The next round re-roots at r04-b01-a03, so that line continues anyway.
- All recorded attempts in r02-r04 were valid, so the repair/failure rules are still only
  exercised by r01.
