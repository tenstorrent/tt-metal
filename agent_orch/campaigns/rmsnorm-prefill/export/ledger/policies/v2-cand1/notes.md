# v2-cand1: v2 + stall closure

**Change (one behavior):** close a branch once its last `stall_k` refinements
(1 below beta 0.8, 2 from 0.8) were valid and each landed within noise_pct of
the branch's prior anchor, measured relative to that anchor. A clear regression
(more than noise_pct below the anchor) is not a stall. It counts as a repair in
progress and keeps its slot. First attempts are never judged by this rule
(the gap rule handles them). Plan unchanged (W=4, R=3 at beta 0.6).

**Trace evidence (v2 on r02, r03):**
- r02: b02 1.2385 -> 1.2292 (-0.75%). v2 spent a third attempt (1.2396,
  +0.09%) that bought nothing.
- r03: step 2 left three branches within noise of their anchors: b01 +0.69%,
  b02 +0.77%, b03 -0.42%. Their third attempts went +0.26%, -2.5% and -0.7%.
  Not one of the three lifted its branch, and the round best (b02-a02) was already in hand.
- r01 is unaffected. Its only within-noise refinement is b04-a04, which R=3
  never reaches. The two -19%/-22% dips at step 2 are repairs and stay open.
- Why k=1: v1-cand1 used k=2 because r01-b02's first attempt sat within noise of the
  root. First attempts are now excluded, so that case doesn't apply.

**Replay (rounds 1-3, beta 0.6):** mean V 1.2503 (v2 1.2457).
r01 1.1942 (unchanged), r02 1.2385 (5 attempts / 2 steps), r03 1.3183 (9 / 3:
only b04's repair runs at step 3, reaching 1.3318, still below the leader).
Sweep: 0.2 1.2269 / 0.4-0.6 1.2503 / 0.8 1.2437 / 1.0 1.2412.

**Verdict:** +0.0046 over v2, which is less than one attempt's cost. On its
own it ties v2. It is the base for cand2-4.
