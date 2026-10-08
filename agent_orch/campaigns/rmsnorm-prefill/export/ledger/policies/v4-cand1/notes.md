# v4-cand1: v4 + earned depth (the previous round decides whether depth R pays)

**Change (one behavior):** plan() reads the previous round's summary. It takes the deepest
recorded attempt depth d and the lift of the best depth-d attempt over everything shallower
in that round (root included). If that lift is within noise_pct, this round caps depth at
max(2, min(R, d) - 1). A branch whose head reaches the cap is closed, and the next round
re-roots at the round best. If the deepest attempts did pay, the full plan depth (R=3) stays.
A round run at the cap whose last attempts pay earns the depth back. Active below beta 0.8.
Plan R stays 3, so it is in support. The cap is applied in select_batch, where noise_pct is known.

**Trace evidence (v4 on rounds 1-5):**
- r04 step 3 (4 attempts) added nothing to the round best (1.3997 at a02). b03-a03 -0.09%,
  b04-a03 -0.34%, and b01-a03/b02-a03 are forbidden_edit (HiFi2) after the r04 override. Cost 0.02 V.
- r05 step 3 (2 repairs) recovered to 1.4661/1.4881 but stayed below the leader 1.5696. Cost 0.0133 V.
- The previous round predicted both. In r03 the depth-3 attempts lifted the best -0.11% (1.3318 vs 1.3333),
  and in r04 -0.34% (1.3949 vs 1.3997, HiFi2 nodes invalid). Depth 3 paid only in r01, where the
  code was far from optimized (0.84 -> 1.2075 repair).
- The r02/r03 caps don't change anything: v4 already stops there via lone-step and stall rules.

**Replay (rounds 1-5, beta 0.6):** mean V **1.3511** vs v4 1.3444 (+0.0067, more than one attempt's cost).
r01 1.1942 (8/3), r02 1.2585 (4/1), r03 1.3333 (8/2), r04 **1.3997** (8/2, was 1.3797), r05 **1.5696** (6/2, was 1.5563).
Sweep: 0.2 1.3340 / 0.4 1.3511 / 0.6 1.3511 / 0.8 1.3334 / 1.0 1.3319.

**Risks:**
- Cross-round value replay can't price. The capped r05 a03 repairs found the "un-gate stick push" fix that
  the r05 summary plans to port to the 2-wave best in r06. Under the cap that idea has to come from the
  next round's workers, who will see the a02 failure analysis.
- Round 6 will run with cap 2 (r05's a03s lifted -5.19%). If depth 2 pays in r06, r07 gets R=3 back.
  If depth 2 doesn't pay, the cap stays at 2 (floor), so the rule can't shrink rounds below one refinement.
