# v3-cand3: v3-cand1 + extra depth for climbing branches

**Change (one behavior on top of cand1):** plan R = campaign default (4) instead of
3. Up to the old soft depth (3 at beta 0.6) every open branch refines as before.
Past it, a branch continues only if its last attempt raised its anchor by more than
2*noise_pct.

**Motivation:** the r04 summary says R=3 cut every branch off while it was still
improving (b01 +4.1%, b02 +7.9% at a03). That is the "premature stop" pattern.

**Replay:** mean V 1.3019 (-0.0015 vs cand1). r01 **1.1882**: b01 (+7.0%) and b04
(+12.2%) were both climbing at a03, so step 4 runs. It lifts the best only 1.2075 ->
1.2132 (+0.5%) for 2 attempts plus a 4th step. r02/r03 are unchanged. r04 is unchanged
because a 4th attempt is out of support (recorded R=3), so the upside can't be scored.
Sweep: 0.2 1.2829 / 0.4 1.3023 / 0.6 1.3019 / 0.8 1.2896 / 1.0 1.2878.

**Verdict: not promoted.** The only recorded case of "climbing at the soft depth"
(r01) says the 4th attempt doesn't pay. In r04 the next round re-roots at the
climbing branch's best (r04-b01-a03), so the depth continues there anyway.
