# v3-cand1: v3 + no lone step at all (leader included)

**Change (one behavior):** v3 stopped instead of running a single-item step only when the
lone branch trailed the round leader. cand1 drops that condition: below beta 0.8 a
batch that would be a single refinement becomes a stop, even if that branch holds the lead.

**Trace evidence:** with v3, r02 runs step 2 as one refinement of the leader b02
(1.2385), after the gap rule closed b01/b03/b04. It reveals 1.2292 (-0.75%), and then
the stall rule stops anyway. That lone step costs 0.005 (one attempt) plus 0.015 of
parallel bonus (mean width 4 -> 2.5), about 0.02 V, so it would have needed a ~1.6%
lift from one attempt. The recorded b02 line went 1.2385 / 1.2292 / 1.2396 / 1.2346,
a plateau. The r02 summary also says that with one branch left, steps 2-4 stayed within noise.
Re-rooting on the leader restores full width next round, and r03 showed that pays off
(+7.6% over the r02 best with 4 branches from the same node).

**Replay (rounds 1-4, beta 0.6):** mean V **1.3034** vs v3 1.2984 (+0.0050).
r01 1.1942 (8/3, unchanged), r02 **1.2585** (4 attempts / 1 step, was 1.2385),
r03 1.3333 (8/2, unchanged), r04 1.4275 (12/3, unchanged).
Sweep: 0.2 1.2751 / 0.4 1.3038 / 0.6 1.3034 / 0.8 1.2896 / 1.0 1.2878.

**Risk:** a lone branch that is climbing fast gets stopped. No recorded round had that
case at a lone step (cand5 tests an exception for it; replay can't tell them apart).
