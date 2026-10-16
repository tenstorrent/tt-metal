# v2-cand1: v2 + late steps only continue repairs (r07 dreaming)

**Change (one behavior over v2):** from step 3 on (step 4 above beta 0.7), a branch is refined only
if its head or the attempt before it failed or scored below the round root. A valid, above-root line
ends there. Only deep rounds are affected, which means round 1 (4 x 4 defaults). Rounds after the
first are one step deep and replay exactly as v2.

**Evidence (v2 traces, r01):**
- At step 3, v2 refined the valid leader b01 (1.4716 → 1.0363, wasted) together with b04 (failed →
  1.4622 → 1.5011, the round's best).
- Both of r01's late recoveries (b03-a03/a04, b04-a03/a04) were repair or rebase lines. The valid
  leader's late attempts added nothing.
- Same rule as the r05/r06 phases' v2-cand1, now replayed on r01-r07.

**Replay (r01-r07, beta 0.6):** V 1.6346 vs v2 1.6342 (+0.0004).
- r01 goes from 11 to 10 attempts (V 1.4761 vs 1.4736). r02-r07 are identical to v2.
- **Beta sweep:** 1.6346 at beta 0.2-0.6. At beta 0.8-1.0 it scores 1.6266, the same as v2, because
  the late step moves to step 4 and v2's full high-beta second step is unchanged.

**Verdict:** a tie: +0.0004 is under one attempt's cost (0.005). It rests on one branch of one round
(see cand5). It is the base for cand2-cand5.
