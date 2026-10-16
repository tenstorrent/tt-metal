# v2-cand3: cand2 + a rescue step when step 1 finds no gain

**Change (one behavior over cand2):** rounds after the first plan R=2 instead of R=1. Step 2 runs
only if no step-1 node beat the root by more than `0.4 + beta` x noise (1x noise at beta 0.6). It
then refines every open head. If step 1 found a gain, the round stops after step 1 exactly as in v2.

**Evidence / motivation:**
- **Diminishing-returns hedge:** round gains were +4.6%, +2.6%, +1.07%, +2.4%, +1.6% and +2.7% (r07).
  A step 1 with no gain outside noise is plausible.
- **What v2 does in that case:** it ends the round and re-roots on a node no better than the root.
  That pays the full round overhead to repeat step 1.

**Replay (r01-r07, beta 0.6):** V 1.6346, identical to cand1 and cand2. Every recorded round after
the first beat its root by more than 1x noise in step 1 (r07 by +2.7%), so the rescue never fires at
the default beta.
- **Beta sweep:** 0.8 → 1.6285 (5.71 attempts), 1.0 → 1.6295 (6.14). The band widens to 1.2-1.4%,
  the rescue fires on r03 (+1.2%) and/or r04 (+1.07%), and the recorded step 2 never beat step 1.

**Verdict:** a tie that replay cannot score by construction. Keep it on file; revisit if a round's
step-1 gain lands inside noise.
