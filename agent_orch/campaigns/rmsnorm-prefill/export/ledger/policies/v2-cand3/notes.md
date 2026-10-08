# v2-cand3: cand1 with the depth cap removed (R = default 4)

**Change:** plan returns the campaign defaults (W=4, R=4). The stall rule is meant
to decide depth per branch instead of v2's fixed R=3.

**Evidence tested:** v2's notes say R=3 can cut a lineage that is still climbing. If
the stall rule already stops flat branches, the fixed cap might be redundant.

**Replay:** mean V 1.2483. r01 1.1882: b01 (+7%) and b04 (+12%) are both
still climbing after a03, so both take a 4th attempt (+1.5% and +0.47%, with
the round best moving 1.2075 -> 1.2132 within noise). That is 2 attempts and a step for
less than noise. r02 1.2385, r03 1.3183: b04-a03 lifted +1.4%, so b04 asks
for a4, which is out of support, so it is unscored.

**Verdict:** loses 0.002 to cand1. The stall rule doesn't replace the cap. In r01
both 4th attempts were "still climbing" by the prefix signal and still didn't pay. Keep R
from beta.
