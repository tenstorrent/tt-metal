# v3-cand4: v3-cand1 with a tighter gap-to-leader closure

**Change (one behavior on top of cand1):** gap = noise_pct * (2 + 6*beta) (5.6% at
beta 0.6) instead of (2 + 8*beta) (6.8%).

**Trace evidence:** in r04 b02 trailed the leader by 5.9% after step 2 (1.3175 vs
1.3997), and its step-3 attempt (1.4216) didn't take the round best.

**Replay:** mean V 1.3038 (+0.0004 vs cand1). Only r04 changes: 11 attempts, V 1.4292.
Sweep: 0.2 1.2751 / 0.4 1.3042 / 0.6 1.3038 / 0.8 1.2896 / 1.0 1.2896.

**Verdict: not promoted.** It's a tie. The closed branch was also the clearest
counter-example in r04: b02 recovered +7.9% by porting the round's writer stack from
sibling branches (r04 summary: workers port ideas across branches inside a round).
That makes a trailing anchor a weaker sign of a dead branch than before.
