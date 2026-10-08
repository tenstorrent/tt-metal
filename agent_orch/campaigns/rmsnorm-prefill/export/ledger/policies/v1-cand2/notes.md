# v1-cand2: v1 with a shallower round (R from beta) -- WINNER, promoted to v2

**Change (one behavior, plan only):** R = clamp(2 + round(2*beta), 2, default R),
which is 3 at beta 0.6 (W stays 4). Batch rule unchanged from v1 (open W
branches, refine every open branch, gap-close branches whose anchor trails the
leader by more than noise_pct * (2 + 8*beta)).

**Evidence (round summaries + v1 traces):** there are five recorded 4th
attempts. None lifted its round's best by more than noise_pct:
- r01: b01-a04 +1.5% over its parent, but the branch was not leading; b02-a04 -18.7%;
  b03-a04 +3.4%, still 6% behind the leader; b04-a04 +0.47%, the round best but
  within noise of b04-a03.
- r02: b02-a04 -0.4%.
By contrast, 3rd attempts carried the big repairs (r01 b01 +32%, b04 +44%), so
R=2 would be wrong. Rounds also re-root at the previous round's best (r02 was
rooted at r01-b04-a04), so a lineage that is still improving continues in the next
round with full width and a fresh root. A 4th attempt mostly buys depth the
next round gets anyway. R=3 is narrower than every recorded round, so it is in support.

**Replay:** mean V 1.2119 (v1 1.2051, +0.0068 > one attempt's cost).
- r01: 8 attempts, 3 steps, best 1.2075 (b04-a03), V 1.1942 (v1 1.1882).
- r02: 6 attempts, 3 steps, best 1.2396 (b02-a03), V 1.2296 (v1 1.2221).
Beta sweep: 0.2 (R=2) 1.1737, which loses the r01 repairs; 0.4-0.6 (R=3) 1.2119;
0.8-1.0 (R=4) 1.2051 / 1.2014. Beta trades attempts for depth as intended.

**Risk:** on r01 the round best drops from 1.2132 to 1.2075. The difference is
within noise, but the DST-accumulate idea (b04-a04) would then have to be found
again in the next round. A round with a slow-but-steady lineage (like r01-b03:
+11% / +3.4% / +3.4%) loses its last step inside the round. If a later round
shows 4th attempts that clearly lift the round best, raise beta or return to R=4.
