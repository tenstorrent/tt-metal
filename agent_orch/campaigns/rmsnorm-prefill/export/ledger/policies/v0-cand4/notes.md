# v0-cand4: cand1 + final-attempt focus

**Change:** on top of cand1, a non-leading branch whose next attempt would be
its last (R-1 attempts done) is closed unless its anchor is within
`noise_pct * 2*beta` (1.2% at beta 0.6) of the leader.

**Trace evidence:** the fourth attempts on r01 gained little (+1.7%, +0.6%,
+3.4%, -18.7% vs parent). Before step 4, b01 (1.187) trailed b04 (1.2075) by 1.7%.
Its last attempt (1.204) didn't take the lead.

**Replay (r01):** V = 1.1907, 9 attempts, best 1.2132. +0.0025 over cand1.

**Verdict:** that is less than one attempt's cost (0.005), so it's a tie under the
small-data rule. The threshold is also within a fraction of a percent of the
one gap that decides it (1.7%), so it is fitted to this round. Kept the simpler
cand1.
