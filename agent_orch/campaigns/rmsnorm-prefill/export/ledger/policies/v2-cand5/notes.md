# v2-cand5: v2 + step-1 duplicate closure only (isolation of cand2's rule)

**Change:** cand2's duplicate rule on plain v2, with no stall closure.

**Replay:** mean V 1.2490. r01 1.1942, r02 1.2296 (unchanged from v2), r03
1.3233 (6 attempts / 3 steps: b02 alone runs a3 = 1.3000).
Sweep: 0.2 1.2269 / 0.4-0.6 1.2490 / 0.8 1.2445 / 1.0 1.2398.

**Verdict:** +0.0033 over v2, a tie. On its own the duplicate rule only helps r03,
and without the stall rule the lone survivor wastes its third attempt. Not promoted.
