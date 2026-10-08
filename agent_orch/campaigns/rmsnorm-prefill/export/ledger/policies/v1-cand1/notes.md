# v1-cand1: v1 + plateau closure

**Change (one behavior):** besides v1's gap-to-leader close, a branch is closed
when its last k attempts (k = 1 + round(2*beta), 2 at beta 0.6) were all valid,
none was a clear regression (more than noise_pct below the prior anchor, which
signals a repair in progress), and together they lifted the branch anchor by
no more than noise_pct. Plan unchanged (W=4, R=4).

**Trace evidence (v1 on r02):** after the prune, b02 ran 1.2385 -> 1.2292
(-0.75%) -> 1.2396 (+0.09% over anchor) -> 1.2346. Two attempts in a row moved
the anchor by less than noise, and the 4th attempt (b02-a04) then added nothing.
k=2 rather than 1 because of r01-b02: its first attempt was within noise of the
root (+0.25%) and its next attempt was a clear +3.1% gain. One within-noise
result is not enough evidence.

**Replay:** mean V 1.2089 (v1 1.2051). r01 1.1882 (unchanged, the rule never
fires: the only within-noise result, b04-a04, is a last attempt). r02 1.2296
(6 attempts, 3 steps; b02-a04 saved).
Beta sweep: 0.2 (k=1) 1.2133 / 0.4-0.6 1.2089 / 0.8 1.2051 / 1.0 1.2014.

**Verdict:** +0.0038 on the mean is less than one attempt's cost (0.005): a tie
with v1. Not promoted on its own. It is still a sensible guard once rounds
contain longer plateaus.
