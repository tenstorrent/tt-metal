# v2-cand4: cand1 + no lone trailing step -- WINNER, promoted to v3

**Change (one behavior on top of cand1):** if the batch would be a single
refinement of a branch that doesn't hold the round lead, close that branch and
stop (below beta 0.8, where `lone_ok` is false). A lone leader still runs, and so does
any batch of 2 or more.

**Why (objective + traces):** V = best - cost*A + bonus*A/S. A one-item step
costs one attempt and also lowers the average batch width. In r03 under cand1
(8 attempts / 2 steps -> 9 / 3) the lone step costs 0.005 + 0.010 = 0.015, about
1.1% of the score. A trailing branch can only repay that by beating the leader by
more than 1.1%. In r03 under cand1, step 3 was exactly this: b04 alone,
repairing its -5.3% two-wave regression, with anchor 1.3131 below leader 1.3333.
The repair reached 1.3318, a good recovery of the branch that still didn't
lift the round. Rounds re-root at the best node, so a repair idea left unfinished here
(gamma streaming) is still available next round from a better root.
In r01 the repairs that produced the round best ran as a 2-item step (b01, b04)
and are untouched.

**Replay (rounds 1-3, beta 0.6):** mean V 1.2553 (v2 1.2457, cand1 1.2503).
- r01 1.1942: 8 attempts / 3 steps, best 1.2075 (unchanged from v2)
- r02 1.2385: 5 / 2; b02 stall-closed after -0.75%
- r03 1.3333: 8 / 2; b01/b02/b03 stall-closed, lone trailing b04 closed
Sweep: 0.2 1.2269 / 0.4-0.6 1.2553 / 0.8 1.2437 / 1.0 1.2412. Beta still trades
attempts for patience: from 0.8 the stall needs 2 refinements and lone steps are allowed.

**Risk:** a single remaining branch whose repair would have taken the lead
gets cut. r01-b04's repair (+12% over its anchor, a new round best) is the
kind of case that would be lost, if it ever ran alone. There is no recorded instance yet.
If one shows up, raise beta to 0.8 or allow lone repairs whose anchor is
within noise of the leader.
