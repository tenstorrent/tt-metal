# v1-cand5: v1 + history-derived depth cap (data-driven alternative to cand2)

**Change (one behavior):** plan stays at defaults. plan() stores the earlier
rounds' summaries. In select_batch a branch is closed once it has `cap`
attempts. `cap` is the deepest attempt index at which an earlier round's
running best (over all attempts up to that depth, across branches) rose by more than
noise_pct, plus depth_slack(beta) = round(2*beta) - 1 (0 at beta 0.6), and at least 2.
With no history there is no cap.

**Evidence:** the same as cand2, but learned from PlanContext history instead of
fixed by me. In r01 the running best rose at depth 1 (1.109) and depth 3
(1.2075). Depth 4 added only 0.47%, so cap = 3 for r02.

**Replay:** mean V 1.2089. r01 1.1882 (no history, so it behaves as v1). r02 1.2296.
Beta sweep 0.2: 1.2133 / 0.4-0.6: 1.2089 / 0.8: 1.2051 / 1.0: 1.2014.

**Verdict:** +0.0038 over v1, a tie. It loses to cand2 only because it can't act
in the first round. It is the more principled long-run form: it would undo
itself if a later round's deep attempts start paying. The cost is extra state (history
kept between plan and select_batch) and a plan R that doesn't match the effective
depth. Candidate to replace cand2's constant once there are 3 or more rounds of history.
