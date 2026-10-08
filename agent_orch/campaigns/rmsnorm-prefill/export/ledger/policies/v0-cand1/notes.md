# v0-cand1: gap-to-leader pruning

**Change (one behavior):** close an open branch whose anchor (best valid score
on it) trails the round's best by more than `noise_pct * (2 + 8*beta)`
(6.8% at beta 0.6). The rule compares anchors, not heads, so a dip never closes a
branch by itself. A branch with no valid result is never gap-closed.

**Trace evidence (r01, v0):** after step 1, b02 (1.003) and b03 (0.987) trailed
the leader b01 (1.109) by 9.6% and 11%. v0 refined them 3 more times each
(6 attempts, 0.03 V). Their best nodes (1.172, 1.165) never came close to the
round best (1.213). b04 trailed by only 3.0% and is kept. It produced the round best.
The deep dips on b01/b04 (0.90, 0.84) don't close anything because the anchor stays at
the step-1 score. The round summary says these repairs produced the two best
nodes.

**Replay (r01):** V = 1.1882 (v0 1.1732), 10 attempts, 4 steps, best 1.2132.
Beta sweep: 0.2-0.8 -> 10 attempts; 1.0 -> 13 attempts (keeps b02), V 1.1807.
So beta does trade attempts for patience.

**Risk:** a recovering branch that would later have become the leader gets cut. In r01,
b03 recovered to 1.165 but stayed below the leader line.
