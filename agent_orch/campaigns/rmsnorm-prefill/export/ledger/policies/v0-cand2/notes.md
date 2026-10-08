# v0-cand2: no-signal pruning (root-relative)

**Change:** an alternative to cand1's single rule. Close a branch whose anchor hasn't beaten
the round root by more than `noise_pct * (2 - 2*beta)` (0.8% at beta 0.6).
Each branch is judged only on its own lift, not against other branches.

**Trace evidence:** the same as cand1. b02 (+0.25%) and b03 (-1.3%) showed no signal on
their first attempt, while b01/b04 were at +10.9% and +7.6%.

**Replay (r01):** V = 1.1882, the same trajectory as cand1 (10 attempts). Beta 1.0 -> 13
attempts (keeps b02, whose anchor is above the root).

**Verdict:** a tie with cand1. I kept cand1 because the round-2 root will be the r01
best. Small lifts will be the norm there, so a root-relative bar would cut
modest-but-real ideas. The leader-relative gap only cuts clear losers.
