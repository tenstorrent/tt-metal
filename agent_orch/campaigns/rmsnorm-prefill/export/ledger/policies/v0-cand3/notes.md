# v0-cand3: cand1 + patience before gap closure

**Change:** cand1's gap rule only fires once a branch has at least
`1 + round(2*beta)` attempts (2 at beta 0.6). It guards against shallow
judgment on one weak first idea.

**Trace evidence:** b03 went 0.987 -> 1.094 -> 1.131 -> 1.165. A weak first attempt
recovered, which is the "shallow judgment" pattern from POLICY_DEV.md.

**Replay (r01):** V = 1.1732 at beta 0.6 (16 attempts: after its second attempt,
b02's 1.034 sits exactly at the 6.8% gap and b03's 1.094 is inside it, so
nothing closes). Beta 0.2 -> 10 attempts / 1.1882; 0.4 -> 13 / 1.1807.

**Verdict:** loses 0.015 to cand1 on r01. Recovery on b03 was real, but it never
threatened the leader, so in this round patience only bought attempts. Revisit
if a later round shows a late-recovering branch that overtakes.
