# v0-cand4: cand3 + stall close (third test of the stall rule)

**Change (one behavior over cand3):** also close a branch whose last two attempts are valid and
within `(0.5 + beta) x noise` of their parent. This is the r01/r02 dreaming's cand3 rule, re-tested
now that r03 exists.

**r03 evidence for:** b02-a02 (+0.4%) and b02-a03 (+0.9%) were both within noise. Closing b02
after a03 skips a04 (1.5717, -2.4%) and ends the round after step 3.

**Replay (r01-r03, beta 0.6):** V 1.5342, +0.0036 vs cand3. All of the gain is r03 (1.5936, 10
attempts, 3 steps); r01 and r02 are identical to cand3.

**Beta sweep:** 1.5315 / 1.5306 / 1.5342 / 1.5264 / 1.5264, so the gain shows up at beta 0.6 only.
- At beta 0.8 (band 1.3%), it closes r03's b02 after a02, because a01 was +1.21% over the root and
  a02 +0.4%. That loses r03's best, b02-a03 1.6103, and r03 V falls to 1.5702.
- This repeats the r02 dreaming's finding: the stall rule closes the leading branch, because
  within-noise steps are what the leader produces once a round has converged.

**Verdict:** a tie with cand3 (+0.0036, less than one attempt's cost), and fragile in beta.
Rejected in favor of the simpler cand3.
