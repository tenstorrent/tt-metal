# v0-cand5: cand3 + frontier prune also closes failed heads (ablation of the failure exemption)

**Change (one behavior over cand3):** the frontier rule also closes a branch whose head failed,
once the branch has two attempts.

**Evidence wanted:** r03's b03-a03 accuracy_fail was followed by a04 at 1.5554, below the root.
That repair was wasted.

**Replay (r01-r03, beta 0.6):** V 1.5250 (-0.0056 vs cand3).
- **r01:** b04 is closed after a02's accuracy_fail. Its repair a03 (1.4622) and r01's best a04
  (1.5011) are never revealed. The best falls to 1.4716, with 9 attempts and 3 steps.
- **r02:** unchanged from cand3.
- **r03:** unchanged from cand3. b03 had already been closed by the frontier rule at step 2, so the
  exemption never mattered there.

Across the three rounds, the failures that were repaired landed near the frontier in 2 of 3 cases
(r01 b04-a03 -0.6%, r03 b01-a02 -0.03%).

**Side check (not saved):** cand3 plus a plateau stop after two flat steps in a row gives V 1.5313,
a tie with cand3. It stops r02 before step 4 (best 1.5435) and r03 after step 3. The evidence on
the plateau stop is still split one round each.

**Verdict:** reject. The failure exemption is load-bearing. Keep failures open, as POLICY_DEV
section 3 says.
