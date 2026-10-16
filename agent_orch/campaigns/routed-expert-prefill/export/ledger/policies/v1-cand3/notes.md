# v1-cand3: two-step rounds after the first (W=4, R=2)

**Change (one behavior over v1):** once there is history, `plan()` returns W=4, R=2 (R=3 above
beta 0.7). The round opens four branches, refines each once, and stops. The next round re-roots
on the best node. Round 1 keeps the campaign defaults, as POLICY_DEV section 4 requires with no
history. v1's `select_batch` is unchanged. Its prune and collapse rules only act from a branch's
second attempt, so in a 2-step round they never fire.

**Evidence:**
- **cand2:** the flat stop ended r02 and r04 at step 2 and r03 at step 3. A fixed depth is the
  simpler way to get that.
- **Rebasing:** in every round, workers on trailing branches rebased onto the leader by hand
  (r01-r04 summaries). A re-root gives every worker the leader's code without a narrow step.

**Replay (r01-r04, beta 0.6):** V 1.5569 (+0.0058 vs v1, +0.0004 vs cand2, a tie with cand2 but
simpler).

| round | V | attempts | best |
|---|---|---|---|
| r01 | 1.4736 | 11 | 1.5011 (defaults, v1 rules) |
| r02 | 1.5314 | 8 | 1.5314 (gives up 1.5630) |
| r03 | 1.5952 | 8 | 1.5952 (gives up 1.6103, +0.95%, inside noise) |
| r04 | 1.6275 | 8 | 1.6275 kept |

**Beta sweep:** 1.5569 at beta 0.2-0.6 (8.75 attempts) and 1.5512 at beta 0.8-1.0 (R=3, 10.25
attempts, mean best 1.5706 vs 1.5638). Beta trades attempts for quality as intended.

**Verdict:** better than v1 by just over one attempt's cost. cand4 tests going one step shallower.
