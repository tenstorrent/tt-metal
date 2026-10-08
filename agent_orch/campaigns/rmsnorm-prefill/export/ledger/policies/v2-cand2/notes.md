# v2-cand2: cand1 + step-1 duplicate closure

**Change (one behavior on top of cand1):** right after step 1, close a
non-leading branch if its first attempt is within dup_band of the leading first
attempt (noise_pct * max(0, 1.6 - beta), 1.0x noise at 0.6) and shares at least
half of the leader's tags (and at least 2). Such a branch is treated as a copy
of the leader's idea.

**Trace evidence:** step-1 convergence showed up in 2 of 3 rounds. In r01, b01 and b04
both put x*gamma under the AG wait (tags compute/overlap/post-phase), 3% apart,
so the band doesn't fire. In r03 all four first attempts did the same row-0 SFPU
stat combine (1.3129 / 1.3231 / 1.3175 / 1.3131, all within 0.8%, sharing
compute/post-phase/sfpu). They kept converging at step 2: b01-a02 and b02-a02
both moved the AG stats scratch to L1.

**Replay:** mean V 1.2553. r01 1.1942, r02 1.2385, r03 1.3333 (5 attempts /
2 steps: only b02 continues, reaches 1.3333, then stall-closes).
Sweep: 0.2 1.2269 / 0.4-0.6 1.2553 / 0.8 1.2470 / 1.0 1.2423.

**Verdict:** ties cand4 exactly at beta 0.6. It is not promoted: it relies on a
tag-overlap heuristic, it is the bigger rule, and it cuts 3 of 4 branches
after one attempt. The r02 summary already flagged that kind of capacity loss.
It also won r03 partly by luck: the kept branch happened to be the one whose next
attempt was best, and b04's divergent third attempt (gamma streaming, 1.3318)
would be lost. Worth revisiting if more rounds show step-1 near-ties.
