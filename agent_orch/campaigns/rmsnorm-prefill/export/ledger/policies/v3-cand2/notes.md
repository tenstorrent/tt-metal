# v3-cand2: v3-cand1 + stop on a converged first step

**Change (one behavior on top of cand1):** after step 1, if all W first attempts are
valid, all beat the root by more than noise_pct, and all lie within noise_pct of each
other, stop and re-root (below beta 0.8). It works on scores alone. Unlike v2-cand2's
duplicate closure, it doesn't keep one survivor running.

**Trace evidence:** in r03 all four first attempts found the same row-0 SFPU stat
finalize (1.3129-1.3231, 0.78% spread). Refining all four in step 2 lifted the best only
+0.8% (1.3231 -> 1.3333, within noise), and the step cost 0.02 V. The other rounds'
first steps are spread out (r01 12%, r02 46%, r04 5.2%), so the rule doesn't fire there.

**Replay:** mean V 1.3058. r01 1.1942, r02 1.2585, r03 **1.3431** (4/1), r04 1.4275.
Sweep: 0.2 1.2775 / 0.4 1.3062 / 0.6 1.3058 / 0.8 1.2896 / 1.0 1.2878.

**Verdict: not promoted.** +0.0024 over cand1 is less than one attempt's cost (0.005),
so it's a tie and the simpler cand1 wins. It also has a cross-round cost that replay
can't see. The r03 steps 2-3 that this rule skips produced the L1 stats scratch
(r03-b02-a02, the r04 root) and streamed gamma (r03-b04-a03, ported in r04-b01-a01).
Both went into the 1.4475 stack in r04. Stopping r03 after step 1 would have re-rooted
r04 at 1.3231 without them. The rule fired in 1 of 4 rounds. Revisit only if more
converged rounds show their later steps adding nothing.
