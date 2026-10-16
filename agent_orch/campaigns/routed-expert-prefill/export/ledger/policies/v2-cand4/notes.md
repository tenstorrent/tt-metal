# v2-cand4: cand1 + rounds after the first are one step deep at every beta

**Change (one behavior over cand1):** v2 plans R=2 with a full second step above beta 0.7. cand4
plans R=1 at every beta once there is history. Beta still sets round 1's patience (collapse, frontier
band, late step), so high beta stays "wider and more patient", just not deeper after round 1. The
default beta 0.6 path is unchanged.

**Evidence:**
- **v2 traces, beta 0.8-1.0:** the second step costs about 1.7 attempts per round across r01-r07 and
  raises the mean best by 0.001 (1.622 vs 1.621). The only gain is r03's b02-a02 (1.5952 vs
  1.5887, +0.4%, inside noise), at a cost of 4 attempts. r04's step 2 lowered every branch except b03.
- **cand2's repair-only step** is still revealed on r03 (b01-a02, 1.5882) and wins nothing.
- **r07:** failed first runs are now repaired inside the attempt (b02, b03), so the in-round repair
  step has less to do than when cand2 was written.

**Replay (r01-r07):**

| beta | v2 V (attempts) | cand2 | cand4 |
|---|---|---|---|
| 0.2-0.6 | 1.6342 (5.00) | 1.6346 (4.86) | 1.6346 (4.86) |
| 0.8-1.0 | 1.6266 (6.71) | 1.6314 (5.14) | 1.6342 (5.00) |

**Verdict:** a tie at the default beta (+0.0004). It is the best high-beta knob of the phase and the
simplest code path after round 1. Promote it only if the orchestrator raises the default beta above
0.7.
