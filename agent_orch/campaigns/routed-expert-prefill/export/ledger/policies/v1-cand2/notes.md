# v1-cand2: v1 + flat-step stop

**Change (one behavior over v1):** from step 2 on, stop the round when the last step's running
best gained no more than 1x noise. The exception is when an open branch's head is a valid repair
of a failed attempt: that branch is still improving, and failures are repairable.

**Evidence:**
- **r03 and r04 summaries:** no step after step 1 set a gain outside noise. r03's later bests were
  +0.4% and +0.95%. r04's step 2 set no new best, and step 3 ran with one worker for 13 minutes.
- **v1 traces:** v1's tail steps are narrow (1-2 items), so each costs its attempts and also dilutes
  the parallel bonus.

**Replay (r01-r04, beta 0.6):** V 1.5565 (+0.0054 vs v1).

| round | V | vs v1 | attempts | what happens |
|---|---|---|---|---|
| r01 | 1.4736 | 0 | 11 | Step 2 gained +2.4%. At step 4, b04-a03 is the repair of b04-a02's accuracy_fail, so the stop is held off and b04-a04 (1.5011, the round's best) is revealed |
| r02 | 1.5314 | -0.0041 | 8 | stops after step 2; loses the late +2.5% (b03-a04 1.5630) |
| r03 | 1.5936 | +0.0108 | 10 | held at step 3 by b01's repair; stops after step 3 with 1.6103 kept |
| r04 | 1.6275 | +0.0150 | 8 | stops after step 2 |

- **Ablation (not saved):** without the repair exemption, V falls to 1.5522. r01 stops after step 3
  and loses its best. The exemption is load-bearing, as in v0-cand5.
- **Beta sweep:** 1.5565 at beta 0.2-0.6 and 1.5561 at beta 0.8-1.0. Beta barely matters here.

**Verdict:** an improvement of about one attempt's cost. It is superseded by cand3, which reaches
the same V with a plan-level cap instead of a stop rule.
