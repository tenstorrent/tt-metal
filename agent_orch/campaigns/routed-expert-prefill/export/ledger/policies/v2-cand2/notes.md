# v2-cand2: cand1 + above beta 0.7 the second step only repairs failed heads

**Change (one behavior over cand1):** above beta 0.7, v2 plans R=2 and refines every head in step 2.
cand2 keeps R=2, but step 2 only refines heads that failed. If nothing failed, the round stops after
step 1. The default beta 0.6 path is unchanged.

**Evidence:**
- **v2's high-beta step is a loss on r01-r07:** at beta 0.8-1.0, v2 spends 6.71 attempts per round
  for a mean best of 1.622. At beta 0.6 it spends 5.0 attempts for 1.621.
- **r07 summary:** both of r07's failures (b02-a01 hang, b03-a01 kernel-config overflow) were
  repaired inside the attempt before the scored run. Together with r06's repair of r05-b04-a01 from
  history, this means a failed idea gets repaired anyway, inside the attempt or in the next round. An
  in-round step on valid heads repeats what the re-root already delivers.

**Replay (r01-r07, beta 0.6):** V 1.6346, identical to cand1.

| beta | v2 V (attempts) | cand2 V (attempts) |
|---|---|---|
| 0.2-0.6 | 1.6342 (5.00) | 1.6346 (4.86) |
| 0.8-1.0 | 1.6266 (6.71) | 1.6314 (5.14) |

At high beta, r03's repair (b01-a02, 1.5882) is revealed but doesn't beat 1.5887. r05's hang repair
was never recorded, so it is out of support and reveals nothing.

**Verdict:** a tie at the default beta. It is a better high-beta knob than v2's, but cand4 is
better still.
