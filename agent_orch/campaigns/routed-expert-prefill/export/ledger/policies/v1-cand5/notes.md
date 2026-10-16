# v1-cand5: cand4 + one repair step for failed first attempts

**Change (one behavior over cand4):** rounds after the first plan R=2 (R=3 above beta 0.7), but the
extra step only refines heads that failed. Build, runtime and accuracy failures are repairable.
Valid heads end with the round. If nothing failed, the round stops after step 1, as in cand4.

**Evidence:** the main online risk of cand4 is losing in-place repairs. r01-b04 (accuracy_fail →
1.4622 → 1.5011, r01's best) and r03-b01 (runtime_error → 1.5882) were both repaired one attempt
later.

**Replay (r01-r04, beta 0.6):** V 1.5653 (-0.0050 vs cand4, exactly one attempt's cost).

| round | V | attempts | what changed vs cand4 |
|---|---|---|---|
| r01 | 1.4736 | 11 | none (defaults) |
| r02 | 1.5514 | 4 | none (no failure) |
| r03 | 1.5887 | 5 | b01-a02 repair revealed at 1.5882, below the best; a one-item step dilutes the parallel bonus |
| r04 | 1.6475 | 4 | none (no failure) |

**Verdict:** reject under the objective. A one-worker repair step costs its attempt plus half the
round's parallel bonus. The next round's step 1, which sees the failed node in history, is the
cheaper repair. Revisit this if a round records a failed first attempt whose repair would have led
the round.
