# v4-cand5: v4-cand1 + close a branch whose head broke a campaign rule

**Change (two behaviors):** cand1's earned depth plus cand2's forbidden_edit closure.

**Replay:** mean V 1.3511, identical to cand1 at the default beta (r04 stops at depth 2 before the
forbidden head matters). At beta 0.8/1.0 (no cap) it gains cand2's r04 attempt: 1.3338 / 1.3323 vs
1.3334 / 1.3319.
Sweep: 0.2 1.3340 / 0.4 1.3511 / 0.6 1.3511 / 0.8 1.3338 / 1.0 1.3323.

**Verdict: not promoted (tie, bigger rule).** Worth adopting if a forbidden_edit head shows up again
in a round now that verify_node enforces forbidden_patterns online.
