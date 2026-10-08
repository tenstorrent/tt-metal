# v1-cand4: v1 + final-attempt gate (adaptive alternative to R=3)

**Change (one behavior):** keep plan R=4. A branch about to take its final
(R-th) attempt is closed unless it currently holds the round lead and its last
attempt lifted its anchor by more than momentum(beta) * noise_pct
(momentum = 1 + 2*(0.6 - beta), 1x noise at beta 0.6).

**Evidence:** the same 4th-attempt record as cand2. The aim is to keep the one
case where depth might still pay: a leader that is still climbing.

**Replay:** mean V 1.2102 (v1 1.2051, cand2 1.2119).
- r01: the gate lets b04 (leader, +12% over its anchor at a03) take a04 and closes b01.
  9 attempts, 4 steps, best 1.2132, V 1.1907.
- r02: b02-a03 lifted its anchor by only 0.09%, so it is closed. 6 attempts, 3 steps, V 1.2296.
Beta sweep is flat (1.2102) for beta 0.2-0.8 and 1.2077 at 1.0, so the momentum bar barely
changes behavior on these rounds.

**Verdict:** it trails cand2 by 0.0017 (a tie) and is a rule rather than a
plan constant, so cand2 is simpler. This matches v0-cand4 (same idea, fitted
threshold), which was also a tie last time. Its advantage is that it keeps the
r01 best (1.2132). Revisit if R=3 is seen cutting a leader mid-climb.
