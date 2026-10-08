# v1-cand3: cand2 (R from beta) + cand1 (plateau closure)

**Change:** both behaviors together: R = 2 + round(2*beta) (3 at 0.6) and the
plateau close (k = 1 + round(2*beta), anchor lift within noise_pct, no clear
regression in the window).

**Evidence:** the same as cand1 and cand2. This checks whether the two savings stack.

**Replay:** mean V 1.2119, identical to cand2 at every beta (r01 1.1942, r02
1.2296). With R=3 the plateau rule can only fire before a 3rd attempt, which needs k=1
(beta <= 0.2), and at beta 0.2 R=2 already ends the branch first.

**Verdict:** an exact tie with cand2 with one more rule. Kept cand2. The plateau rule
becomes worth scoring again if R goes back to 4 or rounds get deeper.
