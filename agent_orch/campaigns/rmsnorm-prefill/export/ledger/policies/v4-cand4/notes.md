# v4-cand4: v4 + aggressive earned depth (cap at the last depth that paid, floor 1)

**Change (one behavior, aggressive variant of cand1):** with p = the previous round's deepest paying
depth, if deeper attempts were recorded and didn't pay, cap this round at max(1, p). A cap of 1
means: open W branches, then stop and re-root.

**Replay:** mean V 1.3504 (+0.0060 vs v4, -0.0007 vs cand1).
r03 **1.3431** (cap 1 from r02, stops after the converged step 1, like v3-cand2), r04 **1.3866**
(cap 1 from r03: loses the depth-2 stacks, 1.3666 instead of 1.3997), r05 1.5696 (cap 2 from r04).
Sweep: 0.2 1.3333 / 0.4 1.3504 / 0.6 1.3504 / 0.8 1.3334 / 1.0 1.3319.

**Verdict: not promoted.** It ties cand1 (within one attempt's cost), and it shows that a floor of 1 is too
aggressive: in r04 the in-round depth-2 stacking (b03-a02, b04-a02) was worth +2.4%. It also skips the r03
steps that produced the L1 stats scratch and streamed gamma (the v3-cand2 cross-round concern).
