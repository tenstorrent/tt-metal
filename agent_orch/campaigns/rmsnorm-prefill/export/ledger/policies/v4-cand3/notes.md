# v4-cand3: v4 + lenient earned depth (one deeper than the last depth that paid)

**Change (one behavior, lenient variant of cand1):** in the previous round, find the deepest depth p whose
best valid attempt beat everything shallower (root included) by more than noise_pct. Cap this round's
depth at max(2, p + 1). This is "grow one past what paid", where cand1 is "shrink one below what didn't pay".

**Replay:** mean V 1.3484 (+0.0040 vs v4, -0.0027 vs cand1).
r04 1.3997 (cap 2: r03's deepest paying depth was 1), but r05 is unchanged at 1.5563. r04's depth 2 paid
(1.3666 -> 1.3997), so the cap is 3 = R, and the failing r05 a03 repairs still run.
Sweep: 0.2 1.3340 / 0.4 1.3484 / 0.6 1.3484 / 0.8 1.3334 / 1.0 1.3319.

**Verdict: not promoted.** It's weaker than cand1 and below one attempt's cost over v4. The r05 gain needs
cand1's signal: the deepest attempts specifically didn't pay.
