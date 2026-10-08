# v3-cand5: v3-cand1, but a lone branch that is still climbing may continue

**Change (one behavior on top of cand1):** the no-lone-step stop is skipped when the
lone branch's last attempt was a refinement that raised its anchor by more than
2*noise_pct. This hedges cand1's main risk: stopping a single branch that is climbing fast.

**Replay:** mean V 1.3034, identical to cand1 in every round and every beta. No
recorded lone step involved a climbing branch: r02's lone candidate was a first
attempt, and r03's was a regression (b04 1.3131 -> 1.2440).
Sweep: 0.2 1.2751 / 0.4 1.3038 / 0.6 1.3034 / 0.8 1.2896 / 1.0 1.2878.

**Verdict: not promoted.** It ties cand1 and is the bigger rule. Worth adopting if a
future round shows a lone climbing branch being stopped.
