<!-- BEGIN optimize -->
# Optimize (perf) — ``

_Updated live: 2026-07-28 12:37:22 UTC · 1 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — model · main (device_ms)
===============================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)

op                                 grid      fidelity  dtype     shard     host      tt-lang   cpp         best ms
------------------------------------------------------------------------------------------------------------------
M                                  ✓win      —         —         —         —         —         —            654.43


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                        lever        ms  gain vs base  result     why tried / why it won or failed
--------------------------------------------------------------------------------------------------------------------
M                                          grid    654.43   +1809.75 ms  ✓ win      committed: win

Limitations / suggested manual next steps:
- (none flagged automatically — see the per-op device report for remaining headroom.)

Reproduce:
  trace+1CQ perf:  python -m pytest t.py -svv

levels: grid -> fidelity -> dtype -> shard -> host -> tt-lang -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->
