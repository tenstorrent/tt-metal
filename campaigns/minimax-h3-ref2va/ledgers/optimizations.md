# Optimizations — MiniMax-H3 `ref2va`

Only rows that passed the gate **and** measurably improved the metric. Short enough to read whole.

This is a **bringup** campaign, so the metric a row must move is a gate going from red to green, not
a latency. A row that merely adds code without a gate crossing does not belong here — it belongs in
`attempts.md`.

| r | sha | change | gate crossed | quality | artifacts |
|---|---|---|---|---|---|
| r1 | PENDING | host `packing_ref2va.py` + `test_packing_ref2va_minimax_h3.py` | gates 1–2 red → green | bit-exact (`torch.equal`) vs the installed reference | host-only, no artifacts |
