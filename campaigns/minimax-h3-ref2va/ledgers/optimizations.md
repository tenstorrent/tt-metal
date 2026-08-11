# Optimizations — MiniMax-H3 `ref2va`

Only rows that passed the gate **and** measurably improved the metric. Short enough to read whole.

This is a **bringup** campaign, so the metric a row must move is a gate going from red to green, not
a latency. A row that merely adds code without a gate crossing does not belong here — it belongs in
`attempts.md`.

| r | sha | change | gate crossed | quality | artifacts |
|---|---|---|---|---|---|
| r1 | 5c8adce1e85 | host `packing_ref2va.py` + `test_packing_ref2va_minimax_h3.py` | gates 1–2 red → green | bit-exact (`torch.equal`) vs the installed reference | host-only, no artifacts |
| r3 | 5ec8933bbfa | typed `condition_blocks` in the transformer + the three ref2va cases | gate 4 red → green | existing PCCs bit-identical, not merely >= 0.9995 | artifacts/round-3/ |
| r5 | 5ec8933bbfa | audio-encoder mesh readback + `l1_small_size` 16384 for ref2va | gate 3 red → green | all three modalities >= 99.99% vs reference | artifacts/round-5/ |
| r6 | 4d04f289379 | ring fabric params for ref2va + `_denoise_and_decode` shared tail | gates 5-6 red → green | t2va CLIP 37.38 and all VBench dimensions unchanged | artifacts/round-6/ |
| r7 | db76ad3807e | fourth AdaLN level for ref2va audio conditioning | gate 7 red → green; audio-bearing e2e red → green | signal 0.128143 vs floor 0.000000 | artifacts/round-8/ |
| r8 | 440bd05c058 | per-axis seam bars (vertical 2.0, horizontal 3.0) | gates 6+8 red → green | seam ratios 1.315 / 2.287, no discontinuity in the boundary strip | artifacts/round-9,10/ |
| r9 | e10e6dda34e | ref2va quality bars derived from measurement | quality unbarred → 6 bars set | CLIP 25.0, subject 0.90, background 0.89, motion 0.97, dynamic 1.0, imaging 0.44 | artifacts/round-11/ |
