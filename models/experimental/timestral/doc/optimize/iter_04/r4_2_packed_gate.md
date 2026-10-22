# R4-2: wg packed into the split projections (rejected on the single stream, accepted on the split streams)

## Re-measured on the split streams (accepted)

Same packing on top of R4-1 and R4-4 to R4-6: `[Wq | Wk | Wv | Wg]` on both streams of every time attention,
`[Wv | Wg]` on the univariate variate path, the sigmoid on the gate MUL's input b. The last layer's context
stream still computes only `[Wk | Wv]`, so the last-layer cost that sank it on the single stream is gone.

Benchmark (`candidates/r4_2b_packed_gate_split.json` vs `candidates/r4_6_cached_constants.json`): paper device
362.8 → 354.2 ms (−2.4 %), wall 412.4 → 408.6 ms; performance 349.8 → 338.1 ms (−3.3 %), long_context
194.1 → 188.5 ms (−2.9 %), multivariate 567.4 → 560.6 ms (−1.2 %). PCC 0.99996.

## Single stream (rejected)

Candidate: time attention `minimal_matmul_split(xn, [Wq | Wk | Wv | Wg], chunks=4)` (M16 / K8 / N6, subblock 4x2),
univariate V-path `[Wv | Wg]` with 2 chunks (M16 / K4 / N6, subblock 8x1), and the sigmoid applied to the
gate MUL's second input (`input_tensor_b_activations`).

Host-timed sweeps (`qkvg_split_sweep_accuracy.md`, `vg_split_sweep_accuracy.md`) predicted −0.67 ms per
projection: 3.71 vs 4.38 ms and 1.90 vs 2.58 ms. The baselines there are two ops each, so they carry about
0.4 ms of host overhead that the device does not see.

Device profile (reduced layers, paper shape, accuracy), µs per op:

| op | round 3 | R4-2 |
|---|---|---|
| time layer: QKV split 3 + wg/sigmoid 2D | 2616 + 1360 | — |
| time layer: QKVG split 4 | — | 3629 |
| time layer: gate MUL | 1007 | 1134 (sigmoid on input b) |
| time layer total | 18 340 | 18 070 |
| time_variate layer total | 23 780 | 23 030 |
| last layer total | 10 160 | 11 120 (the gate now runs on all 98304 rows before the cut) |

Gate MUL at the paper shape (98304 x 768, `bench_gate_mul`), ms:

| variant | ms | mean abs error vs fp32 |
|---|---|---|
| plain MUL | 1.086 | — |
| sigmoid on input b, exact | 1.223 | 7.2e-4 |
| sigmoid on input b, approximate (`UnaryWithParam(SIGMOID, 4, 1)`) | 1.109 | 2.5e-2 (max 0.58): unusable |

Benchmark (`candidates/r4_2_packed_gate.json` vs `iter_03/bench.json`): paper device 470.2 → 462.4 ms
(−1.7 %), wall 540.7 → 519.2 ms (−4.0 %, mostly host noise), every guard faster, PCC 0.99993.
The saving is real but below the 2 % device gate: the packing saves 0.35 ms per time layer, and the
sigmoid moving into the MUL gives 0.13 ms of it back.
