# R4-4, R4-5, R4-6: small items, kept as a bundle below the 2 % device gate

Each was measured on its own against the previous candidate (paper, accuracy; `candidates/*.json`), on top
of R4-1. Together they take paper device time from 368.0 to 362.8 ms (−1.4 %) and upload from 8.7 to 5.6 ms.
That is below the 2 % device gate, so they are kept as a bundle (a deviation from the per-candidate gate)
because every shape got faster or stayed the same and PCC held or improved.

## R4-4: matmul block / subblock search (`tests/perf/sweep_subblocks.py`)

The sweep ran at the split stream's M = 65536 (`subblock_sweep_accuracy.md`); the picks were confirmed at
7 x 10 calls (`subblock_confirm_*.md`), including the 2048-row decoder stream:

| op (65536 rows) | shipped | pick | ms |
|---|---|---|---|
| wo + residual (addcmul) | 8 / 8 / 8, 1 x 8 | 16 / 8 / 4, 4 x 1 | 0.884 → 0.802 |
| w_down + residual | 8 / 8 / 8, 1 x 8 | 16 / 8 / 4, 4 x 1 | 1.374 → 1.349 |
| wv | 8 / 8 / 8, 1 x 8 | 16 / 8 / 4, 4 x 1 | 0.743 → 0.725 |
| patch embedding out | 8 / 8 / 8, 1 x 8 | 16 / 8 / 4, 4 x 1 | 1.676 → 1.647 |
| SwiGLU | 8 / 12 / 2, 4 x 2 | 8 / 12 / 2, 2 x 2 | 2.552 → 2.511 |

The split projections are already at their best (QKV 2 x 4, [Wk | Wv] 8 x 1). The 8 x 1 QKV win from round 3
(at 98304 rows) does not reproduce at 65536 rows (+2.2 %). On the decoder stream the new minimal config is
neutral (wo 0.063 → 0.056, w_down 0.075 → 0.085, wv 0.054 → 0.056 ms; host-timed, tiny ops).

Benchmark vs R4-1: paper device 368.0 → 363.7 ms (−1.2 %); performance −1.4 %, long_context −0.9 %,
multivariate −0.6 %. PCC unchanged (0.99994).

## R4-5: patch embedding out + skip in one matmul

`minimal_matmul` reads a K-concatenated input pair, so `out(silu(hidden(x))) + residual(x)` becomes one
`[silu(hidden(x)) | x] @ [W_out; W_res] + (b_out + b_res)`. That removes the skip matmul and the ADD.
The round-3 MUL keep / ADD query pair is already gone on the split path (the decoder stream starts from the
forecast query). At the paper shape, host-timed: out + skip + add 2.78 ms → 1.74 ms. The hidden matmul stays
on the 2D config: `minimal_matmul` with a fused silu is slower (3.05 vs 2.06 ms).

Benchmark vs R4-4: paper device 363.7 → 363.2 ms (−0.1 %), guards −0.2 to −0.3 %. PCC 0.99994 → 0.99996,
since main and skip no longer round to bf16 before the add.

## R4-6: per-shape device constants cached

`unit_stats`, the shared cos / sin tables (keyed by their positions), the causal cross mask, the forecast
query rows and the single-stream token layout are built once per key in a 32-entry LRU. One call uses at most
7, so a call never evicts its own. Per-row RoPE tables and the masks of missing values stay per call.

Benchmark vs R4-5: upload 8.7 → 5.6 ms, cold call 1.26 → 0.74 s, device unchanged (363.2 → 362.8 ms); wall is
within host noise (410.7 → 412.4 ms).
