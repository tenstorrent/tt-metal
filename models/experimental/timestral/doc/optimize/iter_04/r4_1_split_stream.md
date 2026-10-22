# R4-1: context / decoder split streams (accepted), with R4-3 folded in

At the paper shape each row holds 64 context tokens and 2 decoder tokens, padded to 96. Every per-token op
then ran on 98304 rows, 31 % of them pad.

Candidate: keep the context tokens, `(1, 1, R * L_pad, D)` with L_pad = 64, and the decoder tokens,
`(1, 1, R * h, D)`, as two flat streams. Every op except time attention runs per token on each stream.
Time attention:

- Context queries attend causally on the `(H, R, L_pad, Dh)` view (masked: the context block of the time mask).
- Decoder queries of b = 32 / h rows (16 at the paper shape) fill one tile. They attend over the
  `ttnn.concat` of those rows' context keys and their decoder keys, `(H, R / b, b * (L_pad + h), Dh)`.
  A `(1, 1 or R / b, b * h, b * (L_pad + h))` block mask keeps each query on its own row. The all-valid mask is
  one cached block broadcast over every block; masked inputs upload one per block.
- Last layer (R4-3): the context stream ends there, so it computes only keys and values (`[Wk | Wv]`,
  2-chunk split). The context queries, SDPA, gate, wo, variate sublayer and MLP all drop.
- The decoder stream starts as the forecast query (a cached device tensor, cloned per forward).

The split applies when h divides 32, R is a multiple of 32 / h, and L_pad + h < pad(L + h). Otherwise, as at
long_context (1024 + 32 = 1056 already) and in the small PCC shapes, the single stream runs unchanged.

Micro-benchmark gate (paper shape, host-timed, `bench_split_stream.py`), ms:

| op | ms |
|---|---|
| concat K (12, 64, 1024 + 32, 64) | 0.62 (x 2 with V) |
| decoder SDPA, q (12, 64, 32, 64), mask (1, 64, 32, 1056) | 0.82 to 0.99 |
| context SDPA (12, 1024, 64, 64) causal | 1.16 (vs 1.48 for the padded (12, 1024, 96, 64)) |

Decoder attention costs about 2.1 ms per layer against a gate of 2 ms; device time runs below host time.
Its accuracy against fp32 (PCC 0.999764, mean error 3.4e-3) matches the padded path's decoder positions
(PCC 0.999761).

New tests: `test_split_stream_blocks_pcc` (univariate and multivariate, causal and masked; every block plus
the last block cut to the decoder stream), and `test_predict_block_tiny_pcc[split_*]`. The pretrained
horizon-2 case now packs 16 rows, so it takes the split path.

Benchmark (`candidates/r4_1_split_stream.json` vs `iter_03/bench.json`):

| shape | wall ms | device ms | PCC (norm) |
|---|---|---|---|
| paper | 540.7 → 427.3 (−21.0 %) | 470.2 → 368.0 (−21.7 %) | 0.99994 |
| paper, performance | 501.4 → 403.3 (−19.6 %) | 453.8 → 355.4 (−21.7 %) | 0.99970 |
| long_context (single stream) | 221.8 → 218.9 | 196.3 → 196.3 | 0.99976 |
| multivariate | 813.9 → 618.0 (−24.1 %) | 756.2 → 572.2 (−24.3 %) | 0.99994 |
