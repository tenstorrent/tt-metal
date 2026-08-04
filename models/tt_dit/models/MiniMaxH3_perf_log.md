# MiniMax-H3 transformer block performance log

Append-only. One entry per measured change, newest at the bottom. Never rewrite an entry: the point
is to be able to see what a change actually bought, including the ones that bought little.

All numbers are **device time for one transformer block**, from
`test_transformer_block_perf_minimax_h3.py` under `--profile`, analysed with
`project_block_perf.py` (which reproduces `tt-perf-report`'s totals exactly). 768P (768x1344) on
4x8 Blackhole Galaxy, ring, 2 links, TP=4 / SP=8. Per-device sequence lengths 4768 / 9216 / 13632.

Per denoise step = block x 50 layers. Per video = step x 50 denoise steps. Both cover the block stack
only -- refiner, input projections, `norm_out` and the output heads are excluded, a few percent on top.
Device time is a lower bound on wall clock; see the `device + op gap` note in `project_block_perf.py`.

Reproduce:

```
scripts/run_safe_pytest.sh --profile \
    models/tt_dit/tests/models/minimax_h3/test_transformer_block_perf_minimax_h3.py -k 5s_768p
python models/tt_dit/tests/models/minimax_h3/project_block_perf.py 5s=<csv>
```

| # | date | change | 5s | 10s | 15s | commit |
|---|------|--------|----|-----|-----|--------|
| 1 | 2026-08-03 | baseline: block + attention as first brought up | 26.03 ms | 57.65 ms | 98.72 ms | `12c0e27c922` |
| 2 | 2026-08-03 | addcmul gates + adaLN scale/shift folded into the fused norm | 25.61 ms | 57.19 ms | 97.63 ms | `1739353a695` |
| 3 | 2026-08-04 | ring SDPA chunk sizes tuned from a measured sweep | 24.80 ms | 57.19 ms | 97.63 ms | `acfc6885b12` |
| 4 | 2026-08-04 | QK-norm + head split + RoPE fused into one op | 18.53 ms | not measured | not measured | `9c9f54d877d` |
| 5 | 2026-08-04 | all-gather folded into the matmuls (AGMM), placeholder block sizes | 18.20 ms | 43.22 ms | 78.26 ms | (pending) |

Cumulative at 5s: **26.03 -> 18.20 ms, -30.1%**; 1.30 -> 0.91 s per 50-layer step, 65.1 -> 45.5 s per video.
At 10s: 57.65 -> 43.22 ms (-25.0%). At 15s: 98.72 -> 78.26 ms (-20.7%).

## Notes per entry

**1. Baseline.** Unfused all-gather + matmul; rotary embedding decomposed into slices, permutes and
concats because the 48-channel half boundary is not tile-aligned. 100 device ops.

**2. addcmul + adaLN fusion.** Removed 6 elementwise ops (BinaryNg 16 -> 10 calls). Only ~1%: these
were never expensive ops, and the norm absorbed 0.08-0.30 ms of the work. No dispatch-overhead claim
is made -- the gap-inclusive totals moved +3.5% / -2.5% / -13.2%, no consistent direction.

**3. SDPA chunk sizes.** 5s moved to (320, 384) -- SDPA 8.43 -> 7.56 ms, FLOP util 50.8% -> 56.7%.
10s and 15s were already optimal at (256, 512). The winner needs k=384, which is neither a power of
two nor in the {256, 512} set an initial search covers; L1 rejects (320, 512) but accepts (320, 384).

**4. Fused RoPE.** The largest win so far. The whole 36-op rotary tail collapses into the two QK-norm
ops; 96 -> 60 device ops. Works because MiniMax-H3's half-split rotary is the fused op's interleaved
rotary under a permutation of Q/K channels, and partial rotary needs only cos=1 / sin=0 padding.

10s and 15s are recorded as not measured rather than guessed: those two profiled runs failed -- the
perf test was still passing raw rotary_dim-wide tables and the new width check rejected them -- and
`--profile` masks pytest's exit code, so they reported PASS with a CSV holding only the ops that ran
before the exception. What caught it was the numbers being impossible: 15s "faster" than 5s. Check the
pytest summary line, not the exit code, after any profiled run. Entry 5's 10s/15s figures therefore
cover entries 4 and 5 together.

**5. AGMM.** Folding the TP all-gather into the matmul that consumes it. Only -1.8% at 5s, because
the block sizes are legality-driven placeholders, not swept: `K_block` must divide
`K_tiles_per_device` and `N_block` must be even, and the generic 8x8x8 default satisfies neither. The
sweep is the follow-up; these numbers are the floor, not the result.

## Standing opportunities, from the 5s profile after entry 4

| target | share of block | note |
|--------|----------------|------|
| ring SDPA | ~41% | already chunk-tuned; further needs kernel work |
| FF matmuls + CCL | ~21% | three matmuls flagged SLOW at 57-65% FLOP util |
| all-gathers / reduce-scatter | ~10% | unfused TP path, folds into `all_gather_minimal_matmul_async` |
| adaLN table build + 6 gathers | ~13% | `adaln_proj` runs at 2.4% FLOP util (M=32, 2 real rows) |
