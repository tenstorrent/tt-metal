# Round 2 analysis (paper / accuracy, HEAD fd990b23418, 1002.0 ms)

No fresh profile: ttq refuses profile jobs while disk < 30 GB (27 GB at 2026-10-26 17:50). Sources are the newest
profiles, adjusted by measured deltas: iter_01/profile/sections (pre-R1-2f, estimate 1129.6 vs ~1103 measured,
+2.4 %), iter_01/r1_2f/variance/table.md (per-role R1-2f gains), r1_2fp (w_down 3.92 -> 1.82 ms/call x12),
iter_01/r1_2f/mv_profile/r12f (multivariate). Analyst: read-only Opus, low effort.

## Per-section estimate at HEAD (ms)
embed ~11 | layer_mlstm ~64 x6 = ~384 | slstm_pre ~110 | slstm_first ~3 | slstm_steps ~294 (552 steps, 0.547 ms/step,
27 ops/step) | slstm_post ~189 | head ~12. Matmul-class ~35 % of device time, elementwise / TM ~65 %.

## Top ops (full model)
1. sLSTM step elementwise (12 binary + 8 unary per step, fp32 DRAM [1024,512]): 191 ms (layers.py:449-482)
2. mLSTM chunkwise fp32 binaries (29/layer): 107 ms (layers.py:250-336)
3. mLSTM TM ops (tilize/untilize/slice/concat/transpose/cumsum): ~72 ms
4. slstm_pre/post permutes: 57 ms (layers.py:524,541)
5. slstm_post head_layer_norm averaging matmuls (fp32 HiFi4, 16.6 % FLOPs): 56 ms + 40 ms binaries (layers.py:499-516)
6. sLSTM recurrent matmul 1024x512x2048: 54.5 ms; R is block-diagonal by head, 3/4 of FLOPs wasted (layers.py:448)
7. mLSTM fp32 matmuls: 54 ms
8. FFN w_up minimal_matmul: 50 ms (59 % FLOPs, fine)
9. sLSTM step slices (Wx_t + 4 gates): 50 ms (layers.py:444,451)
10. w_in 29 ms, w_down 22 ms

## Gaps / host
Bench is traced; profile kernel sum over-predicts measured time, so dispatch gaps are ~0. The step loop is bound by
~15k small DRAM ops. Host: no wall figure since round 0 (4518 wall vs 3857 device); re-measure wall at HEAD.

Multivariate (3707 ms): variate permutes ~330 ms (~9 %); R1-2d costs +4.7 % there (D6 c).
