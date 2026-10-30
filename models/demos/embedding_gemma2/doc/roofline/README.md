# Practical roofline: embeddinggemma-2 prefill, P150, 12 x 10, batch 1

Practical roofline = sum over op families of max(compute ideal / 0.5, memory ideal / 0.8). Measured = sustained ms
per traced forward at 12 x 10 (`../prefill/04-attention/README.md`, `../prefill_towers/02-attention/README.md`).
Precision `../datatype_sweep/selected_precision_config.json` (text `sdpa_nofp32acc`; vision and audio sections of
the same file). Rerun: `python3 roofline.py` (CPU only; `--families all` prints every shape's op-family table).

## Per shape

| Shape | Practical roofline ms | Measured ms | % of practical (practical / measured) | Dominant bound (share of practical) |
|---|---|---|---|---|
| text 128 | 1.047 | 4.837 | 22 % | MEMORY (100 %) |
| text 256 | 1.740 | 5.512 | 32 % | MEMORY (100 %) |
| text 512 | 3.125 | 7.130 | 44 % | MEMORY (100 %) |
| text 1024 | 6.088 | 11.279 | 54 % | MEMORY (100 %) |
| text 2048 | 12.374 | 19.235 | 64 % | MEMORY (86 %) |
| text 4096 | 26.188 | 39.701 | 66 % | MEMORY (80 %) |
| text 8192 | 58.787 | 90.479 | 65 % | MEMORY (70 %) |
| text 16384 | 143.869 | 207.514 | 69 % | MEMORY (57 %) |
| text 32768 | 393.709 | 526.523 | 75 % | COMPUTE (58 %) |
| text 65536 | 1211.591 | 1577.429 | 77 % | COMPUTE (73 %) |
| vision 1x2560 | 12.276 | 15.686 | 78 % | MEMORY (84 %) |
| audio 1x256 | 2.240 | 7.267 | 31 % | MEMORY (100 %) |
| audio 1x768 | 5.700 | 11.141 | 51 % | MEMORY (100 %) |

## Hardware numbers

| Hardware number | Value | Source |
|---|---|---|
| Active cores | 120 (12 x 10) | Galaxy target grid (brief) |
| AICLK | 1.35 GHz | P150 nominal; roofline session `roofline.py` CLK_HZ |
| Matmul per core | 4096 FLOP / cycle at LoFi, / fidelity passes (HiFi2 2, HiFi4 4) | roofline session `roofline.py` FLOP_PER_CYCLE_LOFI; `tech_reports/GEMM_FLOPS` (2 x 8 x 16 x 16) |
| Peak compute | LoFi 663.6, HiFi2 331.8, HiFi4 165.9 TFLOP/s | cores x clock x FLOP / cycle |
| DRAM bandwidth | 512 GB/s | P150 datasheet (brief) |
| Ridge point | HiFi2 648, HiFi4 324 FLOP / byte | peak / bandwidth |
| Practical ceilings | compute 50% of peak (HiFi2 165.9 TFLOP/s), memory 80% of bandwidth (410 GB/s); practical ridge HiFi2 405 FLOP / byte | person's rule of thumb |
| Bytes per element | bfp8 1.0625, bfp4 0.5625, bf16 2, fp32 4 | 1088 / 576 B per 32 x 32 tile |

## Op families (3 shapes)

Label = COMPUTE when FLOP / byte is above the ridge, else MEMORY. Practical limit = the larger of compute / 0.5 and
memory / 0.8. Between the practical ridge (405) and the ridge (648) a MEMORY family can hit the compute ceiling first.

**text 2048** (practical 12.374 ms)

| Op family | GFLOP | MB moved | FLOP / byte | Ridge | Label | Ideal ms | Practical ms | Practical limit |
|---|---|---|---|---|---|---|---|---|
| embedding/PLE/stem | 77.3 | 275.9 | 280 | 648 | MEMORY | 0.539 | 0.674 | memory 80 % |
| norms/eltwise/rotary | 0.0 | 2747.3 | 0 | 648 | MEMORY | 5.366 | 6.707 | memory 80 % |
| QKV/O proj | 171.8 | 480.8 | 357 | 648 | MEMORY | 0.939 | 1.174 | memory 80 % |
| attention | 291.8 | 335.5 | 870 | 648 | COMPUTE | 0.879 | 1.759 | compute 50 % |
| MLP | 309.2 | 835.2 | 370 | 648 | MEMORY | 1.631 | 2.039 | memory 80 % |
| head | 1.6 | 8.8 | 183 | 648 | MEMORY | 0.017 | 0.022 | memory 80 % |

**text 32768** (practical 393.709 ms)

| Op family | GFLOP | MB moved | FLOP / byte | Ridge | Label | Ideal ms | Practical ms | Practical limit |
|---|---|---|---|---|---|---|---|---|
| embedding/PLE/stem | 1237.0 | 4113.7 | 301 | 648 | MEMORY | 8.035 | 10.043 | memory 80 % |
| norms/eltwise/rotary | 0.0 | 43956.3 | 0 | 648 | MEMORY | 85.852 | 107.315 | memory 80 % |
| QKV/O proj | 2748.8 | 7023.9 | 391 | 648 | MEMORY | 13.719 | 17.148 | memory 80 % |
| attention | 37996.2 | 5368.7 | 7077 | 648 | COMPUTE | 114.524 | 229.048 | compute 50 % |
| MLP | 4947.8 | 12159.8 | 407 | 648 | MEMORY | 23.750 | 29.826 | compute 50 % |
| head | 25.8 | 134.6 | 191 | 648 | MEMORY | 0.263 | 0.329 | memory 80 % |

**vision 1x2560** (practical 12.276 ms)

| Op family | GFLOP | MB moved | FLOP / byte | Ridge | Label | Ideal ms | Practical ms | Practical limit |
|---|---|---|---|---|---|---|---|---|
| embedding/PLE/stem | 3.0 | 40.5 | 75 | 324 | MEMORY | 0.079 | 0.099 | memory 80 % |
| norms/eltwise/rotary | 0.0 | 2275.4 | 0 | 648 | MEMORY | 4.444 | 5.555 | memory 80 % |
| QKV/O proj | 193.3 | 417.6 | 463 | 648 | MEMORY | 0.816 | 1.165 | compute 50 % |
| attention | 322.1 | 251.7 | 1280 | 648 | COMPUTE | 0.971 | 1.942 | compute 50 % |
| MLP | 579.8 | 1064.0 | 545 | 648 | MEMORY | 2.078 | 3.495 | compute 50 % |
| head | 1.4 | 8.3 | 165 | 324 | MEMORY | 0.016 | 0.020 | memory 80 % |

## Why the old roofline was far

The old one (`../prefill/01-baseline/README.md`, `tt-ops/runs/embeddinggemma-2/roofline/`) is cold: it keeps every
activation on chip, so norms, eltwise, RoPE and residual traffic cost no DRAM time (here they are 54 % of text 2048),
and it takes one max(FPU, SFPU, DRAM) over the whole model, so weight reads hide under compute instead of adding up.
It also runs matmuls at 89 % of peak and DRAM at 450 of 512 GB/s, where the practical ceilings are 50 % and 80 %.

## Counting notes

| Item | How it is counted |
|---|---|
| Bytes | Each op reads its inputs and writes its outputs once (activations bf16); weights once in the precision config's dtype |
| Attention | Q, K, V in and O out once (scores on chip); FLOPs of Q.K^T and P.V only; softmax exp not counted |
| Sliding layers | Visible 32 x 32 blocks of the \|q - k\| <= 512 band (33 key tiles per query tile) |
| Audio attention | 32-query blocks against a 64-key context plus the relative-position Q.E^T, as the model evaluates it |
| Measured input | 3/4 valid; the sliding block map skips padding-only key blocks. The roofline counts the full bucket |
| Clock | Measured sustained AICLK fell from 4096 up (1143 MHz at 32768, 1087 at 65536); the roofline stays at 1.35 GHz |
| Short buckets | About 645 ops per text forward; per-op fixed cost is not in the roofline (text 128 and 256, audio 1x256) |
