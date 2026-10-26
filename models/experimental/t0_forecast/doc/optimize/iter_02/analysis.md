# Round 2 analysis (iter_02, HEAD 328675dac80; jobs 573 paper bench + profile, 575 multivariate profile)

Written from the read-only analyst's report (Opus, low effort). The numbers come from summary.md, perf_report.txt and sections/.

## Paper / accuracy
- Device 541.8 ms measured against a 541.4 ms reduced-profile estimate (−0.1 %). Steady wall 685.1 ms, pcc_normalized 0.99997.
- Per section (ms per layer x count):

  | section | per layer | count | total |
  |---|---|---|---|
  | time layer | 28.35 | 15 | 425.2 |
  | group layer (G = 1, folded) | 12.41 | 7 | 86.9 |
  | cut layer | | | 20.5 |
  | embed | | | 7.8 |
  | head | | | 0.6 |
  | last layer | | | 0.4 |

- Matmuls are about 53 % of device time, at a weighted 29.1 % of peak FLOPs. Non-matmul ops are about 47 %: LayerNorm 15.3 %, RoPE 12.2 %, NlpCreateHeads 7.8 %, SDPA 5.6 %.
- Op-to-op gaps are negligible: about 0.15 ms in total, plus 1-2 ms at section starts.

Top ops (scaled):

| # | op (role) | ms x count | total ms | utilization |
|---|---|---|---|---|
| 1 | fused SwiGLU ff1 98304x1024x4096 | 6.32 x 23 | 145 | 42.9 % FLOPs |
| 2 | RoPE q + k | 2 x 2.14 x 16 | 68 | about 190 GB/s |
| 3 | QKV minimal_matmul | 4.23 x 16 | 68 | 48 % |
| 4 | ff2, residual epilogue | 2.36 x 23 | 54 | 57.5 % |
| 5 | pre-norm RMSNorm | 1.01 x 46 | 47 | about 400 GB/s |
| 6 | q/k RMSNorm (hd 64) | 2 x 1.39 x 16 | 44 | about 145 GB/s |
| 7 | nlp_create_qkv_heads | 2.73 x 16 | 44 | |
| 8 | wo / wvo, residual epilogue | 1.67 x 23 | 38 | 40.5 % |
| 9 | SDPA, time, causal, S = 96 | 1.95 x 16 | 31 | about 80 % of BW |
| 10 | embed matmuls (HiFi4) | once | 7.8 | |

## Multivariate (128 x 8 variates)
- Estimate 1329.7 ms. Group layers take 124.15 ms x 7 = 869 ms (65 %).
- One group layer:
  - The slot-major permute is 33.1 ms. The reshape is 21.8 ms. fill_implicit_tile_padding is 12.6 ms. The return reshape and permute are 10.8 ms. Together these come to about 78 ms of TM ops.
  - Padding G = 8 up to 32 makes heads, q/k norms and SDPA about 4x more expensive, which adds about 29 ms.
  - The matmuls are only 14.6 ms.
