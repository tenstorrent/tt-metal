# Tafsut round 0 analysis (paper, accuracy precision)

Profile job 955 (`profile/`, reduced to 2 layers, scaled by `layer_counts.json`): 1695.3 ms estimated against the
bench's 1697.8 ms (job 942). Embed 19.0 + layer 121.92 x13 + layer_last 83.7 (read-out cut already in) + head 7.7.
M = 1024 x 128 = 131072 rows. Matmuls are 78.5 % of the window at 11.3 % FLOPs utilization. Every matmul uses the auto
program config on DRAM-interleaved inputs, with `fp32_dest_acc_en=True` (tt/common.py:49-55).

| # | op | role | M x K x N | per layer | full ~ | util |
|---|---|---|---|---|---|---|
| 1 | batched matmul b=12288 x128x64x64, 8 cores, x2 | RoPE rotate_half via `self.rot` (model.py:184-188) | 128x64x64 | 45.5 ms | ~630 ms (37 %) | 5 % |
| 2 | matmul + separate relu | ff up wi (model.py:222) | 131072x768x3072 | 17.4 + 4.1 ms | ~285 ms | 12 % |
| 3 | matmul | ff down wo2 (model.py:224) | 131072x3072x768 | 16.4 ms | ~215 ms | 13 % |
| 4 | matmul HiFi4 | fused QKV (model.py:193) | 131072x768x2304 | 14.2 ms | ~197 ms | 22 % |
| 5 | 6 binary ops | RoPE mul/mul/add | - | 8.2 ms | ~113 ms | - |
| 6 | matmul | wo (model.py:217) | 131072x768x768 | 4.4 ms | ~60 ms | 12 % |
| 7 | NlpCreateHeads | head split | - | 3.1 ms | ~43 ms | - |
| 8 | SDPA | q/k chunk 128 | - | 2.9 ms | ~40 ms | - |
| 9 | 2 binary adds | residuals | 131072x768 | 2.7 ms | ~37 ms | - |
| 10 | 2 RMSNorm + concat_heads | | | 3.1 ms | ~42 ms | - |

Gaps: 0-1 us op to op inside sections, so device dispatch overhead is negligible. Host adds about 173 ms per call
(prepare 86, upload 58, postprocess 84 ms in bench.json); that is a later wall-time item.

Not applicable: fused gated MLP (MLP is relu, ungated); read-out cut (already done); SDPA chunking (already one chunk).
Analyst: read-only subagent (opus, low effort), written up here by the worker (PLAN §9 v1.1).
