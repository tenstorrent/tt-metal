# Re-bench: baseline (composite) vs fused — `wan_fused_distributed_rmsnorm`

Fresh traced bench (100 iters, baseline+fused) on a Wormhole **4×8 galaxy**, branch
`cglagovich/fused_rms_norm` @ `43841c2caaa` (post cleanup). RING = full 4×8 mesh, TP
on a closed axis (4-wide for TP=4, 8-wide for TP=8), replicate the other axis. 4 links.
`baseline µs` = on-device composite reference; `fused µs` = the fused op; `↑` = baseline/fused.
Generated via `test_bench` (`RMS_BENCH_METHODS=baseline,fused`).

## Wan2.2 — TP=4 (ring), feat 1280/dev, 10 heads, head_dim 128

| config | rows | pattern | baseline µs | fused µs | ↑ |
|---|---:|---|---:|---:|---:|
| self_sp4_N18944    | 18944 | qk+rope | 1153.62 | 896.13 | **1.29×** |
| self_sp8_N9472     | 9472  | qk+rope | 572.20  | 503.61 | **1.14×** |
| self_sp32_N2368    | 2368  | qk+rope | 187.10  | 191.09 | 0.98× |
| cross_q_sp4_N18944 | 18944 | qk      | 940.36  | 581.33 | **1.62×** |
| cross_q_sp8_N9472  | 9472  | qk      | 472.45  | 333.42 | **1.42×** |
| cross_q_sp32_N2368 | 2368  | qk      | 141.12  | 138.13 | 1.02× |
| cross_k_prompt_L512| 512   | qk      | 73.36   | 67.00  | **1.09×** |

## LTX-2.3 AV — TP=4 (ring)

| config | rows | feat | heads | hd | pattern | baseline µs | fused µs | ↑ |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| tp4_v_block_s1       | 1216 | 1024 | 1 | –   | block+addcmul | 139.59 | 110.23 | **1.27×** |
| tp4_v_block_s2       | 4864 | 1024 | 1 | –   | block+addcmul | 432.91 | 203.93 | **2.12×** |
| tp4_a_block          | 32   | 512  | 1 | –   | block+addcmul | 33.63  | 24.68  | **1.36×** |
| tp4_v_selfattn_qk_s1 | 1216 | 1024 | 8 | 128 | qk+rope       | 144.71 | 128.64 | **1.12×** |
| tp4_v_selfattn_qk_s2 | 4864 | 1024 | 8 | 128 | qk+rope       | 453.47 | 254.88 | **1.78×** |
| tp4_a_selfattn_qk    | 32   | 512  | 8 | 64  | qk+rope       | 51.64  | 29.22  | **1.77×** |
| tp4_a2v_videoQ_s1    | 1216 | 512  | 8 | 64  | qk+rope       | 106.51 | 104.43 | 1.02× |
| tp4_a2v_videoQ_s2    | 4864 | 512  | 8 | 64  | qk+rope       | 289.29 | 184.93 | **1.56×** |
| tp4_a2v_audioK       | 256  | 512  | 8 | 64  | qk+rope       | 79.67  | 58.01  | **1.37×** |
| tp4_v_textcross_q_s1 | 1216 | 1024 | 8 | 128 | qk            | 88.66  | 100.83 | 0.88× |
| tp4_v_textcross_q_s2 | 4864 | 1024 | 8 | 128 | qk            | 246.58 | 184.81 | **1.33×** |
| tp4_v_textcross_k    | 1024 | 1024 | 8 | 128 | qk            | 82.48  | 85.37  | 0.97× |
| tp4_a_textcross_q    | 32   | 512  | 8 | 64  | qk            | 32.37  | 22.46  | **1.44×** |
| tp4_a_textcross_k    | 1024 | 512  | 8 | 64  | qk            | 71.88  | 75.26  | 0.95× |

## FLUX — TP=4 + TP=8 (ring), feat 1536 (TP4) / 768 (TP8)

`phn0` = whole-row RMSNorm (qk+rope). `phn1` = FLUX.2 per-head QK-norm. **Caveat:** the
composite baseline can't do per-head norm, so `phn1` rows benchmark the fused per-head op
against the *full-row-norm* composite of the same shape — a **relative-cost** comparison,
not apples-to-apples (the fused phn1 also skips the all-gather, so its big speedups partly
reflect that).

| config | tp | rows | heads | pattern | baseline µs | fused µs | ↑ |
|---|---:|---:|---:|---|---:|---:|---:|
| flux_tp4_N512_phn0   | 4 | 512   | 12 | qk+rope        | 117.96 | 91.85  | **1.28×** |
| flux_tp4_N512_phn1   | 4 | 512   | 12 | perhead+rope   | 117.79 | 81.18  | **1.45×**\* |
| flux_tp4_N64_phn0    | 4 | 64    | 12 | qk+rope        | 92.93  | 102.97 | 0.90× |
| flux_tp4_N64_phn1    | 4 | 64    | 12 | perhead+rope   | 92.46  | 102.91 | 0.90×\* |
| flux_tp4_N2048_phn0  | 4 | 2048  | 12 | qk+rope        | 175.20 | 167.41 | **1.05×** |
| flux_tp4_N2048_phn1  | 4 | 2048  | 12 | perhead+rope   | 174.92 | 126.12 | **1.39×**\* |
| flux_tp4_N8192_phn0  | 4 | 8192  | 12 | qk+rope        | 579.60 | 480.37 | **1.21×** |
| flux_tp4_N8192_phn1  | 4 | 8192  | 12 | perhead+rope   | 577.75 | 345.77 | **1.67×**\* |
| flux_tp8_N1024_phn0  | 8 | 1024  | 6  | qk+rope        | 122.94 | 108.23 | **1.14×** |
| flux_tp8_N1024_phn1  | 8 | 1024  | 6  | perhead+rope   | 123.02 | 53.42  | **2.30×**\* |
| flux_tp8_N128_phn0   | 8 | 128   | 6  | qk+rope        | 95.87  | 64.19  | **1.49×** |
| flux_tp8_N128_phn1   | 8 | 128   | 6  | perhead+rope   | 95.83  | 44.70  | **2.14×**\* |
| flux_tp8_N4096_phn0  | 8 | 4096  | 6  | qk+rope        | 279.12 | 250.86 | **1.11×** |
| flux_tp8_N4096_phn1  | 8 | 4096  | 6  | perhead+rope   | 278.27 | 115.22 | **2.42×**\* |
| flux_tp8_N16384_phn0 | 8 | 16384 | 6  | qk+rope        | 951.93 | 770.42 | **1.24×** |
| flux_tp8_N16384_phn1 | 8 | 16384 | 6  | perhead+rope   | 959.65 | 366.43 | **2.62×**\* |

\* `phn1` baseline is the full-row composite (see caveat above).

## Notes

- Fused wins biggest on large-token configs (Wan cross_q_sp4 **1.62×**, LTX v_block_s2
  **2.12×**, FLUX phn1 up to **2.62×**). Small/dispatch-bound shapes (N≤512, 32-row audio,
  no-RoPE text-cross-k) sit near or just under 1.0× — the fused op has little to amortize there.
- Wan TP=4 fully benched here, including the heavy N18944 self/cross configs (these had
  timed out in earlier full-batch bench runs; re-running Wan in its own per-model batch let
  them complete).
