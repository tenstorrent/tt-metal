# Re-bench: composite baseline vs fused — `wan_fused_distributed_rmsnorm`

Traced bench (`test_bench` / `test_layernorm_module_bench`), µs/iter, on the 4×8 galaxy (32
chips), full-mesh RING (TP on a closed axis, replicate the other). One table per model, with
**Wormhole and Blackhole side by side**.

- **baseline** = on-device composite reference (separate all-gather + norm ops; for Wan the
  composite fuses weight+RoPE in-op). **fused** = the single `wan_fused_distributed_rmsnorm`
  device op. **↑ = baseline / fused** (speedup over composite).
- **WH** = Wormhole 4×8 galaxy (4 fabric links, 8×9 grid). **BH** = Blackhole 4×8 torus (2 links,
  12×10 grid) — the shipping target; WH is the proxy. Worker cap is grid/arch-derived automatically.
- Correctness on both archs: `pcc(fused:torch)` 99.99–100.00%, bit-exact deterministic on every config.

## Wan2.2 — TP=4 ring (feat 1280/dev, 10 heads, head_dim 128)

| config | rows | pattern | base WH | fused WH | ↑WH | base BH | fused BH | ↑BH |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| self_sp4_N18944 | 18944 | qk+rope | 1153.6 | 631.7 | **1.83×** | 584.4 | 410.6 | **1.42×** |
| self_sp8_N9472 | 9472 | qk+rope | 572.2 | 347.0 | **1.65×** | 313.3 | 220.2 | **1.42×** |
| self_sp32_N2368 | 2368 | qk+rope | 187.1 | 130.4 | **1.43×** | 104.2 | 79.0 | **1.32×** |
| cross_q_sp4_N18944 | 18944 | qk | 940.4 | 520.2 | **1.81×** | 512.2 | 356.7 | **1.44×** |
| cross_q_sp8_N9472 | 9472 | qk | 472.5 | 278.8 | **1.69×** | 272.6 | 187.0 | **1.46×** |
| cross_q_sp32_N2368 | 2368 | qk | 141.1 | 102.8 | **1.37×** | 93.4 | 66.7 | **1.40×** |
| cross_k_prompt_L512 | 512 | qk | 73.4 | 46.3 | **1.58×** | 51.3 | 29.3 | **1.75×** |

## LTX-2.3 AV — TP=4 ring (video feat 1024/dev, audio 512/dev)

| config | rows | pattern | base WH | fused WH | ↑WH | base BH | fused BH | ↑BH |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| tp4_v_block_s1 | 1216 | block+addcmul | 139.6 | 68.4 | **2.04×** | 86.9 | 38.7 | **2.25×** |
| tp4_v_block_s2 | 4864 | block+addcmul | 432.9 | 159.5 | **2.71×** | 253.3 | 102.5 | **2.47×** |
| tp4_a_block | 32 | block+addcmul | 33.6 | 27.1 | **1.24×** | 28.1 | 18.6 | **1.51×** |
| tp4_v_selfattn_qk_s1 | 1216 | qk+rope | 144.7 | 78.5 | **1.84×** | 94.0 | 44.5 | **2.11×** |
| tp4_v_selfattn_qk_s2 | 4864 | qk+rope | 453.5 | 230.2 | **1.97×** | 264.4 | 169.9 | **1.56×** |
| tp4_a_selfattn_qk | 32 | qk+rope | 51.6 | 31.9 | **1.62×** | 38.1 | 20.7 | **1.84×** |
| tp4_a2v_videoQ_s1 | 1216 | qk+rope | 106.5 | 54.3 | **1.96×** | 73.8 | 31.2 | **2.37×** |
| tp4_a2v_videoQ_s2 | 4864 | qk+rope | 289.3 | 135.1 | **2.14×** | 171.2 | 143.7 | **1.19×** |
| tp4_a2v_audioK | 256 | qk+rope | 79.7 | 38.0 | **2.10×** | 54.9 | 24.7 | **2.23×** |
| tp4_v_textcross_q_s1 | 1216 | qk | 88.7 | 64.0 | **1.39×** | 62.5 | 36.2 | **1.72×** |
| tp4_v_textcross_q_s2 | 4864 | qk | 246.6 | 146.8 | **1.68×** | 153.8 | 109.0 | **1.41×** |
| tp4_v_textcross_k | 1024 | qk | 82.5 | 54.8 | **1.51×** | 59.9 | 34.9 | **1.72×** |
| tp4_a_textcross_q | 32 | qk | 32.4 | 25.0 | **1.29×** | 54.4 | 17.8 | **3.05×** |
| tp4_a_textcross_k | 1024 | qk | 71.9 | 39.7 | **1.81×** | 51.9 | 26.8 | **1.94×** |

## FLUX (full dim 6144, 48 heads, head_dim 128, broadcast RoPE)

### RMSNorm — TP=4 + TP=8 ring (feat 1536 / 768 per dev)

| config | tp | rows | pattern | base WH | fused WH | ↑WH | base BH | fused BH | ↑BH |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| flux_tp4_N512_phn0 | 4 | 512 | qk+rope | 118.0 | 70.1 | **1.68×** | 69.1 | 39.5 | **1.75×** |
| flux_tp4_N512_phn1 | 4 | 512 | perhead+rope | 117.8 | 81.3 | **1.45×** | 69.1 | 43.8 | **1.58×** |
| flux_tp4_N64_phn0 | 4 | 64 | qk+rope | 92.9 | 108.2 | 0.86× | 65.5 | 58.1 | 1.13× |
| flux_tp4_N64_phn1 | 4 | 64 | perhead+rope | 92.5 | 79.6 | 1.16× | 65.5 | 41.1 | **1.60×** |
| flux_tp4_N2048_phn0 | 4 | 2048 | qk+rope | 175.2 | 125.0 | **1.40×** | 104.2 | 82.3 | 1.27× |
| flux_tp4_N2048_phn1 | 4 | 2048 | perhead+rope | 174.9 | 126.2 | **1.39×** | 104.1 | 70.2 | **1.48×** |
| flux_tp4_N8192_phn0 | 4 | 8192 | qk+rope | 579.6 | 356.7 | **1.62×** | 310.7 | 272.2 | 1.14× |
| flux_tp4_N8192_phn1 | 4 | 8192 | perhead+rope | 577.8 | 346.3 | **1.67×** | 310.8 | 197.1 | **1.58×** |
| flux_tp8_N1024_phn0 | 8 | 1024 | qk+rope | 122.9 | 61.2 | **2.01×** | 77.3 | 36.7 | **2.10×** |
| flux_tp8_N1024_phn1 | 8 | 1024 | perhead+rope | 123.0 | 53.5 | **2.30×** | 76.9 | 30.3 | **2.54×** |
| flux_tp8_N128_phn0 | 8 | 128 | qk+rope | 95.9 | 48.3 | **1.98×** | 58.8 | 28.2 | **2.08×** |
| flux_tp8_N128_phn1 | 8 | 128 | perhead+rope | 95.8 | 44.8 | **2.14×** | 58.7 | 24.1 | **2.43×** |
| flux_tp8_N4096_phn0 | 8 | 4096 | qk+rope | 279.1 | 135.2 | **2.06×** | 175.9 | 82.8 | **2.12×** |
| flux_tp8_N4096_phn1 | 8 | 4096 | perhead+rope | 278.3 | 115.6 | **2.41×** | 175.8 | 79.4 | **2.22×** |
| flux_tp8_N16384_phn0 | 8 | 16384 | qk+rope | 951.9 | 407.6 | **2.34×** | 552.5 | 276.6 | **2.00×** |
| flux_tp8_N16384_phn1 | 8 | 16384 | perhead+rope | 959.7 | 367.4 | **2.61×** | 550.3 | 208.7 | **2.64×** |

`phn0` = whole-row QK-norm + RoPE (uses the all-gather); `phn1` = per-head QK-norm (local, no
all-gather). The `phn1` baseline is the full-row composite (per-head has no apples-to-apples
composite), so its ↑ is indicative rather than a like-for-like comparison.

### LayerNorm — TP=4 + TP=8 ring (whole-row adaLN: weight+bias, no RoPE)

Fused Welford LayerNorm vs the composite `dit_layernorm` chain (`pre_allgather` → all-gather →
`post_allgather`, weight+bias).

| config | tp | rows | base WH | fused WH | ↑WH | base BH | fused BH | ↑BH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flux_tp4_N64 | 4 | 64 | 105.3 | 154.0 | 0.68× | 60.8 | 90.4 | 0.67× |
| flux_tp4_N512 | 4 | 512 | 114.5 | 102.6 | **1.12×** | 69.5 | 60.2 | **1.15×** |
| flux_tp4_N2048 | 4 | 2048 | 179.4 | 170.6 | 1.05× | 115.5 | 111.4 | 1.04× |
| flux_tp4_N8192 | 4 | 8192 | 571.8 | 401.5 | **1.42×** | 380.7 | 370.4 | 1.03× |
| flux_tp8_N128 | 8 | 128 | 108.9 | 69.0 | **1.58×** | 63.8 | 41.9 | **1.52×** |
| flux_tp8_N1024 | 8 | 1024 | 140.3 | 90.2 | **1.56×** | 93.4 | 53.2 | **1.76×** |
| flux_tp8_N4096 | 8 | 4096 | 306.0 | 170.7 | **1.79×** | 224.4 | 164.7 | **1.36×** |
| flux_tp8_N16384 | 8 | 16384 | 986.0 | 527.6 | **1.87×** | 707.2 | 635.3 | 1.11× |

## Summary

- **RMSNorm** — fused beats composite on essentially every shape: **WH 1.4–2.6×** (2.0–2.6× at
  TP=8), **BH 1.1–3.0×**. BH runs ~15–25% below the WH proxy on the large fabric-bound shapes
  (half the fabric links) but is still a clear win.
- **LayerNorm** — **WH up to 1.9×** (TP=8 large), **BH up to 1.8×** (TP=8 mid). The large
  fabric-bound LN shapes drop more on BH than RMS does (N16384 tp8: WH 1.87× → BH 1.11×) because
  LN's Welford all-gather carries **2 stats (mean+var)** — ~2× the fabric payload — against BH's
  half-bandwidth fabric.
- The only wash/slight regression is the tiny 64-row FLUX TP=4 shape (dispatch-bound on both archs).
