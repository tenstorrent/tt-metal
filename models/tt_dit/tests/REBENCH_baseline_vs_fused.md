# Re-bench: baseline (composite) vs fused — `wan_fused_distributed_rmsnorm`

Traced bench (`test_bench`) on a Wormhole **4×8 galaxy**, branch `cglagovich/fused_rms_norm`.
RING = full 4×8 mesh, TP on a closed axis (4-wide for TP=4, 8-wide for TP=8), replicate the
other axis. 4 links. WAN bench = 100 iters, LTX/FLUX = 50 iters.

Columns:
- **`baseline µs`** — on-device composite reference (separate all-gather + rmsnorm ops),
  measured at `43841c2caaa`. Composite code unchanged since, so reused (it's flaky under
  traced bench on the heavy WAN N≥9472 configs, so not re-measured).
- **`fused-T µs`** — prior fused op, transpose-based gather/scatter + MUX all-gather
  (`0ad69fc8b9b`). Reference.
- **`fused-F⁽ʷ³²⁾ µs`** — fabric-forwarder all-gather (`15a76d8ee6d`) at the **old 32-worker
  cap**.
- **`fused-F⁽ʷ⁶⁴⁾ µs`** — same forwarder op at the **grid-derived worker cap** (`510cc544204`:
  workers = floor((grid − forwarders)/grid.x)·grid.x; **64** on this 8×9 galaxy). This is the
  **current op state**.
- **`w32→w64`** = the worker-cap effect = `fused-F⁽ʷ³²⁾ / fused-F⁽ʷ⁶⁴⁾ − 1`.
- **`↑F`** = `baseline / fused-F⁽ʷ⁶⁴⁾` = current forwarder fused speedup over composite.

The worker cap only affects the all-gather path (`phn0` / qk / block) with >32 tile-rows
(rows > 1024); **`phn1` (per-head norm) takes the local `is_tp_1` path — never 32-capped, so
`w32→w64` ≈ 0 there** (a clean control), as does any shape with ≤32 tile-rows.

## Wan2.2 — TP=4 (ring), feat 1280/dev, 10 heads, head_dim 128

| config | rows | pattern | baseline µs | fused-T µs | fused-F⁽ʷ³²⁾ µs | **fused-F⁽ʷ⁶⁴⁾ µs** | **w32→w64** | ↑F |
|---|---|---|---:|---:|---:|---:|---:|---:|
| self_sp4_N18944 | 18944 | qk+rope | 1153.62 | 896.13 | 836.13 | **634.28** | **+32%** | **1.82×** |
| self_sp8_N9472 | 9472 | qk+rope | 572.20 | 503.61 | 448.71 | **346.57** | **+29%** | **1.65×** |
| self_sp32_N2368 | 2368 | qk+rope | 187.10 | 191.09 | 153.58 | **129.48** | **+19%** | **1.45×** |
| cross_q_sp4_N18944 | 18944 | qk | 940.36 | 581.33 | 582.51 | **516.77** | **+13%** | **1.82×** |
| cross_q_sp8_N9472 | 9472 | qk | 472.45 | 333.42 | 315.94 | **278.48** | **+13%** | **1.70×** |
| cross_q_sp32_N2368 | 2368 | qk | 141.12 | 138.13 | 112.40 | **102.20** | **+10%** | **1.38×** |
| cross_k_prompt_L512 | 512 | qk | 73.36 | 67.00 | 46.24 | **46.20** | +0% | **1.59×** |

## LTX-2.3 AV — TP=4 (ring)

| config | rows | feat | heads | hd | pattern | baseline µs | fused-T µs | fused-F⁽ʷ³²⁾ µs | **fused-F⁽ʷ⁶⁴⁾ µs** | **w32→w64** | ↑F |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| tp4_v_block_s1 | 1216 | 1024 | 1 | – | block+addcmul | 139.59 | 110.23 | 75.00 | **68.55** | **+9%** | **2.04×** |
| tp4_v_block_s2 | 4864 | 1024 | 1 | – | block+addcmul | 432.91 | 203.93 | 173.85 | **159.56** | **+9%** | **2.71×** |
| tp4_a_block | 32 | 512 | 1 | – | block+addcmul | 33.63 | 24.68 | 27.82 | **27.68** | +1% | **1.21×** |
| tp4_v_selfattn_qk_s1 | 1216 | 1024 | 8 | 128 | qk+rope | 144.71 | 128.64 | 91.52 | **78.74** | **+16%** | **1.84×** |
| tp4_v_selfattn_qk_s2 | 4864 | 1024 | 8 | 128 | qk+rope | 453.47 | 254.88 | 223.33 | **231.43** | **-3%** | **1.96×** |
| tp4_a_selfattn_qk | 32 | 512 | 8 | 64 | qk+rope | 51.64 | 29.22 | 32.68 | **32.40** | +1% | **1.59×** |
| tp4_a2v_videoQ_s1 | 1216 | 512 | 8 | 64 | qk+rope | 106.51 | 104.43 | 69.03 | **54.40** | **+27%** | **1.96×** |
| tp4_a2v_videoQ_s2 | 4864 | 512 | 8 | 64 | qk+rope | 289.29 | 184.93 | 137.89 | **135.65** | +2% | **2.13×** |
| tp4_a2v_audioK | 256 | 512 | 8 | 64 | qk+rope | 79.67 | 58.01 | 38.75 | **38.62** | +0% | **2.06×** |
| tp4_v_textcross_q_s1 | 1216 | 1024 | 8 | 128 | qk | 88.66 | 100.83 | 66.96 | **64.28** | **+4%** | **1.38×** |
| tp4_v_textcross_q_s2 | 4864 | 1024 | 8 | 128 | qk | 246.58 | 184.81 | 153.28 | **146.40** | **+5%** | **1.68×** |
| tp4_v_textcross_k | 1024 | 1024 | 8 | 128 | qk | 82.48 | 85.37 | 54.92 | **55.18** | -0% | **1.49×** |
| tp4_a_textcross_q | 32 | 512 | 8 | 64 | qk | 32.37 | 22.46 | 25.61 | **25.60** | +0% | **1.26×** |
| tp4_a_textcross_k | 1024 | 512 | 8 | 64 | qk | 71.88 | 75.26 | 40.14 | **40.42** | -1% | **1.78×** |

## FLUX — TP=4 + TP=8 (ring), feat 1536 (TP4) / 768 (TP8)

| config | tp | rows | heads | pattern | baseline µs | fused-T µs | fused-F⁽ʷ³²⁾ µs | **fused-F⁽ʷ⁶⁴⁾ µs** | **w32→w64** | ↑F |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| flux_tp4_N512_phn0 | 4 | 512 | 12 | qk+rope | 117.96 | 91.85 | 69.84 | **69.72** | +0% | **1.69×** |
| flux_tp4_N512_phn1 | 4 | 512 | 12 | perhead+rope | 117.79 | 81.18 | 81.31 | **81.18** | +0% | **1.45×**\* |
| flux_tp4_N64_phn0 | 4 | 64 | 12 | qk+rope | 92.93 | 102.97 | 107.30 | **107.20** | +0% | **0.87×** |
| flux_tp4_N64_phn1 | 4 | 64 | 12 | perhead+rope | 92.46 | 102.91 | 79.83 | **79.40** | +1% | **1.16×**\* |
| flux_tp4_N2048_phn0 | 4 | 2048 | 12 | qk+rope | 175.20 | 167.41 | 135.42 | **124.55** | **+9%** | **1.41×** |
| flux_tp4_N2048_phn1 | 4 | 2048 | 12 | perhead+rope | 174.92 | 126.12 | 126.27 | **126.71** | -0% | **1.38×**\* |
| flux_tp4_N8192_phn0 | 4 | 8192 | 12 | qk+rope | 579.60 | 480.37 | 438.31 | **356.04** | **+23%** | **1.63×** |
| flux_tp4_N8192_phn1 | 4 | 8192 | 12 | perhead+rope | 577.75 | 345.77 | 345.54 | **346.33** | -0% | **1.67×**\* |
| flux_tp8_N1024_phn0 | 8 | 1024 | 6 | qk+rope | 122.94 | 108.23 | 61.02 | **60.95** | +0% | **2.02×** |
| flux_tp8_N1024_phn1 | 8 | 1024 | 6 | perhead+rope | 123.02 | 53.42 | 53.27 | **53.26** | +0% | **2.31×**\* |
| flux_tp8_N128_phn0 | 8 | 128 | 6 | qk+rope | 95.87 | 64.19 | 48.50 | **48.49** | +0% | **1.98×** |
| flux_tp8_N128_phn1 | 8 | 128 | 6 | perhead+rope | 95.83 | 44.70 | 44.73 | **44.83** | -0% | **2.14×**\* |
| flux_tp8_N4096_phn0 | 8 | 4096 | 6 | qk+rope | 279.12 | 250.86 | 163.15 | **134.88** | **+21%** | **2.07×** |
| flux_tp8_N4096_phn1 | 8 | 4096 | 6 | perhead+rope | 278.27 | 115.22 | 115.95 | **115.70** | +0% | **2.41×**\* |
| flux_tp8_N16384_phn0 | 8 | 16384 | 6 | qk+rope | 951.93 | 770.42 | 586.79 | **407.50** | **+44%** | **2.34×** |
| flux_tp8_N16384_phn1 | 8 | 16384 | 6 | perhead+rope | 959.65 | 366.43 | 366.69 | **367.86** | -0% | **2.61×**\* |

\* `phn1` baseline is the full-row composite (per-head norm has no apples-to-apples
composite); phn1 does not use the all-gather / worker-cap path.

## Notes
- **Worker cap (w32→w64) is a clean win on the large all-gather-bound shapes** — biggest where
  the gather dominates: flux_tp8_N16384 **+44%** (587→408µs), self_sp4 **+32%**, self_sp8 **+29%**,
  flux_tp4_N8192 **+23%**, flux_tp8_N4096 **+21%**, a2v_videoQ_s1 **+27%**. Small / ≤32-tile-row
  shapes are unchanged (≈0%), and one compute-bound shape (v_selfattn_qk_s2) gives back ~3%
  (more workers can't help where the bottleneck is the math).
- **vs composite (↑F):** the current op is **1.4–2.7×** on the all-gather shapes (and 2.0–2.6×
  on TP=8). Only the tiny 64-row flux_tp4_N64 sits at 0.87× — compute/dispatch-bound, fabric
  was never its bottleneck.
- **vs the prior fused-T:** the forwarder + worker cap together cut the large shapes by up to
  ~47% (flux_tp8_N16384 770→408µs).
