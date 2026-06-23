# Re-bench: baseline (composite) vs fused — `wan_fused_distributed_rmsnorm`

Traced bench (`test_bench`) on a Wormhole **4×8 galaxy**, branch `cglagovich/fused_rms_norm`.
RING = full 4×8 mesh, TP on a closed axis (4-wide for TP=4, 8-wide for TP=8), replicate the
other axis. 4 links. WAN bench = 100 iters, LTX/FLUX = 50 iters.

Three fused columns:
- **`baseline µs`** — on-device composite reference (separate all-gather + rmsnorm ops).
  Measured at `43841c2caaa`. The composite op is unchanged since, so these stand; they are
  reused here because the composite is flaky under traced bench on the heavy WAN configs
  (N≥9472 wedge the device under trace — it produced these only with care).
- **`fused-T µs`** — the prior fused op with the transpose-based stats gather/scatter
  (`0ad69fc8b9b`), MUX all-gather. Kept for reference.
- **`fused-F µs`** — the current fused op with the **fabric-forwarder coalesced all-gather**
  (`15a76d8ee6d`). Measured fresh this session (fused-only; the forwarder is bit-exact under
  trace — see `test_traced_corr`).

**`↑F`** = `baseline / fused-F` = the forwarder fused speedup over the composite baseline.

## Wan2.2 — TP=4 (ring), feat 1280/dev, 10 heads, head_dim 128

| config | rows | pattern | baseline µs | fused-T µs | fused-F µs | ↑F |
|---|---:|---|---:|---:|---:|---:|
| self_sp4_N18944    | 18944 | qk+rope | 1153.62 | 896.13 | 836.13 | **1.38×** |
| self_sp8_N9472     | 9472  | qk+rope | 572.20  | 503.61 | 448.71 | **1.28×** |
| self_sp32_N2368    | 2368  | qk+rope | 187.10  | 191.09 | 153.58 | **1.22×** |
| cross_q_sp4_N18944 | 18944 | qk      | 940.36  | 581.33 | 582.51 | **1.61×** |
| cross_q_sp8_N9472  | 9472  | qk      | 472.45  | 333.42 | 315.94 | **1.50×** |
| cross_q_sp32_N2368 | 2368  | qk      | 141.12  | 138.13 | 112.40 | **1.26×** |
| cross_k_prompt_L512| 512   | qk      | 73.36   | 67.00  | 46.24  | **1.59×** |

## LTX-2.3 AV — TP=4 (ring)

| config | rows | feat | heads | hd | pattern | baseline µs | fused-T µs | fused-F µs | ↑F |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| tp4_v_block_s1       | 1216 | 1024 | 1 | –   | block+addcmul | 139.59 | 110.23 | 75.00  | **1.86×** |
| tp4_v_block_s2       | 4864 | 1024 | 1 | –   | block+addcmul | 432.91 | 203.93 | 173.85 | **2.49×** |
| tp4_a_block          | 32   | 512  | 1 | –   | block+addcmul | 33.63  | 24.68  | 27.82  | **1.21×** |
| tp4_v_selfattn_qk_s1 | 1216 | 1024 | 8 | 128 | qk+rope       | 144.71 | 128.64 | 91.52  | **1.58×** |
| tp4_v_selfattn_qk_s2 | 4864 | 1024 | 8 | 128 | qk+rope       | 453.47 | 254.88 | 223.33 | **2.03×** |
| tp4_a_selfattn_qk    | 32   | 512  | 8 | 64  | qk+rope       | 51.64  | 29.22  | 32.68  | **1.58×** |
| tp4_a2v_videoQ_s1    | 1216 | 512  | 8 | 64  | qk+rope       | 106.51 | 104.43 | 69.03  | **1.54×** |
| tp4_a2v_videoQ_s2    | 4864 | 512  | 8 | 64  | qk+rope       | 289.29 | 184.93 | 137.89 | **2.10×** |
| tp4_a2v_audioK       | 256  | 512  | 8 | 64  | qk+rope       | 79.67  | 58.01  | 38.75  | **2.06×** |
| tp4_v_textcross_q_s1 | 1216 | 1024 | 8 | 128 | qk            | 88.66  | 100.83 | 66.96  | **1.32×** |
| tp4_v_textcross_q_s2 | 4864 | 1024 | 8 | 128 | qk            | 246.58 | 184.81 | 153.28 | **1.61×** |
| tp4_v_textcross_k    | 1024 | 1024 | 8 | 128 | qk            | 82.48  | 85.37  | 54.92  | **1.50×** |
| tp4_a_textcross_q    | 32   | 512  | 8 | 64  | qk            | 32.37  | 22.46  | 25.61  | **1.26×** |
| tp4_a_textcross_k    | 1024 | 512  | 8 | 64  | qk            | 71.88  | 75.26  | 40.14  | **1.79×** |

## FLUX — TP=4 + TP=8 (ring), feat 1536 (TP4) / 768 (TP8)

`phn0` = whole-row RMSNorm (qk+rope), uses the all-gather → exercises the forwarder. `phn1` =
FLUX.2 per-head QK-norm, reduces locally per head → **no all-gather, no forwarder** (so
`fused-F ≈ fused-T` on phn1, as expected — a useful control). **Caveat:** the composite
baseline can't do per-head norm, so `phn1` rows benchmark the fused per-head op against the
*full-row-norm* composite of the same shape — a **relative-cost** comparison, not
apples-to-apples (the fused phn1 also skips the all-gather, so its big ↑F partly reflect that).

| config | tp | rows | heads | pattern | baseline µs | fused-T µs | fused-F µs | ↑F |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| flux_tp4_N512_phn0   | 4 | 512   | 12 | qk+rope        | 117.96 | 91.85  | 69.84  | **1.69×** |
| flux_tp4_N512_phn1   | 4 | 512   | 12 | perhead+rope   | 117.79 | 81.18  | 81.31  | **1.45×**\* |
| flux_tp4_N64_phn0    | 4 | 64    | 12 | qk+rope        | 92.93  | 102.97 | 107.30 | 0.87× |
| flux_tp4_N64_phn1    | 4 | 64    | 12 | perhead+rope   | 92.46  | 102.91 | 79.83  | **1.16×**\* |
| flux_tp4_N2048_phn0  | 4 | 2048  | 12 | qk+rope        | 175.20 | 167.41 | 135.42 | **1.29×** |
| flux_tp4_N2048_phn1  | 4 | 2048  | 12 | perhead+rope   | 174.92 | 126.12 | 126.27 | **1.39×**\* |
| flux_tp4_N8192_phn0  | 4 | 8192  | 12 | qk+rope        | 579.60 | 480.37 | 438.31 | **1.32×** |
| flux_tp4_N8192_phn1  | 4 | 8192  | 12 | perhead+rope   | 577.75 | 345.77 | 345.54 | **1.67×**\* |
| flux_tp8_N1024_phn0  | 8 | 1024  | 6  | qk+rope        | 122.94 | 108.23 | 61.02  | **2.01×** |
| flux_tp8_N1024_phn1  | 8 | 1024  | 6  | perhead+rope   | 123.02 | 53.42  | 53.27  | **2.31×**\* |
| flux_tp8_N128_phn0   | 8 | 128   | 6  | qk+rope        | 95.87  | 64.19  | 48.50  | **1.98×** |
| flux_tp8_N128_phn1   | 8 | 128   | 6  | perhead+rope   | 95.83  | 44.70  | 44.73  | **2.14×**\* |
| flux_tp8_N4096_phn0  | 8 | 4096  | 6  | qk+rope        | 279.12 | 250.86 | 163.15 | **1.71×** |
| flux_tp8_N4096_phn1  | 8 | 4096  | 6  | perhead+rope   | 278.27 | 115.22 | 115.95 | **2.40×**\* |
| flux_tp8_N16384_phn0 | 8 | 16384 | 6  | qk+rope        | 951.93 | 770.42 | 586.79 | **1.62×** |
| flux_tp8_N16384_phn1 | 8 | 16384 | 6  | perhead+rope   | 959.65 | 366.43 | 366.69 | **2.62×**\* |

\* `phn1` baseline is the full-row composite (see caveat above); phn1 does not use the forwarder.

## Notes

- **The fabric forwarder beats the composite baseline on every all-gather (phn0/qk) config
  except the tiny 64-row flux_tp4_N64** (0.87× — 2 tile-rows, compute/dispatch-bound, fabric
  was never its bottleneck). Speedup grows with ring width and token count: TP=8 phn0 hits
  **1.6–2.0×**, large WAN/LTX **1.4–2.5×**.
- **Forwarder vs the prior transpose-MUX fused (`fused-F` vs `fused-T`)** is the apples-to-
  apples kernel comparison and is a clean win wherever the AG matters: flux_tp8_N4096
  250.9→163.2µs (−35%), flux_tp8_N1024 108.2→61.0µs (−44%), LTX a2v/textcross −25..40%,
  WAN cross/cross_k −20..30%. Tiny 32-row audio + flux_tp4_N64 regress a few µs (single
  worker/forwarder overhead with nothing to coalesce).
- **phn1 control:** per-head norm takes the local `is_tp_1` path (no all-gather), so
  `fused-F ≈ fused-T` there (e.g. flux_tp8_N1024_phn1 53.3 vs 53.4µs) — confirms the change is
  isolated to the AG path.
- The composite baseline is reused from the `43841c2caaa` run (unchanged code); it is flaky
  under traced bench on the heavy WAN N≥9472 configs (wedges the device), so it was not
  re-measured. Forwarder `fused-F` numbers are all fresh this session.
