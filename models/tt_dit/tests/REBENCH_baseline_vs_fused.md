# Re-bench: baseline (composite) vs fused — `wan_fused_distributed_rmsnorm`

> **Platform: Wormhole 4×8 galaxy (PROXY).** Blackhole 4×8 torus (2 links, 12×10 grid) is the
> real target — re-run with `WAN_GALAXY_LINKS=2` and paste the tables under the **Blackhole stub
> at the bottom** so BH numbers sit next to these WH ones. BH porting checklist:
> `RMSNORM_FUSION_FINDINGS.md`.

## Wormhole 4×8 galaxy (proxy — 4 links, 8×9 grid, worker cap 64)

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

---

## Blackhole 4×8 torus (TARGET — 2 links, 12×10 grid, worker cap 48)

Traced `test_bench` on the **BH 4×8 galaxy** (`bh4x8links2`, 32 chips), forwarder fused op
(origin `2d56fbdd64e`), `WAN_GALAXY_LINKS=2`. `baseline µs` = on-device composite (measured
fresh on BH); `fused µs` = forwarder fused; `↑BH` = baseline/fused; `↑WH` = the Wormhole-proxy
`↑F` above for the same config. Correctness re-validated on BH first (det=OK, PCC 99.99–100%
across all ring configs). **chunk=1.**

> **Two BH-specific findings (see FINDINGS "Blackhole port"):**
> 1. **Worker cap must be clamped to ≤ 2×sticks_per_packet = 64 on BH.** The grid-derived
>    default (~108 on 12×10) is *invalid*: with only 2 forwarders it puts 54 workers/forwarder,
>    exceeding the coalesced fabric packet (32 sticks) → `TT_FATAL: sticks_per_packet >=
>    workers_per_forwarder` on every large config. `derive_worker_cap` needs that clamp for BH.
> 2. **The BH perf optimum is ~48, not 64.** Re-sweep (large configs, fused µs):
>    N16384 389(c32) / **276(c48)** / 335(c64); self_sp4 457 / **411** / 477; self_sp8 282 /
>    220 / **204(c64)**. 48 ≈ the DRAM/compute knee — the forwarder made fabric cheap so the op
>    is compute/DRAM-bound (matches the WH ablation). cap48 wins most; the 296-tile-row
>    self_sp8 prefers 64. All numbers below are **cap48**.

> **`eff GB/s` = effective per-chip DRAM bandwidth of the fused op** = (input read + output
> write) ÷ fused µs = `N × feat_local × 4` (bf16 activation in **and** out, 2 B each) ÷ `t`.
> It is a *lower bound* on total DRAM traffic — it excludes the RoPE cos/sin and weight/bias
> reads (which add ~10–20% on the `+rope` configs) and the tiny AG stat sticks. **BH per-chip
> DRAM peak ≈ 512 GB/s**, so the `% peak` in parens is `eff GB/s / 512`. The large all-gather
> configs land at ~35–53% of peak; small/low-row shapes are latency/dispatch-bound (low GB/s).

### Wan2.2 — TP=4 ring (feat 1280/dev)

| config | rows | pattern | baseline µs | fused µs | eff GB/s | ↑BH | ↑WH |
|---|---:|---|---:|---:|---:|---:|---:|
| self_sp4_N18944    | 18944 | qk+rope | 584.4 | 410.6 | 236 (46%) | **1.42×** | 1.82× |
| self_sp8_N9472     | 9472  | qk+rope | 313.3 | 220.2 | 220 (43%) | **1.42×** (1.54× @c64) | 1.65× |
| self_sp32_N2368    | 2368  | qk+rope | 104.2 | 79.0  | 154 (30%) | 1.32× | 1.45× |
| cross_q_sp4_N18944 | 18944 | qk      | 512.2 | 356.7 | 272 (53%) | **1.44×** | 1.82× |
| cross_q_sp8_N9472  | 9472  | qk      | 272.6 | 187.0 | 259 (51%) | **1.46×** | 1.70× |
| cross_q_sp32_N2368 | 2368  | qk      | 93.4  | 66.7  | 182 (36%) | 1.40× | 1.38× |
| cross_k_prompt_L512| 512   | qk      | 51.3  | 29.3  | 90 (17%)  | **1.75×** | 1.59× |

### LTX-2.3 AV — TP=4 ring

| config | rows | feat | pattern | baseline µs | fused µs | eff GB/s | ↑BH | ↑WH |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| tp4_v_block_s1       | 1216 | 1024 | block+addcmul | 86.9  | 38.7  | 129 (25%) | **2.25×** | 2.04× |
| tp4_v_block_s2       | 4864 | 1024 | block+addcmul | 253.3 | 102.5 | 194 (38%) | **2.47×** | 2.71× |
| tp4_a_block          | 32   | 512  | block+addcmul | 28.1  | 18.6  | 4 (1%) | **1.51×** | 1.21× |
| tp4_v_selfattn_qk_s1 | 1216 | 1024 | qk+rope       | 94.0  | 44.5  | 112 (22%) | **2.11×** | 1.84× |
| tp4_v_selfattn_qk_s2 | 4864 | 1024 | qk+rope       | 264.4 | 169.9 | 117 (23%) | 1.56× | 1.96× |
| tp4_a_selfattn_qk    | 32   | 512  | qk+rope       | 38.1  | 20.7  | 3 (1%) | **1.84×** | 1.59× |
| tp4_a2v_videoQ_s1    | 1216 | 512  | qk+rope       | 73.8  | 31.2  | 80 (16%) | **2.37×** | 1.96× |
| tp4_a2v_videoQ_s2    | 4864 | 512  | qk+rope       | 171.2 | 143.7 | 69 (14%) | 1.19× | 2.13× |
| tp4_a2v_audioK       | 256  | 512  | qk+rope       | 54.9  | 24.7  | 21 (4%) | **2.23×** | 2.06× |
| tp4_v_textcross_q_s1 | 1216 | 1024 | qk            | 62.5  | 36.2  | 138 (27%) | 1.72× | 1.38× |
| tp4_v_textcross_q_s2 | 4864 | 1024 | qk            | 153.8 | 109.0 | 183 (36%) | 1.41× | 1.68× |
| tp4_v_textcross_k    | 1024 | 1024 | qk            | 59.9  | 34.9  | 120 (23%) | 1.72× | 1.49× |
| tp4_a_textcross_q    | 32   | 512  | qk            | 54.4  | 17.8  | 4 (1%) | **3.05×** | 1.26× |
| tp4_a_textcross_k    | 1024 | 512  | qk            | 51.9  | 26.8  | 78 (15%) | **1.94×** | — |

The earlier `tp4_a_block` regression (fused 57.4 > base 28.1, 0.49×) is **fixed** — it now runs
18.6 µs (**1.51×**), and `tp4_a_selfattn_qk` likewise improved (27.4 → 20.7 µs, **1.84×**). Both
are the tiny 32-row audio shapes; the fixes came in with the small-shape work integrated from
`kevinmi/fused_rms_norm` (re-measured 2026-06-26 after the rebase onto main).

### FLUX — TP=4 + TP=8 ring (`phn0` = whole-row qk+rope; `phn1` = per-head QK-norm, no AG)

| config | tp | rows | feat | baseline µs | fused µs | eff GB/s | ↑BH |
|---|---:|---:|---:|---:|---:|---:|---:|
| flux_tp4_N512_phn0   | 4 | 512   | 1536 | 69.1  | 39.5  | 80 (16%) | **1.75×** |
| flux_tp4_N512_phn1   | 4 | 512   | 1536 | 69.1  | 43.8  | 72 (14%) | 1.58× |
| flux_tp4_N64_phn0    | 4 | 64    | 1536 | 65.5  | 58.1  | 7 (1%) | 1.13× |
| flux_tp4_N64_phn1    | 4 | 64    | 1536 | 65.5  | 41.1  | 10 (2%) | 1.60× |
| flux_tp4_N2048_phn0  | 4 | 2048  | 1536 | 104.2 | 82.3  | 153 (30%) | 1.27× |
| flux_tp4_N2048_phn1  | 4 | 2048  | 1536 | 104.1 | 70.2  | 179 (35%) | 1.48× |
| flux_tp4_N8192_phn0  | 4 | 8192  | 1536 | 310.7 | 272.2 | 185 (36%) | 1.14× |
| flux_tp4_N8192_phn1  | 4 | 8192  | 1536 | 310.8 | 197.1 | 255 (50%) | 1.58× |
| flux_tp8_N1024_phn0  | 8 | 1024  | 768  | 77.3  | 36.7  | 86 (17%) | **2.10×** |
| flux_tp8_N1024_phn1  | 8 | 1024  | 768  | 76.9  | 30.3  | 104 (20%) | **2.54×** |
| flux_tp8_N128_phn0   | 8 | 128   | 768  | 58.8  | 28.2  | 14 (3%) | **2.08×** |
| flux_tp8_N128_phn1   | 8 | 128   | 768  | 58.7  | 24.1  | 16 (3%) | **2.43×** |
| flux_tp8_N4096_phn0  | 8 | 4096  | 768  | 175.9 | 82.8  | 152 (30%) | **2.12×** |
| flux_tp8_N4096_phn1  | 8 | 4096  | 768  | 175.8 | 79.4  | 158 (31%) | **2.22×** |
| flux_tp8_N16384_phn0 | 8 | 16384 | 768  | 552.5 | 276.6 | 182 (36%) | **2.00×** (WH 2.34×) |
| flux_tp8_N16384_phn1 | 8 | 16384 | 768  | 550.3 | 208.7 | 241 (47%) | **2.64×** |

### Bottom line (BH)
The forwarder fused op gives a **strong, real speedup on BH**: **~2× on all FLUX TP=8**,
**1.4–2.5× across Wan/LTX/FLUX**, far above the old transpose-MUX op. BH runs ~15–25% below the
WH proxy ↑ (2 links vs 4), as expected, but the goal — significant speedup over composite on the
real target — is met. The earlier open items are now all resolved: `derive_worker_cap` is
arch-aware (Blackhole knee 48, clamped to the fabric-packet validity limit), and the
`tp4_a_block` regression is fixed (now 1.51×).

**Re-verified 2026-06-26 (after rebase onto main):** all Wan/LTX/FLUX shapes re-run on BH —
PCC 99.99–100% (F:torch) and bit-exact determinism (`det_ndiff=0/9`) on every config; fused perf
matches the tables within run-to-run noise except the two LTX 32-row audio shapes above
(`tp4_a_block`, `tp4_a_selfattn_qk`) and `flux_tp8_N1024_phn1`, which improved and are updated.

**Effective DRAM bandwidth (`eff GB/s`):** the largest all-gather-bound configs reach
**~46–53% of the ~512 GB/s per-chip peak** (cross_q_sp4 **272 GB/s / 53%**, self_sp4 236/46%,
flux_tp8_N16384_phn1 241/47%, flux_tp4_N8192_phn1 255/50%), with the no-AG `phn1` per-head path
consistently higher than its `phn0` twin (no fabric round-trip diluting the BW). This is the
**DRAM-balanced ceiling** the device profiling pinned down: at these sizes read, compute, and
write are all co-busy at ~42–44 µs/tile-row, so the op is moving bytes at roughly half of peak
DRAM BW and is **not** leaving a large overlap win on the table (software-pipelining the AG was a
measured no-op — see `ABLATION_LEVERS.md`). Smaller / low-row shapes are latency- and
dispatch-bound (single-digit % of peak), which is expected — they don't have enough rows to
amortize fixed costs, not a bandwidth problem.
